"""Paired signal metrics with a noisy bypass baseline; not paper-quality claims."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .data import PairedAudioDataset, SyntheticDataset
from .infer import enhance_waveform, load_model


METRIC_IMPLEMENTATIONS = {
    "mse": "Mean squared sample error in original waveform amplitude; lower is better.",
    "snr_db": "10*log10(mean(reference^2)/mean((estimate-reference)^2)); no gain/time alignment.",
    "si_sdr_db": "Zero-mean utterance SI-SDR, least-squares scalar projection onto reference; no time alignment.",
    "si_sdr_improvement_db": "Per-utterance enhanced SI-SDR minus noisy-bypass SI-SDR.",
    "aggregation": "Unweighted mean across utterances; undefined values are excluded and counts are reported.",
    "numerics": "Float64 arithmetic, power floor 1e-12. Reference RMS <=1e-8 gives undefined SNR/SI-SDR; silent estimate gives undefined SI-SDR.",
}


def signal_metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Measure equally sized, already aligned mono waveforms without rescaling."""
    if estimate.ndim != 1 or reference.ndim != 1 or estimate.shape != reference.shape or reference.numel() == 0:
        raise ValueError("metrics require nonempty, equally sized 1D waveforms")
    estimate = estimate.detach().to(device="cpu", dtype=torch.float64)
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(estimate).all() and torch.isfinite(reference).all()):
        raise ValueError("metrics require finite samples")
    reference_power = float(reference.square().mean())
    estimate_power = float(estimate.square().mean())
    mse = float((estimate - reference).square().mean())
    silent_reference = reference_power <= 1e-16
    snr = None if silent_reference else 10 * math.log10(max(reference_power, 1e-12) / max(mse, 1e-12))
    centered_reference = reference - reference.mean()
    centered_estimate = estimate - estimate.mean()
    centered_power = float(centered_reference.square().mean())
    si_sdr = None
    if centered_power > 1e-16 and float(centered_estimate.square().mean()) > 1e-16:
        scale = (centered_estimate * centered_reference).sum() / centered_reference.square().sum()
        projected = scale * centered_reference
        residual = centered_estimate - projected
        si_sdr = 10 * math.log10(max(float(projected.square().mean()), 1e-12) / max(float(residual.square().mean()), 1e-12))
    return {
        "mse": mse,
        "snr_db": snr,
        "si_sdr_db": si_sdr,
        "reference_rms": math.sqrt(reference_power),
        "estimate_rms": math.sqrt(estimate_power),
        "silent_reference": silent_reference,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report valid counts rather than treating undefined metrics as zero."""
    result: dict[str, Any] = {"utterances": len(rows)}
    for name in ("mse", "snr_db", "si_sdr_db", "si_sdr_improvement_db", "reference_rms", "estimate_rms"):
        values = [row[name] for row in rows if row.get(name) is not None]
        if values or any(name in row for row in rows):
            result[name] = {"mean": sum(values) / len(values) if values else None, "valid_utterances": len(values)}
    result["silent_reference_utterances"] = sum(bool(row.get("silent_reference")) for row in rows)
    return result


@torch.inference_mode()
def evaluate_dataset(
    model: Any,
    dataset: Any,
    exits: tuple[int, ...] | list[int] | None = None,
    streaming: bool = False,
    chunk_samples: int | None = None,
    check_clean: bool = False,
) -> dict[str, Any]:
    exits = tuple(model.config.exits) if exits is None else tuple(exits)
    if not exits or any(depth not in model.config.exits for depth in exits):
        raise ValueError(f"evaluation exits must be drawn from {model.config.exits}")
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        length = int(item.get("length", item["clean"].numel()))
        noisy, clean = item["noisy"][:length], item["clean"][:length]
        bypass = signal_metrics(noisy, clean)
        record: dict[str, Any] = {"id": str(item.get("id", index)), "samples": length, "bypass": bypass, "exits": {}}
        if check_clean:
            record["clean_input_preservation"] = {}
        for depth in exits:
            enhanced = enhance_waveform(model, noisy, depth, streaming, chunk_samples)[0]
            measured = signal_metrics(enhanced, clean)
            measured["si_sdr_improvement_db"] = (
                measured["si_sdr_db"] - bypass["si_sdr_db"]
                if measured["si_sdr_db"] is not None and bypass["si_sdr_db"] is not None else None
            )
            record["exits"][str(depth)] = measured
            if check_clean:
                clean_enhanced = enhance_waveform(model, clean, depth, streaming, chunk_samples)[0]
                record["clean_input_preservation"][str(depth)] = signal_metrics(clean_enhanced, clean)
        records.append(record)
    if not records:
        raise ValueError("evaluation requires at least one example")
    summary: dict[str, Any] = {
        "bypass": summarize_metrics([record["bypass"] for record in records]),
        "exits": {str(depth): summarize_metrics([record["exits"][str(depth)] for record in records]) for depth in exits},
    }
    if check_clean:
        summary["clean_input_preservation"] = {
            str(depth): summarize_metrics([record["clean_input_preservation"][str(depth)] for record in records])
            for depth in exits
        }
    return {
        "sample_rate": model.config.sample_rate,
        "inference_mode": "streaming" if streaming else "whole_waveform_causal_model",
        "exits": list(exits),
        "metric_implementations": METRIC_IMPLEMENTATIONS,
        "limitations": "Signal metrics only. No PESQ, STOI, perceptual MOS, ASR/content, speaker identity, or listening test is measured. These results do not establish paper parity or production readiness.",
        "summary": summary,
        "examples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--manifest")
    data.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exits", help="Comma-separated configured exit depths; default: all configured exits")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--check-clean", action="store_true")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    dataset = (
        SyntheticDataset(model.config.sample_rate, model.config.sample_rate // 5, length=2)
        if args.smoke else PairedAudioDataset(args.manifest, model.config.sample_rate, training=False, load_teacher=False)
    )
    exits = tuple(int(value.strip()) for value in args.exits.split(",")) if args.exits else None
    report = evaluate_dataset(model, dataset, exits, args.stream, args.chunk_samples, args.check_clean)
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["data_kind"] = "synthetic_smoke_not_quality_evidence" if args.smoke else "user_supplied_paired_manifest"
    if args.manifest:
        report["manifest"] = str(Path(args.manifest).resolve())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved signal metrics for {len(dataset)} examples to {destination}")


if __name__ == "__main__":
    main()
