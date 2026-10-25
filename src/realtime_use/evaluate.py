"""Paired signal and optional published quality metrics with a noisy baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
import torchmetrics
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, signal_noise_ratio

from .data import PairedAudioDataset, SyntheticDataset
from .infer import enhance_waveform, load_model
from .quality_metrics import QualityEvaluator


METRIC_IMPLEMENTATIONS = {
    "revision": 2,
    "versions": {"torch": str(torch.__version__), "torchmetrics": torchmetrics.__version__},
    "mse": "torch.nn.functional.mse_loss(reduction='mean'); original waveform amplitude, lower is better.",
    "snr_db": "torchmetrics.functional.audio.signal_noise_ratio(zero_mean=False); no gain/time alignment.",
    "si_sdr_db": "torchmetrics.functional.audio.scale_invariant_signal_distortion_ratio(zero_mean=True); no time alignment.",
    "si_sdr_improvement_db": "Per-utterance enhanced SI-SDR minus noisy-bypass SI-SDR.",
    "aggregation": "Unweighted mean across utterances; undefined values are excluded and counts are reported.",
    "numerics": "CPU float64 with TorchMetrics dtype-epsilon regularization of summed energies. Replaces the earlier custom 1e-12 mean-power floor; results can differ, especially for perfect or very quiet signals. Nonfinite inputs or computed values raise an error.",
    "undefined_values": "Project reporting policy: reference RMS <=1e-8 gives null SNR; centered reference or centered estimate RMS <=1e-8 gives null SI-SDR. Null values are excluded from aggregates.",
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
    mse = float(F.mse_loss(estimate, reference, reduction="mean"))
    silent_reference = reference_power <= 1e-16
    snr = None if silent_reference else float(signal_noise_ratio(estimate, reference, zero_mean=False))
    si_sdr = None
    # Keep undefined silence/DC reporting separate from the library formula.
    if float(reference.var(unbiased=False)) > 1e-16 and float(estimate.var(unbiased=False)) > 1e-16:
        si_sdr = float(scale_invariant_signal_distortion_ratio(estimate, reference, zero_mean=True))
    result = {
        "mse": mse,
        "snr_db": snr,
        "si_sdr_db": si_sdr,
        "reference_rms": math.sqrt(reference_power),
        "estimate_rms": math.sqrt(estimate_power),
        "silent_reference": silent_reference,
    }
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError("metrics require finite computed values; samples exceed the supported numeric range")
    return result


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report valid counts rather than treating undefined metrics as zero."""
    result: dict[str, Any] = {"utterances": len(rows)}
    for name in ("mse", "snr_db", "si_sdr_db", "si_sdr_improvement_db", "reference_rms", "estimate_rms"):
        values = [row[name] for row in rows if row.get(name) is not None]
        if values or any(name in row for row in rows):
            result[name] = {"mean": sum(values) / len(values) if values else None, "valid_utterances": len(values)}
    result["silent_reference_utterances"] = sum(bool(row.get("silent_reference")) for row in rows)
    return result


def summarize_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep optional score names explicit; no missing-score imputation."""
    names = sorted({name for row in rows for name in row.get("quality", {})})
    result = {}
    for name in names:
        values = [row["quality"][name] for row in rows if name in row.get("quality", {})]
        result[name] = {"mean": sum(values) / len(values), "valid_utterances": len(values)}
    if "cacc" in names:
        weighted = [(row["quality"]["cacc"], row.get("cacc_reference_characters", 0))
                    for row in rows if "cacc" in row.get("quality", {})]
        characters = sum(count for _, count in weighted)
        if characters:
            result["cacc"]["corpus_character_accuracy"] = sum(score * count for score, count in weighted) / characters
            result["cacc"]["reference_characters"] = characters
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize_metrics(rows)
    quality = summarize_quality(rows)
    if quality:
        result["quality"] = quality
    return result


@torch.inference_mode()
def evaluate_dataset(
    model: Any,
    dataset: Any,
    exits: tuple[int, ...] | list[int] | None = None,
    streaming: bool = False,
    chunk_samples: int | None = None,
    check_clean: bool = False,
    quality_evaluator: QualityEvaluator | None = None,
    lookaheads: tuple[int, ...] | list[int] = (0,),
    text_metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    exits = tuple(model.config.exits) if exits is None else tuple(exits)
    if not exits or len(set(exits)) != len(exits) or any(depth not in model.config.exits for depth in exits):
        raise ValueError(f"evaluation exits must be drawn from {model.config.exits}")
    lookaheads = tuple(lookaheads)
    allowed_ahead = getattr(model.config, "lookahead_frames", (0,))
    if (not lookaheads or len(set(lookaheads)) != len(lookaheads)
            or any(isinstance(value, bool) or not isinstance(value, int) or value not in allowed_ahead for value in lookaheads)):
        raise ValueError(f"evaluation lookaheads must be drawn from {allowed_ahead}")
    allowed_rates = getattr(model.config, "supported_sample_rates", (model.config.sample_rate,))
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        length = int(item.get("length", item["clean"].numel()))
        if not 0 < length <= item["clean"].numel() or length > item["noisy"].numel():
            raise ValueError("invalid evaluation utterance length")
        rate = int(item.get("sample_rate", model.config.sample_rate))
        if rate not in allowed_rates:
            raise ValueError(f"utterance sample rate {rate} is not configured: {allowed_rates}")
        noisy, clean = item["noisy"][:length], item["clean"][:length]
        identifier = str(item.get("id", index))
        text = (text_metadata or {}).get(identifier, item)

        def measure(estimate):
            measured = signal_metrics(estimate, clean)
            if quality_evaluator is not None and quality_evaluator.metrics:
                measured["quality"] = quality_evaluator.score(
                    estimate, clean, rate, transcript=text.get("transcript"), language=text.get("language"))
                cacc = quality_evaluator.backends.get("cacc")
                if cacc is not None:
                    measured["cacc_reference_characters"] = cacc.last_reference_characters
            return measured

        def enhance(waveform, depth, ahead):
            # Preserve the simple legacy callable contract at its default rate.
            if rate == model.config.sample_rate and ahead == 0:
                return enhance_waveform(model, waveform, depth, streaming, chunk_samples)[0]
            return enhance_waveform(model, waveform, depth, streaming, chunk_samples,
                                    sample_rate=rate, lookahead=ahead)[0]

        bypass = measure(noisy)
        record: dict[str, Any] = {"id": identifier, "samples": length, "sample_rate": rate,
                                  "bypass": bypass, "by_lookahead": {}}
        for ahead in lookaheads:
            scenario: dict[str, Any] = {"exits": {}}
            if check_clean:
                scenario["clean_input_preservation"] = {}
            for depth in exits:
                measured = measure(enhance(noisy, depth, ahead))
                measured["si_sdr_improvement_db"] = (
                    measured["si_sdr_db"] - bypass["si_sdr_db"]
                    if measured["si_sdr_db"] is not None and bypass["si_sdr_db"] is not None else None
                )
                scenario["exits"][str(depth)] = measured
                if check_clean:
                    scenario["clean_input_preservation"][str(depth)] = measure(enhance(clean, depth, ahead))
            record["by_lookahead"][str(ahead)] = scenario
        # Preserve old consumers: top-level exits refer to primary_lookahead.
        record.update(record["by_lookahead"][str(lookaheads[0])])
        records.append(record)
    if not records:
        raise ValueError("evaluation requires at least one example")

    def summarize_examples(examples):
        summary = {"bypass": _summary([row["bypass"] for row in examples]), "by_lookahead": {}}
        for ahead in lookaheads:
            scenario = {"exits": {str(depth): _summary([
                row["by_lookahead"][str(ahead)]["exits"][str(depth)] for row in examples]) for depth in exits}}
            if check_clean:
                scenario["clean_input_preservation"] = {str(depth): _summary([
                    row["by_lookahead"][str(ahead)]["clean_input_preservation"][str(depth)]
                    for row in examples]) for depth in exits}
            summary["by_lookahead"][str(ahead)] = scenario
        summary.update(summary["by_lookahead"][str(lookaheads[0])])
        return summary

    rates = sorted({row["sample_rate"] for row in records})
    return {
        "sample_rate": rates[0] if len(rates) == 1 else None,
        "sample_rates": rates,
        "inference_mode": "streaming" if streaming else "whole_waveform",
        "exits": list(exits), "lookahead_frames": list(lookaheads), "primary_lookahead": lookaheads[0],
        "metric_implementations": METRIC_IMPLEMENTATIONS,
        "quality_metric_implementations": quality_evaluator.metadata if quality_evaluator is not None else {},
        "limitations": "Only explicitly requested scores are measured. No listening test, paper parity, or production readiness is established. Learned metric backends require separately provisioned local assets; unrequested metrics are absent. Aggregate means weight utterances equally; per-rate results are also reported.",
        "summary": summarize_examples(records),
        "summary_by_sample_rate": {str(rate): summarize_examples([row for row in records if row["sample_rate"] == rate]) for rate in rates},
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
    parser.add_argument("--lookaheads", default="0", help="Comma-separated configured lookahead frame counts")
    parser.add_argument("--metrics", default="", help="Optional comma-separated metrics: pesq,utmos,estoi,dnsmos,nisqa,speechbert_score,lps,cacc")
    parser.add_argument("--metrics-config", help="JSON mapping metric names to explicit local asset settings")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    dataset = (
        SyntheticDataset(model.config.sample_rate, model.config.sample_rate // 5, length=2)
        if args.smoke else PairedAudioDataset(args.manifest, model.config.supported_sample_rates, training=False, load_teacher=False)
    )
    exits = tuple(int(value.strip()) for value in args.exits.split(",")) if args.exits else None
    settings = json.loads(Path(args.metrics_config).read_text()) if args.metrics_config else {}
    if not isinstance(settings, dict):
        parser.error("--metrics-config must contain a JSON object")
    quality = QualityEvaluator([name.strip() for name in args.metrics.split(",") if name.strip()], args.device, settings)
    lookaheads = tuple(int(value.strip()) for value in args.lookaheads.split(","))
    report = evaluate_dataset(model, dataset, exits, args.stream, args.chunk_samples, args.check_clean,
                              quality_evaluator=quality, lookaheads=lookaheads)
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["data_kind"] = "synthetic_smoke_not_quality_evidence" if args.smoke else "user_supplied_paired_manifest"
    if args.manifest:
        report["manifest"] = str(Path(args.manifest).resolve())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved requested metrics for {len(dataset)} examples to {destination}")


if __name__ == "__main__":
    main()
