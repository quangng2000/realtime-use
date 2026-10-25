"""Export one trained exit as a smaller inference-only PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .infer import enhance_waveform, load_model
from .model import Enhancer


def extract_exit(source: Enhancer, exit_depth: int, lookahead: int = 0,
                 sample_rate: int | None = None, retain_all_rates: bool = True) -> Enhancer:
    """Retain the encoder, selected prefix of blocks and matching trained decoder."""
    if exit_depth not in source.config.exits:
        raise ValueError(f"exit_depth must be one of the trained exits {source.config.exits}")
    settings = source.config.to_dict()
    source._check_lookahead(lookahead)
    source.transform_for(sample_rate)
    rate = source.config.sample_rate if sample_rate is None else sample_rate
    settings.update(depth=exit_depth, exits=[exit_depth], decoder_mode="shared", scan_backend="auto",
                    lookahead_frames=[lookahead], sample_rate=rate,
                    sample_rates=list(source.config.supported_sample_rates) if retain_all_rates else [])
    exported = Enhancer(ModelConfig.from_dict(settings))
    source_state = source.state_dict()
    selected_state = {}
    for key in exported.state_dict():
        source_key = key
        if key == "transform.window":
            selected_state[key] = exported.state_dict()[key].detach().cpu().clone()
            continue
        if source.config.codec_impl == "nemotron_causal" and key.startswith("encoder.codec.dense_conv_1."):
            source_key = ("encoder." + source.encoder.branch_state_prefix(lookahead) + "."
                          + key.removeprefix("encoder.codec.dense_conv_1."))
        if key.startswith("shared_decoder.") and source.config.decoder_mode == "per_exit":
            source_key = f"decoders.{exit_depth}." + key.removeprefix("shared_decoder.")
        selected_state[key] = source_state[source_key].detach().cpu().clone()
    exported.load_state_dict(selected_state, strict=True)
    return exported.to(next(source.parameters()).device).eval()


@torch.inference_mode()
def verify_export(source: Enhancer, exported: Enhancer, exit_depth: int, lookahead: int = 0) -> dict:
    """Compare full and chunked outputs on the selected architecture and device."""
    reference = source.eval()
    generator = torch.Generator().manual_seed(2026)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    try:
        errors = {"full_max_absolute_error": 0.0, "stream_max_absolute_error": 0.0,
                  "stream_vs_full_max_absolute_error": 0.0}
        for rate in exported.config.supported_sample_rates:
            hop = round(rate * source.config.hop_ms / 1000)
            waveform = 0.05 * torch.randn(1, (3 + lookahead) * hop + 7, generator=generator)
            options = {"sample_rate": rate, "lookahead": lookahead}
            original_full = enhance_waveform(reference, waveform, exit_depth, **options)
            exported_full = enhance_waveform(exported, waveform, exit_depth, **options)
            original_stream = enhance_waveform(reference, waveform, exit_depth, streaming=True,
                                               chunk_samples=max(1, hop // 2 + 1), **options)
            exported_stream = enhance_waveform(exported, waveform, exit_depth, streaming=True,
                                               chunk_samples=max(1, hop // 2 + 1), **options)
            for candidate in (exported_full, original_stream, exported_stream):
                if not torch.isfinite(candidate).all():
                    raise RuntimeError("Export verification produced nonfinite output")
                torch.testing.assert_close(candidate, original_full, atol=3e-4, rtol=3e-4)
            for key, difference in (
                ("full_max_absolute_error", exported_full - original_full),
                ("stream_max_absolute_error", exported_stream - original_stream),
                ("stream_vs_full_max_absolute_error", exported_stream - original_full),
            ):
                errors[key] = max(errors[key], float(difference.abs().max()))
        return {**errors, "sample_rates_checked": list(exported.config.supported_sample_rates),
                "lookahead_frames": lookahead}
    finally:
        torch.set_num_threads(previous_threads)


def export_exit(checkpoint: str | Path, exit_depth: int, output: str | Path,
                device: str | torch.device = "cpu", lookahead: int = 0,
                sample_rate: int | None = None, retain_all_rates: bool = True) -> Path:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.resolve() == Path(checkpoint).resolve():
        raise ValueError("Export destination must differ from the source checkpoint")
    source = load_model(checkpoint, device=device)
    exported = extract_exit(source, exit_depth, lookahead, sample_rate, retain_all_rates)
    verification = verify_export(source, exported, exit_depth, lookahead)
    payload = {
        "format_version": 1, "model_config": exported.config.to_dict(),
        "model_state": {key: value.detach().cpu() for key, value in exported.state_dict().items()}, "stage": "export",
        "exit_depth": exit_depth, "lookahead_frames": lookahead,
        "algorithmic_latency_ms": exported.config.window_ms + lookahead * exported.config.hop_ms,
        "active_parameters": exported.active_parameter_count(exit_depth, lookahead),
        "verification": {"backend": source.config.mixer_impl,
                         "device": str(next(source.parameters()).device), **verification},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrently created destination being replaced.
    try:
        with destination.open("xb") as handle:
            torch.save(payload, handle)
    except FileExistsError:
        raise
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exit-depth", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", help="Official Mamba checkpoints require cuda")
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, help="Default native rate for the exported checkpoint")
    parser.add_argument("--single-rate", action="store_true", help="Retain only the selected/default sample rate")
    args = parser.parse_args()
    result = export_exit(args.checkpoint, args.exit_depth, args.output, device=args.device,
                         lookahead=args.lookahead, sample_rate=args.sample_rate, retain_all_rates=not args.single_rate)
    print(json.dumps({"event": "exported", "checkpoint": str(result),
                      "exit_depth": args.exit_depth, "verification": "full/stream equivalence",
                      "device": args.device, "lookahead_frames": args.lookahead}))


if __name__ == "__main__":
    main()
