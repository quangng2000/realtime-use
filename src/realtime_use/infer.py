"""Checkpoint loading and mono-file inference for the independent student."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from .audio import StreamingEnhancer
from .config import ModelConfig
from .data import read_mono_audio
from .model import Enhancer


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> Enhancer:
    """Load tensor/builtin-only checkpoints, with strict architecture matching.

    Scan backends share parameter layouts. Non-CUDA inference selects the
    portable reference scan, including checkpoints trained with an explicitly
    requested CUDA kernel; the checkpoint file itself remains unchanged.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    payload: Any = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model_config", "model_state"} <= payload.keys():
        raise ValueError("checkpoint must contain model_config and model_state")
    model_config = dict(payload["model_config"])
    if device.type != "cuda":
        model_config["scan_backend"] = "reference"
    model = Enhancer(ModelConfig.from_dict(model_config))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def enhance_waveform(
    model: Enhancer,
    waveform: torch.Tensor,
    exit_depth: int | None = None,
    streaming: bool = False,
    chunk_samples: int | None = None,
) -> torch.Tensor:
    """Return [B,N], constructing fresh streaming state for each independent call."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError("waveform must be nonempty [N] or [B,N]")
    exit_depth = max(model.config.exits) if exit_depth is None else exit_depth
    if exit_depth not in model.config.exits:
        raise ValueError(f"exit_depth must be one of {model.config.exits}")
    waveform = waveform.to(next(model.parameters()).device)
    if not streaming:
        return model.enhance(waveform, exit_depths=[exit_depth])[exit_depth]
    hop = round(model.config.sample_rate * model.config.hop_ms / 1000)
    chunk_samples = hop if chunk_samples is None else chunk_samples
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    stream = StreamingEnhancer(model, exit_depth)
    chunks = [stream.push(chunk) for chunk in waveform.split(chunk_samples, dim=-1)]
    chunks.append(stream.flush())
    result = torch.cat(chunks, dim=-1)
    if result.shape != waveform.shape:
        raise RuntimeError(f"stream returned {tuple(result.shape)}, expected {tuple(waveform.shape)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exit-depth", type=int)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--chunk-samples", type=int)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.suffix.lower() != ".wav":
        parser.error("--output must end in .wav; output uses floating-point WAV")
    if destination.resolve() == Path(args.input).resolve():
        parser.error("output must differ from input")
    model = load_model(args.checkpoint, args.device)
    audio = read_mono_audio(args.input, model.config.sample_rate)
    enhanced = enhance_waveform(model, audio, args.exit_depth, args.stream, args.chunk_samples)
    if not bool(torch.isfinite(enhanced).all()):
        raise RuntimeError("model produced non-finite audio; output was not saved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Float WAV preserves output without implicit PCM clipping or quantization.
    sf.write(str(destination), enhanced[0].cpu().numpy(), model.config.sample_rate, subtype="FLOAT", format="WAV")
    print(f"Saved {enhanced.shape[-1]} samples at {model.config.sample_rate} Hz to {destination}")


if __name__ == "__main__":
    main()
