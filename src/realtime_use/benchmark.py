"""Measure actual per-hop streaming work, including transforms and GPU transfers."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import soundfile as sf

from .audio import StreamingEnhancer
from .data import read_mono_audio
from .infer import load_model


def summarize_timings(milliseconds: list[float], hop_ms: float) -> dict[str, Any]:
    if not milliseconds or hop_ms <= 0 or any(not math.isfinite(value) or value < 0 for value in milliseconds):
        raise ValueError("timings must be nonempty, finite, nonnegative, with a positive hop")
    values = np.asarray(milliseconds, dtype=np.float64)
    misses = int((values > hop_ms).sum())
    return {
        "calls": len(milliseconds),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "hop_budget_ms": hop_ms,
        "deadline_misses": misses,
        "deadline_miss_fraction": misses / len(milliseconds),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_stream(
    model: Any,
    waveform: torch.Tensor,
    exit_depth: int | None = None,
    warmup_seconds: float = 1.0,
    sample_rate: int | None = None,
    lookahead: int = 0,
) -> dict[str, Any]:
    """Single stream, batch one; host input/output transfers are inside timing.

    The loop is deliberately unpaced: it measures service time, not microphone
    capture, playback, operating-system scheduling, or network latency.
    """
    if waveform.ndim != 1 or not bool(torch.isfinite(waveform).all()):
        raise ValueError("benchmark input must be a finite mono waveform")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be nonnegative")
    device = next(model.parameters()).device
    config = model.config
    rate = config.sample_rate if sample_rate is None else sample_rate
    exit_depth = max(config.exits) if exit_depth is None else exit_depth
    if exit_depth not in config.exits:
        raise ValueError(f"exit_depth must be one of {config.exits}")
    stream = StreamingEnhancer(model, exit_depth, sample_rate=rate, lookahead=lookahead)
    hop_samples = stream.transform.hop_samples
    startup_hops = math.ceil(config.window_ms / config.hop_ms) + lookahead
    total_hops = waveform.numel() // hop_samples
    if total_hops <= startup_hops:
        raise ValueError(f"benchmark needs more than {startup_hops} complete hops")
    host_input = waveform[: total_hops * hop_samples].detach().to("cpu").unsqueeze(0)
    warmup_hops = math.ceil(warmup_seconds * rate / hop_samples)
    for index in range(warmup_hops):
        offset = (index % total_hops) * hop_samples
        stream.push(host_input[:, offset : offset + hop_samples].to(device)).cpu()
    _synchronize(device)
    stream.reset()  # Warmup audio must never leak into the measured stream.
    timings = []
    first_output_input_samples = None
    output_samples = 0
    for index, chunk in enumerate(host_input.split(hop_samples, dim=-1)):
        _synchronize(device)
        start = time.perf_counter_ns()
        output = stream.push(chunk.to(device)).cpu()
        _synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        timings.append(elapsed_ms)
        if output.numel() and first_output_input_samples is None:
            first_output_input_samples = (index + 1) * hop_samples
        output_samples += output.shape[-1]
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("non-finite streaming output")
    _synchronize(device)
    start = time.perf_counter_ns()
    tail = stream.flush().cpu()
    _synchronize(device)
    flush_ms = (time.perf_counter_ns() - start) / 1_000_000
    output_samples += tail.shape[-1]
    if output_samples != host_input.shape[-1]:
        raise RuntimeError(f"stream output length {output_samples}, expected {host_input.shape[-1]}")
    if not bool(torch.isfinite(tail).all()):
        raise RuntimeError("non-finite flush output")
    return {
        "mode": "single_stream_batch_one_per_hop_service_time",
        "timing_scope": "CPU input to device, streaming STFT/model/overlap-add, output to CPU, and CUDA synchronization when applicable.",
        "not_measured": "Audio-device capture/playback, network delay, real-time scheduling, concurrency, and sustained production load. This is not an offline whole-file RTF benchmark or an end-to-end latency guarantee.",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
        "torch_version": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "sample_rate": rate,
        "exit_depth": exit_depth,
        "active_parameters": model.active_parameter_count(exit_depth, lookahead),
        "algorithmic_window_ms": config.window_ms,
        "lookahead_frames": lookahead,
        "algorithmic_latency_ms": config.window_ms + lookahead * config.hop_ms,
        "input_seconds_measured": host_input.shape[-1] / rate,
        "input_tail_samples_excluded": waveform.numel() - host_input.shape[-1],
        "warmup_hops": warmup_hops,
        "startup_hops_excluded_from_steady_state": startup_hops,
        "input_ms_before_first_output": first_output_input_samples * 1000 / rate if first_output_input_samples else None,
        "all_hop_calls": summarize_timings(timings, config.hop_ms),
        "steady_state_hop_calls": summarize_timings(timings[startup_hops:], config.hop_ms),
        "flush_ms": flush_ms,
        "per_hop_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exit-depth", type=int)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--input", help="Optional mono timing input at the model sample rate")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--lookahead", type=int)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.threads is not None:
        if args.threads <= 0:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, args.device)
    rate = (args.sample_rate if args.sample_rate is not None else
            int(sf.info(args.input).samplerate) if args.input else model.config.sample_rate)
    lookahead = min(model.config.lookahead_frames) if args.lookahead is None else args.lookahead
    model.transform_for(rate)
    samples = round(args.seconds * rate)
    waveform = (
        read_mono_audio(args.input, rate)[:samples]
        if args.input else torch.randn(samples, generator=torch.Generator().manual_seed(31)) * 0.05
    )
    report = benchmark_stream(model, waveform, args.exit_depth, args.warmup_seconds, rate, lookahead)
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["input_kind"] = "user_audio" if args.input else "synthetic_noise_for_timing_only"
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    stats = report["steady_state_hop_calls"]
    print(f"Saved streaming timings to {destination}; p95={stats['p95_ms']:.3f} ms, misses={stats['deadline_misses']}/{stats['calls']}")


if __name__ == "__main__":
    main()
