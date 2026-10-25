"""Readiness checks without Trainer.fit, optimizer steps, or checkpoint writes."""

from __future__ import annotations

import importlib.metadata
import json

import hydra
from omegaconf import DictConfig
import torch

from .audio import StreamingEnhancer
from .datamodule import PairedAudioDataModule
from .lightning_module import EnhancementModule
from .train import _checkpoint_config, prepare_config


def run_preflight(cfg: DictConfig | dict) -> dict:
    settings = prepare_config(cfg)
    checkpoint = None
    if source := settings.get("resume") or settings.get("init_from"):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        _checkpoint_config(settings, checkpoint)
    data = PairedAudioDataModule(settings)
    data.setup("fit")
    accelerator = settings["trainer"].get("accelerator", "cpu")
    use_cuda = accelerator in {"gpu", "cuda"} or (accelerator == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this configuration; no training or GPU allocation was started")
    devices = settings["trainer"].get("devices", 1)
    index = devices[0] if isinstance(devices, (list, tuple)) else 0
    device = torch.device("cuda", index) if use_cuda else torch.device("cpu")
    module = EnhancementModule(settings).to(device).eval()
    if checkpoint is not None:
        module.load_portable_weights(checkpoint)
    module._ensure_quality_evaluator()
    report = {
        "training_started": False, "optimizer_updates": 0,
        "scope": "one-device FP32 forward/streaming, native-rate geometry, local metric asset loading; not AMP/NCCL or quality validation",
        "device": str(device), "cuda_version": torch.version.cuda,
        "torch_version": str(torch.__version__),
        "dataset_rows": {"train": len(data.train_dataset),
                         "validation": len(data.validation_dataset) if data.validation_dataset is not None else 0},
        "training_sample_rates": sorted(set(getattr(data.train_dataset, "item_sample_rates", [data.sample_rate]))),
        "configured_sample_rates": list(module.model.config.supported_sample_rates),
        "lookaheads": list(module.model.config.lookahead_frames),
        "parameter_count_lookahead": min(module.model.config.lookahead_frames),
        "registered_enhancer_parameters_including_transition_heads": sum(p.numel() for p in module.model.parameters()),
        "active_parameters_by_exit": {str(depth): module.model.active_parameter_count(
                                          depth, lookahead=min(module.model.config.lookahead_frames))
                                      for depth in module.model.config.exits},
        "checks": [],
    }
    if use_cuda:
        report["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    # A small deterministic signal verifies every configured native geometry.
    # It is neither a training batch nor a speech-quality evaluation.
    with torch.inference_mode():
        for rate in module.model.config.supported_sample_rates:
            transform = module.model.transform_for(rate)
            n = transform.window_samples * 3 + 7
            waveform = (torch.sin(torch.arange(n, device=device) * (2 * torch.pi * 170 / rate)) * .03)[None]
            for lookahead in module.model.config.lookahead_frames:
                outputs = module(waveform, sample_rate=rate, lookahead=lookahead)
                for depth, expected in outputs.items():
                    if expected.shape != waveform.shape or not bool(torch.isfinite(expected).all()):
                        raise RuntimeError("Model returned invalid waveform shape or nonfinite output")
                    stream = StreamingEnhancer(module.model, depth, sample_rate=rate, lookahead=lookahead)
                    parts = [stream.push(chunk) for chunk in waveform.split(max(1, transform.hop_samples // 2), dim=-1)]
                    parts.append(stream.flush())
                    actual = torch.cat(parts, dim=-1)
                    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
                    report["checks"].append({"sample_rate": rate, "exit": depth, "lookahead": lookahead,
                                            "streaming_max_abs_error": float((actual - expected).abs().max())})
    if module._quality_evaluator is not None:
        report["metric_backends"] = module._quality_evaluator.metadata
    if use_cuda:
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["dependencies"] = {}
    for package in ("lightning", "mamba-ssm", "causal-conv1d", "descript-audio-codec", "auraloss"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = None
    return report


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_preflight(cfg), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
