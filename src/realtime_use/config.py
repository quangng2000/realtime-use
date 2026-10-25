"""Validated configuration for independently trained causal baselines."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    window_ms: float = 40.0
    hop_ms: float = 20.0
    depth: int = 12
    exits: tuple[int, ...] = (4, 8, 12)
    width: int = 64
    state_dim: int = 16
    expand: int = 4
    conv_kernel: int = 4
    decoder_mode: str = "shared"
    codec_impl: str = "legacy"
    mixer_impl: str = "legacy"
    scan_backend: str = "reference"

    def __post_init__(self) -> None:
        self.exits = tuple(self.exits)
        for name in ("sample_rate", "depth", "width", "state_dim", "expand", "conv_kernel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window_ms and hop_ms must be positive")
        if abs(self.window_ms - 2 * self.hop_ms) > 1e-6:
            raise ValueError("This baseline supports window_ms = 2 * hop_ms")
        for ms in (self.window_ms, self.hop_ms):
            samples = self.sample_rate * ms / 1000
            if abs(samples - round(samples)) > 1e-6:
                raise ValueError("Window and hop must correspond to integer samples")
        if not self.exits or self.exits != tuple(sorted(set(self.exits))):
            raise ValueError("exits must be nonempty, unique, and increasing")
        if any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= self.depth for x in self.exits):
            raise ValueError("Each exit must be an integer between 1 and depth")
        if self.decoder_mode not in {"shared", "per_exit"}:
            raise ValueError("decoder_mode must be shared or per_exit")
        if self.codec_impl not in {"legacy", "nemotron_causal"}:
            raise ValueError("codec_impl must be legacy or nemotron_causal")
        if self.mixer_impl not in {"legacy", "mamba_ssm"}:
            raise ValueError("mixer_impl must be legacy or mamba_ssm")
        if self.scan_backend not in {"reference", "auto", "mamba_ssm"}:
            raise ValueError("scan_backend must be reference, auto, or mamba_ssm")
        if self.mixer_impl == "mamba_ssm" and self.scan_backend != "auto":
            raise ValueError("The official mamba_ssm mixer requires scan_backend=auto; "
                             "scan_backend selects kernels only for mixer_impl=legacy")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["exits"] = list(self.exits)
        return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict) or not isinstance(result.get("model"), dict):
        raise ValueError("Configuration must contain a model mapping")
    ModelConfig.from_dict(result["model"])
    return result
