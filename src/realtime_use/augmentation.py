"""Optional training-input corruptions delegated to audiomentations.

These transforms preserve sample count and never modify the clean target. They
are data preparation, not DSP used by the streaming enhancer. Randomness belongs
to audiomentations (Python/NumPy), seeded by Lightning's worker initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import torch


def augmentation_enabled(config: Mapping[str, Any] | None) -> bool:
    if config is None:
        return False
    if not isinstance(config, Mapping) or not isinstance(config.get("enabled", False), bool):
        raise ValueError("data.augmentation must be a mapping with a boolean enabled flag")
    return config.get("enabled", False)


class InputAugmentation:
    """CPU float32 tensor adapter around the library's waveform Compose."""

    def __init__(self, pipeline: Any, sample_rate: int) -> None:
        self.pipeline = pipeline
        self.sample_rate = sample_rate

    def __call__(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 1 or noisy.device.type != "cpu" or noisy.dtype != torch.float32:
            raise ValueError("Input augmentation requires a mono CPU float32 tensor")
        # Library transforms receive their own array; even an in-place transform
        # cannot mutate a decoded waveform or a clean target sharing its storage.
        samples = noisy.detach().numpy().copy()
        try:
            result = self.pipeline(samples=samples, sample_rate=self.sample_rate)
        except ValueError as exc:
            raise ValueError(
                f"audiomentations failed for a {noisy.numel()}-sample clip: {exc}. "
                "Check transform settings and clip length; zero-phase low-pass filtering "
                "requires enough samples for its filter padding."
            ) from exc
        if result.shape != samples.shape or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("audiomentations must return finite float32 audio with unchanged sample count")
        return torch.from_numpy(np.ascontiguousarray(result))


def build_input_augmentation(
    config: Mapping[str, Any] | None,
    sample_rate: int,
    *,
    training: bool,
    load_teacher: bool,
) -> InputAugmentation | None:
    """Build only for training, rejecting stale cached-teacher supervision."""
    if not augmentation_enabled(config) or not training:
        return None
    if load_teacher:
        raise ValueError(
            "Enabled input augmentation is incompatible with cached teacher targets; "
            "disable augmentation or set loss.kd=0 (load_teacher=False)."
        )
    unknown = set(config) - {"enabled", "gaussian_snr", "clipping", "low_pass"}
    if unknown:
        raise ValueError(f"Unsupported data.augmentation keys: {sorted(unknown)}")
    try:
        from audiomentations import AddGaussianSNR, ClippingDistortion, Compose, LowPassFilter
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Enabled augmentation requires audiomentations; install realtime-use[augmentation]."
        ) from exc
    defaults = {
        "gaussian_snr": {"p": 0.3, "min_snr_db": 5.0, "max_snr_db": 30.0},
        "clipping": {"p": 0.2, "min_percentile_threshold": 0, "max_percentile_threshold": 10},
        "low_pass": {"p": 0.2, "min_cutoff_freq": 1000.0, "max_cutoff_freq": 6000.0,
                     "min_rolloff": 12, "max_rolloff": 24, "zero_phase": True},
    }
    transforms = []
    for name, transform in (("gaussian_snr", AddGaussianSNR), ("clipping", ClippingDistortion),
                            ("low_pass", LowPassFilter)):
        supplied = config.get(name, {})
        if not isinstance(supplied, Mapping):
            raise ValueError(f"data.augmentation.{name} must be a mapping")
        parameters = {**defaults[name], **supplied}
        probability = parameters["p"]
        if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not 0 <= probability <= 1:
            raise ValueError(f"data.augmentation.{name}.p must lie in [0, 1]")
        if probability == 0:
            continue
        if name == "low_pass":
            low, high = parameters["min_cutoff_freq"], parameters["max_cutoff_freq"]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in (low, high)) or not 0 < low <= high < sample_rate / 2:
                raise ValueError("Low-pass cutoff frequencies must be positive, ordered, and below Nyquist")
            if parameters["zero_phase"] is not True:
                raise ValueError("Low-pass augmentation requires zero_phase=True to preserve phase alignment")
        transforms.append(transform(**parameters))
    return InputAugmentation(Compose(transforms, shuffle=False), sample_rate)
