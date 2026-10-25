"""Strict sample-aligned paired audio and explicitly synthetic smoke-test data.

Matching sample counts cannot prove acoustic alignment: the manifest producer is
responsible for aligning noisy, clean, and optional teacher waveforms beforehand.
This module never resamples, downmixes, shifts, or independently crops a pair.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .augmentation import build_input_augmentation


def read_mono_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Read finite mono audio at the exact requested rate, without conversion."""
    samples, actual_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(f"{path}: sample rate {actual_rate}, expected {sample_rate}")
    if samples.shape[1] != 1:
        raise ValueError(f"{path}: expected mono audio, got {samples.shape[1]} channels")
    if samples.shape[0] == 0:
        raise ValueError(f"{path}: empty audio is not supported")
    if not np.isfinite(samples).all():
        raise ValueError(f"{path}: audio contains non-finite samples")
    return torch.from_numpy(np.ascontiguousarray(samples[:, 0]))


class PairedAudioDataset(Dataset):
    """JSONL rows require noisy, clean, and sample_rate; teacher is optional.

    Relative paths resolve against the manifest directory. Each row is checked
    at construction, and decoded lengths are checked again when loading. Short
    training examples receive identical right padding across every target.
    Set load_teacher=False when distillation is disabled to avoid accessing any
    teacher files, including teacher paths retained in a reused manifest.
    Optional augmentation corrupts only the valid noisy crop during training,
    before padding. It cannot be combined with cached teacher loading.
    """

    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int | Sequence[int],
        crop_samples: int | None = None,
        training: bool = True,
        load_teacher: bool = True,
        augmentation: dict[str, Any] | None = None,
        crop_seconds: float | None = None,
    ) -> None:
        rates = (sample_rate,) if isinstance(sample_rate, int) else tuple(sample_rate)
        if not rates or len(set(rates)) != len(rates) or any(
            isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0 for rate in rates
        ):
            raise ValueError("sample_rate must contain unique positive integers")
        if crop_seconds is not None and (not math.isfinite(crop_seconds) or crop_seconds <= 0
                                         or any(round(crop_seconds * rate) < 1 for rate in rates)):
            raise ValueError("crop_seconds must produce at least one sample at every rate")
        if crop_seconds is not None and crop_samples is not None:
            raise ValueError("Choose crop_seconds or crop_samples, not both")
        if len(rates) > 1 and crop_samples is not None:
            raise ValueError("Use crop_seconds for native multi-rate crops")
        if crop_samples is not None and (
            isinstance(crop_samples, bool) or not isinstance(crop_samples, int) or crop_samples <= 0
        ):
            raise ValueError("crop_samples must be a positive integer or None")
        self.manifest = Path(manifest).expanduser().resolve(strict=True)
        self.sample_rates = rates
        self.sample_rate = rates[0]
        self.crop_samples = crop_samples
        self.crop_seconds = crop_seconds
        self.training = training
        self.load_teacher = load_teacher
        self.augmentations = {}
        for rate in rates:
            rate_config = deepcopy(augmentation)
            if len(rates) > 1 and rate_config and rate_config.get("enabled"):
                # Low-pass cutoffs are expressed in Hz. Keep the configured
                # upper bound where possible and respect each native Nyquist.
                low_pass = rate_config.setdefault("low_pass", {})
                low_pass["max_cutoff_freq"] = min(low_pass.get("max_cutoff_freq", 6000.0), rate / 2 * 0.999)
            self.augmentations[rate] = build_input_augmentation(
                rate_config, rate, training=training, load_teacher=load_teacher,
            )
        self.augmentation = self.augmentations[rates[0]]
        self.records: list[dict[str, Any]] = []
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prefix = f"{self.manifest}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{prefix}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{prefix}: each manifest row must be an object")
                rate = row.get("sample_rate")
                if isinstance(rate, bool) or not isinstance(rate, int) or rate not in rates:
                    raise ValueError(f"{prefix}: sample_rate must be one of {rates}")
                record: dict[str, Any] = {"id": str(row.get("id", f"row-{line_number}")), "sample_rate": rate}
                for key in ("transcript", "language"):
                    if key in row:
                        if not isinstance(row[key], str):
                            raise ValueError(f"{prefix}: {key} must be a string")
                        record[key] = row[key]
                counts = []
                for key in ("noisy", "clean", "teacher"):
                    if key == "teacher" and (not self.load_teacher or key not in row):
                        continue
                    if not isinstance(row.get(key), str) or not row[key]:
                        raise ValueError(f"{prefix}: {key} must be a nonempty file path")
                    path = Path(row[key]).expanduser()
                    if not path.is_absolute():
                        path = self.manifest.parent / path
                    path = path.resolve(strict=True)
                    info = sf.info(str(path))
                    if info.samplerate != rate:
                        raise ValueError(f"{prefix}: {key} sample rate {info.samplerate}, expected {rate}")
                    if info.channels != 1:
                        raise ValueError(f"{prefix}: {key} must be mono; got {info.channels} channels")
                    if info.frames <= 0:
                        raise ValueError(f"{prefix}: {key} is empty")
                    record[key] = path
                    counts.append(info.frames)
                if len(set(counts)) != 1:
                    raise ValueError(f"{prefix}: unequal lengths {counts}; pairs must be sample-aligned")
                record["frames"] = counts[0]
                self.records.append(record)
        if not self.records:
            raise ValueError(f"{self.manifest}: no audio examples")
        self.item_sample_rates = tuple(record["sample_rate"] for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rate = record["sample_rate"]
        crop_samples = round(self.crop_seconds * rate) if self.crop_seconds is not None else self.crop_samples
        audio = {
            key: read_mono_audio(record[key], rate)
            for key in ("noisy", "clean", "teacher")
            if key in record
        }
        lengths = {value.numel() for value in audio.values()}
        if len(lengths) != 1 or next(iter(lengths)) != record["frames"]:
            raise ValueError(f"{record['id']}: audio changed or paired lengths differ")
        count = next(iter(lengths))
        valid_length = count
        if crop_samples is not None:
            available = max(0, count - crop_samples)
            start = int(torch.randint(available + 1, ()).item()) if self.training and available else 0
            valid_length = min(count, crop_samples)
            audio = {
                key: value[start : start + crop_samples]
                for key, value in audio.items()
            }
        if self.augmentations[rate] is not None:
            audio["noisy"] = self.augmentations[rate](audio["noisy"])
        if crop_samples is not None and valid_length < crop_samples:
            audio = {key: F.pad(value, (0, crop_samples - valid_length)) for key, value in audio.items()}
        return {**audio, "length": valid_length, "id": record["id"], "sample_rate": rate,
                **{key: record[key] for key in ("transcript", "language") if key in record}}


def collate_audio(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a batch identically and retain valid lengths for loss/metric masking."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    rates = {item["sample_rate"] for item in examples if "sample_rate" in item}
    if len(rates) > 1:
        raise ValueError("A batch must have one native sample_rate; use rate-bucket batching")
    if any(isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0 for rate in rates):
        raise ValueError("sample_rate must be a positive integer")
    teacher_presence = ["teacher" in item for item in examples]
    if any(teacher_presence) and not all(teacher_presence):
        raise ValueError("a batch must consistently include teacher targets or omit them")
    keys = ("noisy", "clean", "teacher") if all(teacher_presence) else ("noisy", "clean")
    lengths = []
    for item in examples:
        shapes = [item[key].shape for key in keys]
        if any(len(shape) != 1 for shape in shapes) or len(set(shapes)) != 1:
            raise ValueError("each example must contain equal-length 1D paired tensors")
        length = int(item.get("length", item["noisy"].numel()))
        if not 0 < length <= item["noisy"].numel():
            raise ValueError("invalid valid length")
        lengths.append(length)
    padded_length = max(item["noisy"].numel() for item in examples)
    result = {
        key: torch.stack([F.pad(item[key], (0, padded_length - item[key].numel())) for item in examples])
        for key in keys
    }
    if rates:
        result["sample_rate"] = next(iter(rates))
    result["lengths"] = torch.tensor(lengths, dtype=torch.long)
    result["ids"] = [str(item.get("id", index)) for index, item in enumerate(examples)]
    for key in ("transcript", "language"):
        if any(key in item for item in examples):
            result[key] = [item.get(key) for item in examples]
    if any("distributed_valid" in item for item in examples):
        if not all(isinstance(item.get("distributed_valid"), bool) for item in examples):
            raise ValueError("a distributed validation batch requires a boolean validity flag on every row")
        result["distributed_valid"] = torch.tensor(
            [item["distributed_valid"] for item in examples], dtype=torch.bool,
        )
    return result


class SyntheticDataset(Dataset):
    """Deterministic tones/noise for execution checks, never quality evaluation."""

    def __init__(self, sample_rate: int, crop_samples: int, length: int = 8) -> None:
        if min(sample_rate, crop_samples, length) <= 0:
            raise ValueError("sample_rate, crop_samples, and length must be positive")
        self.sample_rate, self.crop_samples, self.length = sample_rate, crop_samples, length
        self.item_sample_rates = (sample_rate,) * length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(1907 + index)
        time = torch.arange(self.crop_samples, dtype=torch.float32) / self.sample_rate
        frequency = 140.0 + 19.0 * index
        clean = 0.12 * torch.sin(2 * torch.pi * frequency * time)
        clean += 0.04 * torch.sin(2 * torch.pi * 2 * frequency * time + 0.3)
        noisy = clean + 0.035 * torch.randn(self.crop_samples, generator=generator)
        return {"noisy": noisy, "clean": clean, "length": self.crop_samples, "id": f"synthetic-{index}", "sample_rate": self.sample_rate}
