"""Lightning data lifecycle for strict paired audio and explicit smoke data.

The entrypoint must call ``lightning.seed_everything(seed, workers=True)`` once
before constructing the model and data module. Lightning then initializes worker
seeds and advances the training sampler epoch. Use ``use_distributed_sampler=False``
on the Trainer: this module owns both training and validation sharding. Native
checkpoint resume restores training state,
but this module does not claim exact mid-epoch data order or crop replay.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import ModelConfig
from .data import PairedAudioDataset, SyntheticDataset, collate_audio
from .distributed_data import PaddedValidationSampler, ValidationDataset


class PairedAudioDataModule(LightningDataModule):
    """Build train/validation loaders from a resolved, plain configuration.

    The entrypoint resolves manifest paths before construction. Paths inside a
    manifest remain relative to that manifest, as enforced by PairedAudioDataset.
    Synthetic validation is an execution check, never held-out quality evidence.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = deepcopy(config)
        model = ModelConfig.from_dict(self.config.get("model", {}))
        training = self.config.get("training", {})
        data = self.config.get("data", {})
        loss = self.config.get("loss", {})
        self.sample_rate = model.sample_rate
        self.seed = int(self.config.get("seed", 0))
        self.batch_size = training.get("batch_size", 4)
        self.workers = training.get("workers", 0)
        for name, value, minimum in (("batch_size", self.batch_size, 1), ("workers", self.workers, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"training.{name} must be an integer >= {minimum}")
        crop_seconds = float(training.get("crop_seconds", 2.0))
        if not math.isfinite(crop_seconds) or crop_seconds <= 0:
            raise ValueError("training.crop_seconds must be finite and positive")
        self.crop_samples = round(crop_seconds * self.sample_rate)
        if self.crop_samples < 1:
            raise ValueError("training.crop_seconds must produce at least one sample")
        stage = training.get("stage", "regression")
        if stage not in {"regression", "adversarial"}:
            raise ValueError("training.stage must be regression or adversarial")
        kd_weight = float(loss.get("kd", 0.0))
        if not math.isfinite(kd_weight) or kd_weight < 0:
            raise ValueError("loss.kd must be finite and nonnegative")
        self.load_teacher = stage == "regression" and kd_weight > 0
        self.smoke = data.get("smoke", False)
        self.pin_memory = data.get("pin_memory", False)
        if not isinstance(self.smoke, bool) or not isinstance(self.pin_memory, bool):
            raise ValueError("data.smoke and data.pin_memory must be booleans")
        if self.smoke and self.load_teacher:
            raise ValueError("Synthetic smoke data has no teacher targets; set loss.kd=0")
        self.train_manifest = data.get("train_manifest")
        self.validation_manifest = data.get("validation_manifest")
        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None

    @property
    def has_validation(self) -> bool:
        return self.smoke or bool(self.validation_manifest)

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit", "validate"}:
            raise ValueError("PairedAudioDataModule supports fit and validate; use evaluate.py for signal metrics")
        if stage in {None, "fit"} and self.train_dataset is None:
            if self.smoke:
                self.train_dataset = SyntheticDataset(self.sample_rate, self.crop_samples, length=8)
            else:
                if not self.train_manifest:
                    raise ValueError("data.train_manifest is required unless data.smoke=true")
                dataset = PairedAudioDataset(
                    self.train_manifest,
                    self.sample_rate,
                    crop_samples=self.crop_samples,
                    training=True,
                    load_teacher=self.load_teacher,
                )
                if self.load_teacher and any("teacher" not in row for row in dataset.records):
                    raise ValueError("Regression loss.kd > 0 requires a teacher target on every training row")
                self.train_dataset = dataset
            if len(self.train_dataset) == 0:
                raise ValueError("Training dataset must contain at least one example")
        if self.has_validation and self.validation_dataset is None:
            self.validation_dataset = (
                SyntheticDataset(self.sample_rate, self.crop_samples, length=2)
                if self.smoke else PairedAudioDataset(
                    self.validation_manifest,
                    self.sample_rate,
                    crop_samples=self.crop_samples,
                    training=False,
                    load_teacher=False,
                )
            )

    def _distributed_context(self) -> tuple[int, int]:
        # setup() can run before attaching the Trainer. Resolve ranks only when
        # it requests loaders, after the distributed strategy is initialized.
        trainer = self.trainer
        return (int(trainer.world_size), int(trainer.global_rank)) if trainer is not None else (1, 0)

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        world_size, rank = self._distributed_context()
        sampler = None
        if not shuffle:
            dataset = ValidationDataset(dataset)
        if world_size > 1:
            sampler = (
                DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                   shuffle=True, seed=self.seed, drop_last=False)
                if shuffle else PaddedValidationSampler(dataset, num_replicas=world_size, rank=rank)
            )
        # Leave worker_init_fn to Lightning's seed_everything(workers=True)
        # integration. Lightning calls set_epoch on our training sampler even
        # with automatic distributed sampler replacement disabled.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.workers,
            collate_fn=collate_audio,
            pin_memory=self.pin_memory,
            persistent_workers=self.workers > 0,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("Training dataset must contain at least one example")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | list:
        if not self.has_validation:
            return []
        if self.validation_dataset is None:
            self.setup("validate")
        if self.validation_dataset is None:
            raise RuntimeError("Validation dataset was not initialized")
        return self._loader(self.validation_dataset, shuffle=False)
