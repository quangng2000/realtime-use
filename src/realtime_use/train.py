"""Hydra entrypoint for Lightning training: python -m realtime_use.train smoke=true."""

from __future__ import annotations

import copy
from datetime import timedelta
import math
import os
from pathlib import Path
import sys
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import to_absolute_path
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
import torch

from .config import ModelConfig
from .datamodule import PairedAudioDataModule
from .lightning_module import EnhancementModule


class TrainingCheckpoints(Callback):
    """Save native Lightning checkpoints only at complete generator updates."""

    def __init__(self, directory: Path, every_updates: int):
        self.directory = directory
        self.every_updates = every_updates
        self.best_score: float | None = None
        self.last_periodic_update = -1
        self.final_validation_rng: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {"best_score": self.best_score, "last_periodic_update": self.last_periodic_update}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_score = state_dict.get("best_score")
        self.last_periodic_update = state_dict.get("last_periodic_update", -1)

    def on_fit_start(self, trainer, pl_module) -> None:
        # A re-executed Hydra worker may initially resolve a different run path.
        # Rank zero owns artifacts; every rank must use the same checkpoint path.
        shared = trainer.strategy.broadcast(str(self.directory), src=0)
        self.directory = Path(shared)
        exists = trainer.strategy.broadcast(
            (self.directory / "last.ckpt").exists() if trainer.is_global_zero else False, src=0)
        if exists and not pl_module.config.get("resume"):
            raise FileExistsError("Output already contains a run; choose another output_dir or resume")
        if trainer.is_global_zero:
            self.directory.parent.mkdir(parents=True, exist_ok=True)
            settings = copy.deepcopy(pl_module.config)
            settings["output_dir"] = str(self.directory.parent)
            OmegaConf.save(OmegaConf.create(settings), self.directory.parent / "resolved_config.yaml")
        trainer.strategy.barrier("run artifacts")

    def _save(self, trainer: L.Trainer, name: str) -> None:
        if trainer.is_global_zero:
            self.directory.mkdir(parents=True, exist_ok=True)
        # All ranks participate: checkpoint hooks gather per-rank RNG state.
        trainer.save_checkpoint(self.directory / name, weights_only=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        update = int(pl_module.completed_updates)
        if pl_module.microbatches_in_update or update == self.last_periodic_update:
            return
        if update and update % self.every_updates == 0:
            self.last_periodic_update = update
            self._save(trainer, f"step_{update:07d}.ckpt")
            self._save(trainer, "last.ckpt")

    def on_validation_end(self, trainer, pl_module) -> None:
        if self.final_validation_rng is not None:
            # A standalone Trainer.validate() sets up DDP again (and reseeds).
            # Validation must not rewrite the training RNG saved in best.ckpt.
            pl_module.restore_rng_state(self.final_validation_rng)
        if trainer.sanity_checking or not int(pl_module.completed_updates):
            return
        if pl_module.microbatches_in_update:
            raise RuntimeError("Validation/checkpointing must run at an optimizer-update boundary")
        value = trainer.callback_metrics.get("validation_mean_clean_loss")
        if value is None:
            return
        score = float(value)
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self._save(trainer, "best.ckpt")

    def on_validation_start(self, trainer, pl_module) -> None:
        if self.final_validation_rng is not None:
            pl_module.restore_rng_state(self.final_validation_rng)

    def on_fit_end(self, trainer, pl_module) -> None:
        self._save(trainer, "last.ckpt")


def prepare_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    settings = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True, throw_on_missing=True)
    settings = copy.deepcopy(settings)
    if settings.get("smoke", False):
        settings["model"].update(depth=1, exits=[1], width=4, state_dim=2, expand=1,
                                  decoder_mode="shared", scan_backend="reference")
        settings["training"].update(steps=min(3, int(settings["training"]["steps"])),
            batch_size=1, crop_seconds=0.08, workers=0, grad_accum=1,
            split_decoders_at=None, checkpoint_every=1, log_every=1,
            validate_every=1, validation_batches=1)
        settings["loss"].update(fft_sizes=[64, 128], kd=0.0)
        settings["adversarial"].update(channels=4, scales=2)
        settings["data"]["smoke"] = True
        settings["trainer"]["precision"] = "32-true"
        torch.set_num_threads(1)
    ModelConfig.from_dict(settings["model"])
    training = settings["training"]
    for key in ("steps", "grad_accum", "checkpoint_every", "validate_every", "validation_batches", "log_every"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["stage"] not in {"regression", "adversarial"}:
        raise ValueError("training.stage must be regression or adversarial")
    if settings.get("resume") and settings.get("init_from"):
        raise ValueError("Choose resume or init_from, not both")
    if training["stage"] == "adversarial" and not any(settings.get(key) for key in ("resume", "init_from", "smoke")):
        raise ValueError("Adversarial training requires init_from a regression checkpoint or resume")
    trainer = settings["trainer"]
    devices = trainer.get("devices", 1)
    valid_devices = ((isinstance(devices, int) and not isinstance(devices, bool) and (devices > 0 or devices == -1))
        or devices == "auto"
        or (isinstance(devices, (list, tuple)) and len(devices) > 0 and len(set(devices)) == len(devices)
            and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in devices)))
    if not valid_devices:
        raise ValueError("trainer.devices must be a positive count, unique device IDs, -1, or auto")
    if trainer.get("strategy", "auto") not in {"auto", "ddp", "ddp_find_unused_parameters_true"}:
        raise ValueError("Use trainer.strategy=auto or ddp; dynamic exits require unused-parameter detection")
    nodes = trainer.get("num_nodes", 1)
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError("trainer.num_nodes must be a positive integer")
    reserved = {"max_steps", "max_epochs", "accumulate_grad_batches", "gradient_clip_val", "callbacks",
                "logger", "enable_checkpointing", "val_check_interval", "check_val_every_n_epoch",
                "limit_val_batches", "default_root_dir", "log_every_n_steps", "use_distributed_sampler"}
    if reserved.intersection(trainer):
        raise ValueError("Use the training group for budgets/accumulation/checkpoint settings, not reserved Trainer keys: "
                         + ", ".join(sorted(reserved.intersection(trainer))))
    for key in ("train_manifest", "validation_manifest"):
        if settings["data"].get(key):
            settings["data"][key] = to_absolute_path(str(settings["data"][key]))
    for key in ("output_dir", "resume", "init_from"):
        if settings.get(key):
            settings[key] = to_absolute_path(str(settings[key]))
    return settings


def uses_ddp(trainer: dict[str, Any]) -> bool:
    devices = trainer.get("devices", 1)
    if devices == "auto" or devices == -1:
        count = torch.cuda.device_count() if trainer.get("accelerator") in {"auto", "gpu", "cuda"} else 1
    else:
        count = len(devices) if isinstance(devices, (list, tuple)) else devices
    return (trainer.get("strategy", "auto") != "auto" or count > 1
            or trainer.get("num_nodes", 1) > 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1)


def trainer_options(settings: dict[str, Any]) -> dict[str, Any]:
    options = dict(settings["trainer"])
    if uses_ddp(options):
        # Sampled exits, phased heads, and alternating GAN optimizers use a
        # changing set of parameters. Never enable static_graph here.
        options["strategy"] = DDPStrategy(find_unused_parameters=True, static_graph=False,
            broadcast_buffers=False, timeout=timedelta(minutes=3))
    return options


def _checkpoint_config(settings: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    requested = ModelConfig.from_dict(settings["model"]).to_dict()
    restored = ModelConfig.from_dict(checkpoint["model_config"]).to_dict()
    for key in requested:
        if key not in {"decoder_mode", "scan_backend"} and requested[key] != restored[key]:
            raise ValueError(f"Checkpoint architecture differs at {key}")
    restored["scan_backend"] = requested["scan_backend"]
    settings["model"] = restored
    if settings.get("resume"):
        if not {"state_dict", "optimizer_states", "loops"} <= checkpoint.keys():
            raise ValueError("resume requires a native Lightning .ckpt; use init_from for legacy .pt weights")
        if checkpoint["stage"] != settings["training"]["stage"]:
            raise ValueError("resume must preserve stage; use init_from for a fresh stage")
        if int(checkpoint["step"]) >= settings["training"]["steps"]:
            raise ValueError("training.steps must exceed the resumed generator-update count")
        previous = checkpoint.get("hyper_parameters", {}).get("config", {})
        # Changing optimizer structure/objective mid-resume is a fresh experiment.
        for group in ("loss", "adversarial"):
            if group in previous and previous[group] != settings[group]:
                raise ValueError(f"resume must preserve {group}; use init_from for a fresh experiment")
        for key in ("lr", "weight_decay", "grad_accum", "predecoder_lr_scale", "sample_exit", "split_decoders_at"):
            old = previous.get("training", {})
            if key in old and old[key] != settings["training"].get(key):
                raise ValueError(f"resume must preserve training.{key}; use init_from to change it")
    elif settings["training"]["stage"] == "adversarial" and checkpoint.get("stage") != "regression":
        raise ValueError("Adversarial init_from requires a regression checkpoint")


def run_training(cfg: DictConfig | dict[str, Any]) -> Path:
    settings = prepare_config(cfg)
    if (uses_ddp(settings["trainer"]) and HydraConfig.initialized()
            and HydraConfig.get().mode == RunMode.MULTIRUN):
        raise ValueError("Launch DDP experiments separately; Hydra -m is supported for single-device jobs")
    L.seed_everything(int(settings["seed"]), workers=True)
    checkpoint = None
    source = settings.get("resume") or settings.get("init_from")
    if source:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        _checkpoint_config(settings, checkpoint)
    destination = Path(settings["output_dir"])
    data = PairedAudioDataModule(settings)
    # Fail invalid/missing manifests before writing run artifacts or initializing a GPU.
    data.setup("fit")
    module = EnhancementModule(settings)
    if checkpoint is not None and settings.get("init_from"):
        module.load_portable_weights(checkpoint)
    training = settings["training"]
    checkpoints = TrainingCheckpoints(destination / "checkpoints", training["checkpoint_every"])
    optimizer_count = 2 if training["stage"] == "adversarial" else 1
    trainer = L.Trainer(**trainer_options(settings),
        default_root_dir=str(destination),
        max_steps=training["steps"] * optimizer_count, max_epochs=-1,
        accumulate_grad_batches=1, enable_checkpointing=False, use_distributed_sampler=False,
        callbacks=[checkpoints], logger=CSVLogger(str(destination), name="metrics"),
        log_every_n_steps=training["log_every"],
        limit_val_batches=training["validation_batches"] if data.has_validation else 0,
        val_check_interval=training["validate_every"] * training["grad_accum"],
        check_val_every_n_epoch=None)
    trainer.fit(module, datamodule=data, ckpt_path=settings.get("resume"))
    # Also evaluate the final update when its budget is not a validation multiple.
    if data.has_validation and int(module.completed_updates) % training["validate_every"]:
        checkpoints.final_validation_rng = module.capture_rng_state()
        try:
            trainer.validate(module, datamodule=data, verbose=False)
        finally:
            module.restore_rng_state(checkpoints.final_validation_rng)
            checkpoints.final_validation_rng = None
        checkpoints._save(trainer, "last.ckpt")
    return checkpoints.directory / "last.ckpt"


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    checkpoint = run_training(cfg)
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0 and int(os.environ.get("NODE_RANK", "0")) == 0:
        print(f"Training complete: {checkpoint}")


def main() -> None:
    # torchrun workers enter Hydra independently; only rank zero writes Hydra
    # metadata/logs. Lightning's launcher already sets output_subdir=null, but
    # these overrides also cover external launchers and the installed command.
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank > 0 or int(os.environ.get("NODE_RANK", "0")) > 0:
        sys.argv.extend(["hydra.output_subdir=null", "hydra/job_logging=disabled", "hydra/hydra_logging=disabled"])
    hydra_main()


if __name__ == "__main__":
    main()
