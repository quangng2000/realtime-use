"""Standalone torchrun worker for CPU/Gloo training correctness tests.

This is deliberately not a pytest process: each distributed worker performs a
bounded experiment and writes its own artifact for the parent test to inspect.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy
import torch
from torch.utils.data import DataLoader, Dataset

from realtime_use.datamodule import PairedAudioDataModule
from realtime_use.lightning_module import EnhancementModule


def tiny_config(stage: str, steps: int, depth: int = 1) -> dict:
    return {
        "seed": 7,
        "model": {
            "sample_rate": 800, "window_ms": 40.0, "hop_ms": 20.0,
            "depth": depth, "exits": list(range(1, depth + 1)), "width": 4,
            "state_dim": 2, "expand": 1, "conv_kernel": 2,
            "decoder_mode": "shared", "scan_backend": "reference",
        },
        "training": {
            "stage": stage, "steps": steps, "batch_size": 1,
            "crop_seconds": 0.16, "workers": 0, "grad_accum": 2,
            "lr": 2e-4, "weight_decay": 0.01, "clip_grad_norm": 5.0,
            "sample_exit": depth > 1, "split_decoders_at": 1 if depth > 1 else None,
            "predecoder_lr_scale": 0.1,
        },
        "data": {"train_manifest": None, "validation_manifest": None, "pin_memory": False},
        "loss": {"waveform": 1.0, "stft": 0.1, "kd": 0.0, "fft_sizes": [16, 32]},
        "adversarial": {"channels": 4, "scales": 1, "lr": 2e-4,
                        "weight": 0.01, "feature_matching": 0.1},
    }


class OrderedSignals(Dataset):
    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        time = torch.arange(128) / 800
        clean = (0.12 + index * 0.01) * torch.sin(2 * torch.pi * (40 + index * 7) * time)
        noise = 0.025 * torch.randn(128, generator=torch.Generator().manual_seed(100 + index))
        return {"noisy": clean + noise, "clean": clean, "lengths": torch.tensor(128)}


class UpdateTrace(Callback):
    def __init__(self):
        self.rows = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.rows.append({
            "updates": int(pl_module.completed_updates),
            "global_step": int(trainer.global_step),
            "decoder_mode": pl_module.model.config.decoder_mode,
            "exits": list(getattr(pl_module, "last_exit_depths", [])),
            "partial_microbatches": int(pl_module.microbatches_in_update),
        })


def trainer_for(config, trace, *, validation: bool):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    strategy = (DDPStrategy(find_unused_parameters=True, static_graph=False,
                            broadcast_buffers=False, process_group_backend="gloo")
                if world_size > 1 else "auto")
    return L.Trainer(
        accelerator="cpu", devices=world_size, strategy=strategy,
        max_steps=config["training"]["steps"] * (2 if config["training"]["stage"] == "adversarial" else 1),
        max_epochs=-1, accumulate_grad_batches=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        callbacks=[trace], use_distributed_sampler=False,
        limit_val_batches=1.0 if validation else 0,
        val_check_interval=config["training"]["grad_accum"] if validation else 1.0,
        check_val_every_n_epoch=None if validation else 1,
    )


def artifact(module, trainer, trace):
    return {
        "state": {name: value.detach().cpu().clone() for name, value in module.state_dict().items()},
        "gradients": {name: parameter.grad.detach().cpu().clone()
                      for name, parameter in module.named_parameters() if parameter.grad is not None},
        "optimizers": [optimizer.state_dict() for optimizer in trainer.optimizers],
        "portable": module.portable_checkpoint(),
        "updates": int(module.completed_updates), "global_step": int(trainer.global_step),
        "decoder_mode": module.model.config.decoder_mode, "trace": trace.rows,
        "validation": {name: float(value) for name, value in trainer.callback_metrics.items()
                       if name.startswith("validation_")},
    }


def equivalence(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = tiny_config(args.stage, 1)
    L.seed_everything(config["seed"], workers=True)
    module = EnhancementModule(config)
    trace = UpdateTrace()
    trainer = trainer_for(config, trace, validation=False)
    # Each DDP microbatch contributes [0,1], then [2,3] globally. The single
    # process receives those exact pairs as batch=2, with the same accumulation.
    indices = range(rank, 4, world_size) if world_size > 1 else range(4)
    loader = DataLoader(OrderedSignals(indices), batch_size=1 if world_size > 1 else 2,
                        shuffle=False, num_workers=0)
    trainer.fit(module, train_dataloaders=loader)
    torch.save(artifact(module, trainer, trace), args.output / f"{args.stage}_rank{rank}.pt")


def lifecycle(args):
    rank = int(os.environ.get("RANK", "0"))
    base = tiny_config("regression", 1, depth=2)
    base["data"].update(train_manifest=str(args.train_manifest),
                        validation_manifest=str(args.validation_manifest))

    def phase(name, config, *, resume=None, init=None):
        L.seed_everything(config["seed"], workers=True)
        module = EnhancementModule(config)
        if init is not None:
            module.load_portable_weights(torch.load(init, map_location="cpu", weights_only=True))
        data = PairedAudioDataModule(config)
        trace = UpdateTrace()
        trainer = trainer_for(config, trace, validation=True)
        trainer.fit(module, datamodule=data, ckpt_path=str(resume) if resume else None)
        trainer.validate(module, datamodule=data, verbose=False)
        checkpoint = args.output / f"{name}.ckpt"
        # Every rank participates: checkpoint hooks may gather rank-specific RNG.
        trainer.save_checkpoint(checkpoint)
        torch.save(artifact(module, trainer, trace), args.output / f"{name}_rank{rank}.pt")
        trainer.strategy.barrier()
        if name == "adversarial_resumed":
            singleton = copy.deepcopy(config)
            singleton["data"]["validation_manifest"] = str(args.validation_manifest.with_name("validation-single.jsonl"))
            trainer.validate(module, datamodule=PairedAudioDataModule(singleton), verbose=False)
            torch.save(artifact(module, trainer, trace), args.output / f"validation_single_rank{rank}.pt")
        return checkpoint

    regression = phase("regression_first", copy.deepcopy(base))
    continued = copy.deepcopy(base)
    continued["training"]["steps"] = 3
    phase("regression_resumed", continued, resume=regression)
    adversarial = copy.deepcopy(base)
    adversarial["training"]["stage"] = "adversarial"
    first_gan = phase("adversarial_first", adversarial, init=regression)
    continued_gan = copy.deepcopy(adversarial)
    continued_gan["training"]["steps"] = 3
    phase("adversarial_resumed", continued_gan, resume=first_gan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("equivalence", "lifecycle"), required=True)
    parser.add_argument("--stage", choices=("regression", "adversarial"), default="regression")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "equivalence":
        equivalence(args)
    else:
        lifecycle(args)


if __name__ == "__main__":
    main()
