"""Regression, cached-output distillation, and teacher-free adversarial training.

The schedule and objectives here are independent engineering choices. A successful
smoke run verifies execution, not enhancement quality or reproduction of a paper.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, load_config
from .discriminators import MultiScaleDiscriminator
from .losses import ReconstructionLoss, discriminator_audio_loss, generator_audio_losses
from .model import Enhancer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: Enhancer, training: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(training.get("lr", 2e-4))
    weight_decay = float(training.get("weight_decay", 0.01))
    scale = float(training.get("predecoder_lr_scale", 0.1))
    if (not all(math.isfinite(value) for value in (lr, weight_decay, scale))
            or lr <= 0 or weight_decay < 0 or not 0 < scale <= 1):
        raise ValueError("lr must be positive, weight_decay >= 0, and predecoder_lr_scale in (0,1]")
    if model.config.decoder_mode == "per_exit":
        heads, backbone = [], []
        for name, parameter in model.named_parameters():
            (heads if "decoder" in name else backbone).append(parameter)
        if not heads or not backbone:
            raise ValueError("Per-exit optimizer requires named decoder and backbone parameters")
        groups = [{"params": backbone, "lr": lr * scale, "name": "backbone"},
                  {"params": heads, "lr": lr, "name": "decoders"}]
    else:
        groups = [{"params": list(model.parameters()), "lr": lr, "name": "shared"}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def transition_decoders(model: Enhancer, optimizer: torch.optim.Optimizer,
                        training: dict[str, Any]) -> torch.optim.Optimizer:
    """Clone the learned shared head; retain optimizer moments of surviving tensors."""
    if model.config.decoder_mode == "per_exit":
        return optimizer
    model.split_decoders()
    replacement = build_optimizer(model, training)
    for group in replacement.param_groups:
        for parameter in group["params"]:
            if parameter in optimizer.state:
                replacement.state[parameter] = optimizer.state[parameter]
    return replacement


def _assert_finite(value: Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {label}; update aborted. Inspect data/losses or use fp32.")


def _finite_gradients(module: nn.Module, label: str, max_norm: float) -> float:
    parameters = [parameter for parameter in module.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError(f"No gradients reached {label}")
    for parameter in parameters:
        _assert_finite(parameter.grad, f"{label} gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    return float(norm.detach())


def save_checkpoint(path: Path, model: Enhancer, optimizer: torch.optim.Optimizer,
                    *, step: int, stage: str, config: dict[str, Any],
                    discriminator: MultiScaleDiscriminator | None = None,
                    discriminator_optimizer: torch.optim.Optimizer | None = None,
                    scaler: Any = None, best_validation: float | None = None) -> None:
    """Store tensors and built-in values only, compatible with weights_only=True."""
    payload = {
        "format_version": 1, "model_config": model.config.to_dict(),
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "step": step, "stage": stage,
        "training_config": {**config, "model": model.config.to_dict()},
        "best_validation": best_validation, "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    if discriminator is not None:
        payload["discriminator_state"] = discriminator.state_dict()
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer_state"] = discriminator_optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _compatible_checkpoint_config(requested: ModelConfig, saved: dict[str, Any]) -> ModelConfig:
    restored = ModelConfig.from_dict(saved)
    left, right = requested.to_dict(), restored.to_dict()
    # A split is checkpoint state; execution backend may be changed without changing weights.
    for key in ("decoder_mode", "scan_backend"):
        left.pop(key, None)
        right.pop(key, None)
    if left != right:
        raise ValueError("Checkpoint architecture differs from config; use the same dimensions and exits")
    saved = dict(saved)
    saved["scan_backend"] = requested.scan_backend
    return ModelConfig.from_dict(saved)


def _loader_batches(loader: DataLoader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Training dataset is empty")


class _SmokeDataset(Dataset):
    """Tiny deterministic signals; never used unless --smoke is explicit."""

    def __init__(self, sample_rate: int, samples: int) -> None:
        self.sample_rate, self.samples = sample_rate, samples

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(index)
        time_axis = torch.arange(self.samples) / self.sample_rate
        clean = 0.15 * torch.sin(2 * torch.pi * (180 + index * 20) * time_axis)
        noisy = clean + 0.03 * torch.randn(self.samples, generator=generator)
        return {"noisy": noisy, "clean": clean, "lengths": torch.tensor(self.samples)}


@torch.no_grad()
def validate(model: Enhancer, loader: DataLoader, criterion: ReconstructionLoss,
             device: torch.device, max_batches: int) -> dict[str, float]:
    """Report clean reconstruction loss on fixed held-out crops, not MOS/PESQ."""
    if max_batches < 1:
        raise ValueError("validation_batches must be positive")
    previous_mode = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    try:
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            predictions = model.enhance(noisy)
            for exit_depth, prediction in predictions.items():
                loss, _ = criterion(prediction, clean, lengths=batch.get("lengths"))
                _assert_finite(loss, "validation loss")
                key = f"validation_clean_loss_exit_{exit_depth}"
                totals[key] = totals.get(key, 0.0) + float(loss) * len(noisy)
            count += len(noisy)
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Validation dataset is empty")
    result = {key: total / count for key, total in totals.items()}
    result["validation_mean_clean_loss"] = sum(result.values()) / len(result)
    return result


def run_training(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    training = dict(config.get("training", {}))
    loss_config = dict(config.get("loss", {}))
    adversarial = dict(config.get("adversarial", {}))
    stage = args.stage or training.get("stage", "regression")
    if stage not in ("regression", "adversarial"):
        raise ValueError("stage must be regression or adversarial")
    if args.resume and args.init_from:
        raise ValueError("Choose --resume or --init-from, not both")
    if args.smoke and (args.resume or args.init_from):
        raise ValueError("Smoke uses a tiny synthetic experiment; do not supply checkpoints")
    if stage == "adversarial" and not (args.resume or args.init_from or args.smoke):
        raise ValueError("Adversarial training requires --init-from a regression checkpoint or --resume")
    if not args.smoke and not args.manifest:
        raise ValueError("--manifest is required outside --smoke")
    steps = args.steps if args.steps is not None else int(training.get("steps", 100000))
    if args.smoke:
        steps = min(3, args.steps if args.steps is not None else 2)
        training.update(batch_size=1, grad_accum=1, precision="fp32", workers=0,
                        checkpoint_every=steps, log_every=1, split_decoders_at=None)
        loss_config["fft_sizes"] = [64, 128]
        loss_config["kd"] = 0.0
        adversarial.update(channels=4, scales=2)
        torch.set_num_threads(1)
    if steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(int(training.get("seed", 0)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a smoke test")
    precision = training.get("precision", "fp32")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("This trainer enables mixed precision only on CUDA; use fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but this CUDA device does not support it")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    autocast = lambda: (torch.autocast(device_type="cuda", dtype=dtype)
                        if precision != "fp32" else nullcontext())
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model_dict = dict(config.get("model", {}))
    if args.smoke:
        model_dict.update(depth=1, exits=[1], width=4, state_dim=2, expand=1,
                          decoder_mode="shared", scan_backend="reference")
    model_config = ModelConfig.from_dict(model_dict)
    checkpoint = None
    checkpoint_path = args.resume or args.init_from
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_config = _compatible_checkpoint_config(model_config, checkpoint["model_config"])
        if args.resume and checkpoint["stage"] != stage:
            raise ValueError("--resume must preserve stage; use --init-from to start another stage")
        if args.init_from and stage == "adversarial" and checkpoint["stage"] != "regression":
            raise ValueError("Stage-two --init-from requires a regression checkpoint; use --resume for GAN")
    model = Enhancer(model_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer = build_optimizer(model, training)
    discriminator = None
    discriminator_optimizer = None
    if stage == "adversarial":
        discriminator = MultiScaleDiscriminator(int(adversarial.get("channels", 16)),
                                                int(adversarial.get("scales", 3))).to(device)
        discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(),
                                                    lr=float(adversarial.get("lr", 2e-4)),
                                                    betas=(0.5, 0.9))
    start_step = 0
    best_validation = None
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        best_validation = checkpoint.get("best_validation")
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint["discriminator_state"], strict=True)
            discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    if start_step >= steps:
        raise ValueError("--steps is the total target optimizer-step count and must exceed checkpoint step")
    criterion = ReconstructionLoss(float(loss_config.get("waveform", 1)),
                                   float(loss_config.get("stft", 1)),
                                   loss_config.get("fft_sizes", (256, 512, 1024)))
    kd_weight = float(loss_config.get("kd", 0)) if stage == "regression" else 0.0
    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("loss.kd must be nonnegative")
    for key in ("weight", "feature_matching"):
        weight = float(adversarial.get(key, 0.01 if key == "weight" else 0.1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"adversarial.{key} must be nonnegative")
    crop_seconds = float(training.get("crop_seconds", 2.0))
    if not math.isfinite(crop_seconds) or crop_seconds <= 0:
        raise ValueError("crop_seconds must be positive")
    crop_samples = round(crop_seconds * model_config.sample_rate)
    validation_loader = None
    if args.smoke:
        dataset = _SmokeDataset(model_config.sample_rate,
                                max(128, round(model_config.sample_rate * 0.08)))
        collate = None
    else:
        from .data import PairedAudioDataset, collate_audio
        dataset = PairedAudioDataset(args.manifest, model_config.sample_rate,
                                     crop_samples=crop_samples, training=True,
                                     load_teacher=kd_weight > 0)
        collate = collate_audio
        if args.validation_manifest:
            validation_dataset = PairedAudioDataset(args.validation_manifest, model_config.sample_rate,
                                                    crop_samples=crop_samples, training=False,
                                                    load_teacher=False)
            validation_loader = DataLoader(validation_dataset, batch_size=int(training.get("batch_size", 4)),
                                           shuffle=False, num_workers=0, collate_fn=collate)
    batch_size = int(training.get("batch_size", 4))
    workers = int(training.get("workers", 0))
    accumulation = int(training.get("grad_accum", 1))
    clip_norm = float(training.get("clip_grad_norm", 5))
    checkpoint_every = int(training.get("checkpoint_every", 1000))
    log_every = int(training.get("log_every", 10))
    validate_every = int(training.get("validate_every", checkpoint_every))
    if (min(batch_size, accumulation, checkpoint_every, log_every, validate_every) < 1
            or workers < 0 or not math.isfinite(clip_norm) or clip_norm <= 0):
        raise ValueError("Batch, accumulation, intervals and clip norm must be positive; workers >= 0")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                        collate_fn=collate, pin_memory=device.type == "cuda")
    batches = _loader_batches(loader)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config.update(model=model.config.to_dict(), training=training, loss=loss_config, adversarial=adversarial)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"event": "start", "stage": stage, "start_step": start_step,
                      "target_step": steps, "device": str(device), "smoke": args.smoke,
                      "teacher": "cached_outputs" if kd_weight else "absent",
                      "resume_data_order_exact": False}), flush=True)
    model.train()
    split_at = training.get("split_decoders_at")
    if split_at is not None and int(split_at) < 0:
        raise ValueError("split_decoders_at must be nonnegative")
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a") as log:
        for step in range(start_step + 1, steps + 1):
            # Threshold means completed updates before the first per-exit update.
            if split_at is not None and step - 1 >= int(split_at):
                was_shared = model.config.decoder_mode == "shared"
                optimizer = transition_decoders(model, optimizer, training)
                if was_shared:
                    print(json.dumps({"event": "split_decoders", "step": step,
                                      "backbone_lr_scale": training.get("predecoder_lr_scale", 0.1)}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
            metrics: dict[str, float] = {}
            for _ in range(accumulation):
                batch = next(batches)
                noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
                lengths = batch.get("lengths")
                _assert_finite(noisy, "noisy input")
                _assert_finite(clean, "clean input")
                exit_depths = ([random.choice(model.config.exits)]
                               if training.get("sample_exit", True) else list(model.config.exits))
                with autocast():
                    predictions = model.enhance(noisy, exit_depths=exit_depths)
                    reconstructed, terms = [], {}
                    for prediction in predictions.values():
                        loss, details = criterion(prediction, clean, lengths=lengths)
                        reconstructed.append(loss)
                        for key, value in details.items():
                            terms[key] = terms.get(key, 0.0) + float(value) / len(predictions)
                    generator_loss = torch.stack(reconstructed).mean()
                    terms["clean_loss"] = float(generator_loss.detach())
                    if kd_weight:
                        if "teacher" not in batch:
                            raise ValueError("loss.kd > 0 requires aligned cached teacher audio on every row")
                        teacher = batch["teacher"].to(device)
                        _assert_finite(teacher, "cached teacher input")
                        kd = torch.stack([criterion(prediction, teacher, lengths=lengths)[0]
                                          for prediction in predictions.values()]).mean()
                        generator_loss = generator_loss + kd_weight * kd
                        terms["kd_loss"] = float(kd.detach())
                if discriminator is not None:
                    enhanced = torch.cat(list(predictions.values()), dim=0)
                    reference = torch.cat([clean] * len(predictions), dim=0)
                    critic_lengths = lengths.repeat(len(predictions)) if lengths is not None else None
                    with autocast():
                        d_loss = discriminator_audio_loss(discriminator, reference, enhanced, critic_lengths)
                    _assert_finite(d_loss, "discriminator loss")
                    scaler.scale(d_loss / accumulation).backward()
                    discriminator.requires_grad_(False)
                    try:
                        with autocast():
                            g_adversarial, matching = generator_audio_losses(
                                discriminator, reference, enhanced, critic_lengths)
                            generator_loss = (generator_loss
                                              + float(adversarial.get("weight", 0.01)) * g_adversarial
                                              + float(adversarial.get("feature_matching", 0.1)) * matching)
                        _assert_finite(generator_loss, "generator loss")
                        scaler.scale(generator_loss / accumulation).backward()
                    finally:
                        discriminator.requires_grad_(True)
                    terms.update(discriminator_loss=float(d_loss.detach()),
                                 adversarial_loss=float(g_adversarial.detach()),
                                 feature_matching_loss=float(matching.detach()))
                else:
                    _assert_finite(generator_loss, "regression loss")
                    scaler.scale(generator_loss / accumulation).backward()
                terms["generator_loss"] = float(generator_loss.detach())
                for key, value in terms.items():
                    metrics[key] = metrics.get(key, 0.0) + value / accumulation
            scaler.unscale_(optimizer)
            metrics["generator_grad_norm"] = _finite_gradients(model, "generator", clip_norm)
            if discriminator_optimizer is not None:
                scaler.unscale_(discriminator_optimizer)
                metrics["discriminator_grad_norm"] = _finite_gradients(discriminator, "discriminator", clip_norm)
                scaler.step(discriminator_optimizer)
            scaler.step(optimizer)
            scaler.update()
            metrics.update(step=step, stage=stage, elapsed_seconds=time.monotonic() - started)
            improved = False
            if validation_loader is not None and (step % validate_every == 0 or step == steps):
                values = validate(model, validation_loader, criterion, device,
                                  int(training.get("validation_batches", 8)))
                metrics.update(values)
                score = values["validation_mean_clean_loss"]
                if best_validation is None or score < best_validation:
                    best_validation, improved = score, True
            if step % log_every == 0 or step == steps or improved:
                serialized = json.dumps(metrics, allow_nan=False)
                log.write(serialized + "\n")
                log.flush()
                print(serialized, flush=True)
            if step % checkpoint_every == 0 or step == steps or improved:
                kwargs = dict(step=step, stage=stage, config=config,
                              discriminator=discriminator, discriminator_optimizer=discriminator_optimizer,
                              scaler=scaler, best_validation=best_validation)
                save_checkpoint(output / "last.pt", model, optimizer, **kwargs)
                if step % checkpoint_every == 0:
                    save_checkpoint(output / f"checkpoint_{step:07d}.pt", model, optimizer, **kwargs)
                if improved:
                    save_checkpoint(output / "best.pt", model, optimizer, **kwargs)
    return output / "last.pt"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--validation-manifest", help="Held-out paired audio; selects best clean-loss checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Continue the same stage; data iterator order is restarted")
    parser.add_argument("--init-from", help="Load model weights and start a fresh stage/optimizer at step zero")
    parser.add_argument("--stage", choices=("regression", "adversarial"))
    parser.add_argument("--steps", type=int, help="Total target optimizer-step count, including resumed steps")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="At most three tiny synthetic steps, no quality claim")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = run_training(args)
    print(json.dumps({"event": "complete", "checkpoint": str(result)}), flush=True)


if __name__ == "__main__":
    main()
