"""Lightning training for independently implemented causal speech enhancement.

Manual optimization keeps generator-update counts explicit for both regression
and two-optimizer GAN training. Checkpoints are taken only between updates.
"""

from __future__ import annotations

import copy
from functools import partial
import math
import random
from typing import Any

from lightning.pytorch import LightningModule
from lightning.pytorch.strategies import DDPStrategy
import numpy as np
import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from .config import ModelConfig
from .discriminators import build_discriminator
from .losses import ReconstructionLoss, discriminator_audio_loss, generator_audio_losses
from .model import Enhancer


def _finite(value: Tensor, name: str) -> None:
    finite = torch.isfinite(value).all().to(dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite):
        raise FloatingPointError(f"Nonfinite {name}; update aborted. Inspect data or use full precision.")


def _finite_gradients(gradients: list[Tensor], device: torch.device, name: str) -> None:
    finite = (torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()
              if gradients else torch.zeros((), dtype=torch.bool, device=device)).to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite):
        raise FloatingPointError(f"Missing or nonfinite {name}; update aborted on every rank")


class EnhancementModule(LightningModule):
    """Regression/KD or teacher-free GAN with stable decoder parameter registration."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.save_hyperparameters({"config": self.config})
        self.automatic_optimization = False
        self.training_config = self.config.get("training", {})
        self.loss_config = self.config.get("loss", {})
        self.adversarial_config = self.config.get("adversarial", {})
        self.validation_config = self.config.get("validation", {})
        self.stage = self.training_config.get("stage", "regression")
        if self.stage not in {"regression", "adversarial"}:
            raise ValueError("training.stage must be regression or adversarial")
        self.accumulation = int(self.training_config.get("grad_accum", 1))
        self.target_updates = int(self.training_config.get("steps", 100000))
        self.clip_norm = float(self.training_config.get("clip_grad_norm", 5.0))
        self.base_lr = float(self.training_config.get("lr", 2e-4))
        self.backbone_lr_scale = float(self.training_config.get("predecoder_lr_scale", 0.1))
        self.weight_decay = float(self.training_config.get("weight_decay", 0.01))
        numbers = (self.clip_norm, self.base_lr, self.backbone_lr_scale, self.weight_decay)
        if (self.accumulation < 1 or self.target_updates < 1
                or not all(math.isfinite(number) for number in numbers)
                or self.clip_norm <= 0 or self.base_lr <= 0 or self.weight_decay < 0
                or not 0 < self.backbone_lr_scale <= 1):
            raise ValueError("Invalid accumulation, step count, clipping, or optimizer configuration")
        self.split_at = self.training_config.get("split_decoders_at")
        if self.split_at is not None and int(self.split_at) < 0:
            raise ValueError("split_decoders_at must be nonnegative")
        self.kd_weight = float(self.loss_config.get("kd", 0)) if self.stage == "regression" else 0.0
        self.adversarial_weight = float(self.adversarial_config.get("weight", 0.01))
        self.matching_weight = float(self.adversarial_config.get("feature_matching", 0.1))
        if any(not math.isfinite(value) or value < 0 for value in
               (self.kd_weight, self.adversarial_weight, self.matching_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        self.model = Enhancer(ModelConfig.from_dict(self.config["model"]))
        # All parameters and optimizer groups exist before Trainer/precision setup.
        self.model.prepare_decoder_transition()
        self._set_active_head_gradients()
        self.criterion = ReconstructionLoss(float(self.loss_config.get("waveform", 1)),
                                            float(self.loss_config.get("stft", 1)),
                                            self.loss_config.get("fft_sizes", (256, 512, 1024)),
                                            implementation=self.loss_config.get("implementation", "legacy"))
        rates = self.model.config.sample_rates or (self.model.config.sample_rate,)
        # These loss modules have no learned parameters. Register every rate
        # before device/DDP setup rather than constructing modules in forward.
        self.rate_criteria = nn.ModuleDict()
        if self.loss_config.get("sample_rate_scaling", False):
            for rate in rates:
                if rate != self.model.config.sample_rate:
                    sizes = [max(8, round(size * rate / self.model.config.sample_rate))
                             for size in self.loss_config.get("fft_sizes", (256, 512, 1024))]
                    self.rate_criteria[str(rate)] = ReconstructionLoss(
                        float(self.loss_config.get("waveform", 1)), float(self.loss_config.get("stft", 1)),
                        sizes, implementation=self.loss_config.get("implementation", "legacy"))
        self.discriminator = (build_discriminator(self.adversarial_config, self.model.config.sample_rate, rates)
                              if self.stage == "adversarial" else None)
        self.register_buffer("completed_updates", torch.zeros((), dtype=torch.long))
        self.microbatches_in_update = 0
        self._update_metrics: dict[str, Tensor] = {}
        self._generator_optimizer: torch.optim.Optimizer | None = None
        self.validation_metrics = nn.ModuleDict({
            str(depth): MeanMetric(nan_strategy="error").set_dtype(torch.float64)
            for depth in self.model.config.exits
        })
        configured_lookaheads = self.validation_config.get("lookaheads")
        self.validation_lookaheads = tuple(self.model.config.lookahead_frames
                                         if configured_lookaheads is None else configured_lookaheads)
        if (not self.validation_lookaheads or len(set(self.validation_lookaheads)) != len(self.validation_lookaheads)
                or any(value not in self.model.config.lookahead_frames for value in self.validation_lookaheads)):
            raise ValueError("validation.lookaheads must be a nonempty subset of model.lookahead_frames")
        self.validation_extra_metrics = nn.ModuleDict({
            f"{depth}_{lookahead}": MeanMetric(nan_strategy="error").set_dtype(torch.float64)
            for depth in self.model.config.exits for lookahead in self.validation_lookaheads if lookahead != 0
        })
        self.quality_names = tuple(self.validation_config.get("metrics", ()))
        self.validation_quality_metrics = nn.ModuleDict({
            f"{name}_{depth}_{lookahead}": MeanMetric(nan_strategy="error").set_dtype(torch.float64)
            for name in self.quality_names for depth in self.model.config.exits
            for lookahead in self.validation_lookaheads
        })
        self._quality_evaluator = None
        self.last_exit_depths: tuple[int, ...] = ()
        self.last_lookahead = 0

    def _set_active_head_gradients(self) -> None:
        # DDP installs reducer hooks only on parameters requiring gradients at
        # wrapper construction. Keep future heads registered and trainable;
        # find_unused_parameters handles the heads absent from the current graph.
        self.model.shared_decoder.requires_grad_(True)
        self.model.decoders.requires_grad_(True)

    def forward(self, waveform: Tensor, exit_depths=None, sample_rate=None, lookahead=0) -> dict[int, Tensor]:
        return self.model.enhance(waveform, exit_depths=exit_depths,
                                  sample_rate=sample_rate, lookahead=lookahead)

    def _batch_rate(self, batch: dict[str, Any]) -> int:
        rate = batch.get("sample_rate", self.model.config.sample_rate)
        if isinstance(rate, Tensor):
            if rate.numel() != 1:
                raise ValueError("A batch must have one homogeneous sample_rate")
            rate = rate.item()
        if (isinstance(rate, bool) or not isinstance(rate, int)
                or rate not in (self.model.config.sample_rates or (self.model.config.sample_rate,))):
            raise ValueError("Batch sample_rate must be an integer supported by the model")
        return rate

    def _criterion_for_rate(self, rate: int) -> ReconstructionLoss:
        return self.rate_criteria[str(rate)] if str(rate) in self.rate_criteria else self.criterion

    def _ensure_quality_evaluator(self) -> None:
        if self.quality_names and self._quality_evaluator is None:
            from .quality_metrics import QualityEvaluator
            self._quality_evaluator = QualityEvaluator(
                metrics=list(self.quality_names), device=str(self.device),
                config=self.validation_config.get("quality", {}))

    def configure_optimizers(self):
        backbone, heads, backbone_no_decay, heads_no_decay = [], [], [], []
        for name, parameter in self.model.named_parameters():
            if getattr(parameter, "_no_weight_decay", False):
                (heads_no_decay if "decoder" in name else backbone_no_decay).append(parameter)
            else:
                (heads if "decoder" in name else backbone).append(parameter)
        backbone_lr = self.base_lr * (self.backbone_lr_scale
                                      if self.model.config.decoder_mode == "per_exit" else 1.0)
        groups = [
            {"params": backbone, "lr": backbone_lr, "name": "backbone"},
            {"params": heads, "lr": self.base_lr, "name": "decoders"},
        ]
        # Upstream Mamba exempts its state dynamics and skip parameters from
        # weight decay. Leave unflagged legacy models' two groups unchanged so
        # their existing native optimizer checkpoints remain compatible.
        for parameters, learning_rate, name in (
                (backbone_no_decay, backbone_lr, "backbone"),
                (heads_no_decay, self.base_lr, "decoders")):
            if parameters:
                groups.append({"params": parameters, "lr": learning_rate,
                               "name": name, "weight_decay": 0.0})
        generator = torch.optim.AdamW(groups, lr=self.base_lr, weight_decay=self.weight_decay)
        self._generator_optimizer = generator
        if self.discriminator is None:
            return generator
        discriminator_lr = float(self.adversarial_config.get("lr", 2e-4))
        if not math.isfinite(discriminator_lr) or discriminator_lr <= 0:
            raise ValueError("adversarial.lr must be finite and positive")
        critic = torch.optim.AdamW(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.9))
        return [generator, critic]

    def on_fit_start(self) -> None:
        # Fail missing local metric assets before the first optimizer update.
        self._ensure_quality_evaluator()
        if self.trainer.world_size != 1:
            wrapped = self.trainer.strategy.model
            if (not isinstance(self.trainer.strategy, DDPStrategy)
                    or not isinstance(wrapped, DistributedDataParallel)
                    or not wrapped.find_unused_parameters or wrapped.static_graph):
                raise RuntimeError("Multi-device training requires DDP with find_unused_parameters=True and static_graph=False")
        if self.trainer.accumulate_grad_batches != 1:
            raise ValueError("Set Trainer.accumulate_grad_batches=1; training.grad_accum controls manual accumulation")
        self._set_active_head_gradients()
        if int(self.completed_updates) >= self.target_updates:
            raise ValueError("training.steps must exceed the completed generator updates in the checkpoint")

    def _maybe_split_decoders(self, optimizer) -> None:
        if (self.split_at is not None and int(self.completed_updates) >= int(self.split_at)
                and self.model.config.decoder_mode == "shared"):
            self.model.activate_prepared_decoders()
            self._set_active_head_gradients()
            for group in optimizer.param_groups:
                if group["name"] == "backbone":
                    group["lr"] = self.base_lr * self.backbone_lr_scale

    def _optimizers(self):
        optimizers = self.optimizers()
        if self.discriminator is None:
            return optimizers, None
        return optimizers[0], optimizers[1]

    def _select_exit_depths(self) -> list[int]:
        if self.training_config.get("sample_exit", True):
            distributed = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if distributed else 0
            selected = random.choice(self.model.config.exits) if rank == 0 else 0
            choice = torch.tensor(selected, device=self.device, dtype=torch.long)
            if distributed:
                dist.broadcast(choice, src=0)
            result = [int(choice)]
        else:
            result = list(self.model.config.exits)
        self.last_exit_depths = tuple(result)
        return result

    def _select_lookahead(self) -> int:
        if self.training_config.get("sample_lookahead", False):
            distributed = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if distributed else 0
            selected = random.choice(self.model.config.lookahead_frames) if rank == 0 else 0
            choice = torch.tensor(selected, device=self.device, dtype=torch.long)
            if distributed:
                dist.broadcast(choice, src=0)
            selected = int(choice)
        else:
            selected = self.training_config.get("lookahead", 0)
        if isinstance(selected, bool) or selected not in self.model.config.lookahead_frames:
            raise ValueError("training.lookahead must select a configured lookahead branch")
        self.last_lookahead = selected
        return selected

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        generator_optimizer, discriminator_optimizer = self._optimizers()
        if self.microbatches_in_update == 0:
            self._maybe_split_decoders(generator_optimizer)
            generator_optimizer.zero_grad(set_to_none=True)
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
            self._update_metrics = {}
        noisy, clean, lengths = batch["noisy"], batch["clean"], batch.get("lengths")
        _finite(noisy, "noisy input")
        _finite(clean, "clean input")
        exits = self._select_exit_depths()
        rate, lookahead = self._batch_rate(batch), self._select_lookahead()
        criterion = self._criterion_for_rate(rate)
        predictions = self(noisy, exit_depths=exits, sample_rate=rate, lookahead=lookahead)
        reconstruction, metrics = [], {}
        for prediction in predictions.values():
            loss, details = criterion(prediction, clean, lengths=lengths)
            reconstruction.append(loss)
            for key, value in details.items():
                metrics[key] = metrics.get(key, 0.0) + value / len(predictions)
        generator_loss = torch.stack(reconstruction).mean()
        metrics["clean_loss"] = generator_loss.detach()
        if self.kd_weight:
            if "teacher" not in batch:
                raise ValueError("loss.kd > 0 requires aligned cached teacher audio")
            _finite(batch["teacher"], "teacher target")
            kd = torch.stack([criterion(prediction, batch["teacher"], lengths=lengths)[0]
                              for prediction in predictions.values()]).mean()
            generator_loss = generator_loss + self.kd_weight * kd
            metrics["kd_loss"] = kd.detach()
        if self.discriminator is not None:
            critic = (partial(self.discriminator, sample_rate=rate)
                      if self.adversarial_config.get("implementation") == "dac_mrd" else self.discriminator)
            enhanced = torch.cat(list(predictions.values()), dim=0)
            reference = torch.cat([clean] * len(predictions), dim=0)
            critic_lengths = lengths.repeat(len(predictions)) if lengths is not None else None
            self.toggle_optimizer(discriminator_optimizer)
            try:
                critic_loss = discriminator_audio_loss(critic, reference, enhanced, critic_lengths)
                _finite(critic_loss, "discriminator loss")
                self.manual_backward(critic_loss / self.accumulation)
            finally:
                self.untoggle_optimizer(discriminator_optimizer)
            self.toggle_optimizer(generator_optimizer)
            try:
                adversarial, matching = generator_audio_losses(critic, reference, enhanced, critic_lengths)
                generator_loss = (generator_loss + self.adversarial_weight * adversarial
                                  + self.matching_weight * matching)
                _finite(generator_loss, "generator loss")
                self.manual_backward(generator_loss / self.accumulation)
            finally:
                self.untoggle_optimizer(generator_optimizer)
            metrics.update(discriminator_loss=critic_loss.detach(), adversarial_loss=adversarial.detach(),
                           feature_matching_loss=matching.detach())
        else:
            _finite(generator_loss, "regression loss")
            self.manual_backward(generator_loss / self.accumulation)
        metrics["generator_loss"] = generator_loss.detach()
        for key, value in metrics.items():
            self._update_metrics[key] = self._update_metrics.get(key, 0.0) + value / self.accumulation
        self.microbatches_in_update += 1
        if self.microbatches_in_update == self.accumulation:
            # Catch invalid accumulated gradients before either GAN optimizer can update.
            _finite_gradients([parameter.grad for parameter in self.parameters() if parameter.grad is not None],
                              self.device, "accumulated gradients")
            if discriminator_optimizer is not None:
                scaler = getattr(self.trainer.precision_plugin, "scaler", None)
                old_scale = scaler.get_scale() if scaler is not None else None
                discriminator_optimizer.step()
                # Lightning updates its shared FP16 scaler after each optimizer.
                # Pending G gradients were accumulated at the pre-D scale.
                if scaler is not None and scaler.get_scale() != old_scale:
                    ratio = scaler.get_scale() / old_scale
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(ratio)
            generator_optimizer.step()
            self.completed_updates.add_(1)
            self.microbatches_in_update = 0
            self.log_dict(self._update_metrics, on_step=True, on_epoch=False,
                          batch_size=len(noisy), logger=self.logger is not None,
                          sync_dist=self.trainer.world_size > 1)
            self.log("completed_updates", self.completed_updates.float(), on_step=True, on_epoch=False,
                     batch_size=len(noisy), logger=self.logger is not None,
                     sync_dist=self.trainer.world_size > 1)
        return generator_loss.detach()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Lightning invokes this after AMP unscaling and before the optimizer update.
        gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"]
                     if parameter.grad is not None]
        _finite_gradients(gradients, self.device, "unscaled gradients")
        self.clip_gradients(optimizer, gradient_clip_val=self.clip_norm, gradient_clip_algorithm="norm")
        _finite_gradients(gradients, self.device, "clipped gradients")

    def on_validation_epoch_start(self) -> None:
        if self.microbatches_in_update:
            raise RuntimeError("Validate only between optimizer updates; align validation interval with grad_accum")
        self._ensure_quality_evaluator()
        for metric in (*self.validation_metrics.values(), *self.validation_extra_metrics.values(),
                       *self.validation_quality_metrics.values()):
            metric.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        # Equal-sized padded batches execute on every rank. Padding duplicates
        # participate in forward calls but never in validation numerators/counts.
        rate = self._batch_rate(batch)
        criterion = self._criterion_for_rate(rate)
        valid = batch.get("distributed_valid")
        valid = (torch.ones(len(batch["noisy"]), dtype=torch.bool, device=self.device)
                 if valid is None else valid.to(device=self.device, dtype=torch.bool))
        if valid.ndim != 1 or len(valid) != len(batch["noisy"]):
            raise ValueError("distributed_valid must contain one boolean per validation utterance")
        count = int(valid.sum())
        lengths = batch.get("lengths")
        lengths = lengths.to(self.device)[valid] if lengths is not None else None
        for lookahead in self.validation_lookaheads:
            predictions = self(batch["noisy"], sample_rate=rate, lookahead=lookahead)
            for depth, prediction in predictions.items():
                loss = prediction.new_zeros(())
                if count:
                    loss, _ = criterion(prediction[valid], batch["clean"][valid], lengths=lengths)
                # All ranks participate, including zero-weight padding batches.
                _finite(loss, "validation loss")
                metric = (self.validation_metrics[str(depth)] if lookahead == 0
                          else self.validation_extra_metrics[f"{depth}_{lookahead}"])
                metric.update(loss.detach(), weight=count)
                if self._quality_evaluator is not None:
                    scores = {name: [] for name in self.quality_names}
                    for index in torch.nonzero(valid, as_tuple=True)[0].tolist():
                        length = int(batch["lengths"][index]) if "lengths" in batch else prediction.shape[-1]
                        with torch.inference_mode():
                            measured = self._quality_evaluator.score(
                                prediction[index, :length], batch["clean"][index, :length], sample_rate=rate)
                        for name in self.quality_names:
                            scores[name].append(measured[name])
                    for name, values in scores.items():
                        value = torch.tensor(sum(values) / count if count else 0.0,
                                             device=self.device, dtype=torch.float64)
                        _finite(value, f"validation {name}")
                        self.validation_quality_metrics[f"{name}_{depth}_{lookahead}"].update(value, weight=count)

    def on_validation_epoch_end(self) -> None:
        # MeanMetric synchronizes its weighted sum and weight on every rank,
        # including ranks that only saw zero-weight validation duplicates.
        metrics = {}
        for lookahead in self.validation_lookaheads:
            for depth in self.model.config.exits:
                key = f"validation_clean_loss_exit_{depth}"
                metric = self.validation_metrics[str(depth)] if lookahead == 0 else self.validation_extra_metrics[f"{depth}_{lookahead}"]
                if lookahead:
                    key += f"_lookahead_{lookahead}"
                metrics[key] = metric.compute()
        values = torch.stack(list(metrics.values()))
        if bool(torch.isnan(values).all()):
            # All updates were finite with nonnegative weights, so 0/0 means
            # the global validation input contained no original utterances.
            return
        _finite(values, "aggregated validation loss")
        metrics["validation_mean_clean_loss"] = values.mean()
        for name in self.quality_names:
            scores = {}
            for depth in self.model.config.exits:
                for lookahead in self.validation_lookaheads:
                    scores[f"validation_{name}_exit_{depth}_lookahead_{lookahead}"] = (
                        self.validation_quality_metrics[f"{name}_{depth}_{lookahead}"].compute())
            quality = torch.stack(list(scores.values()))
            _finite(quality, f"aggregated validation {name}")
            metrics.update(scores)
            metrics[f"validation_mean_{name}"] = quality.mean()
        # The computed TorchMetrics values are already synchronized globally.
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=self.logger is not None, sync_dist=False)

    def _portable_model_state(self) -> dict[str, Tensor]:
        inactive = "decoders." if self.model.config.decoder_mode == "shared" else "shared_decoder."
        return {key: value for key, value in self.model.state_dict().items() if not key.startswith(inactive)}

    def portable_checkpoint(self) -> dict[str, Any]:
        return {"format_version": 2, "model_config": self.model.config.to_dict(),
                "model_state": self._portable_model_state(), "stage": self.stage,
                "step": int(self.completed_updates),
                "training_config": {**self.config, "model": self.model.config.to_dict()}}

    def load_portable_weights(self, payload: dict[str, Any]) -> None:
        """Initialize active weights without importing optimizer state or update counts."""
        saved = ModelConfig.from_dict(payload["model_config"])
        requested, actual = self.model.config.to_dict(), saved.to_dict()
        for key in ("decoder_mode", "scan_backend"):
            requested.pop(key)
            actual.pop(key)
        if requested != actual:
            raise ValueError("Checkpoint architecture differs from the requested model")
        self.model.config.decoder_mode = saved.decoder_mode
        expected = set(self._portable_model_state())
        if set(payload["model_state"]) != expected:
            raise ValueError("Portable checkpoint has missing or unexpected model parameters")
        combined = self.model.state_dict()
        combined.update(payload["model_state"])
        self.model.load_state_dict(combined, strict=True)
        self._set_active_head_gradients()

    def _rng_device(self) -> torch.device:
        # Lightning restores before moving the module to its device. The
        # strategy already knows the rank's assigned accelerator at that point.
        return self.trainer.strategy.root_device if self._trainer is not None else self.device

    def capture_rng_state(self) -> dict[str, Any]:
        """Capture this rank's Python, NumPy and PyTorch RNGs without a collective."""
        state = {"python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state()}
        numpy_state = np.random.get_state()
        # Builtins keep weights_only checkpoint loading safe. NumPy ndarray
        # pickles would otherwise require additional deserialization globals.
        state["numpy_rng_state"] = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
        rng_device = self._rng_device()
        if rng_device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state(rng_device)
        return state

    def restore_rng_state(self, state: dict[str, Any]) -> None:
        """Restore this rank's RNGs; older checkpoints may omit NumPy state."""
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if "numpy_rng_state" in state:
            numpy_state = state["numpy_rng_state"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        rng_device = self._rng_device()
        if rng_device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), rng_device)
        elif torch.cuda.is_available() and "cuda_rng_states" in state:
            if len(state["cuda_rng_states"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.microbatches_in_update:
            raise RuntimeError("Checkpoint only at completed updates; partial accumulated gradients are not saved")
        checkpoint.update(self.portable_checkpoint())
        local_rng = self.capture_rng_state()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank_states: list[Any] = [None] * world_size
        if world_size > 1:
            # Trainer.save_checkpoint must be called by every rank; its strategy
            # restricts actual filesystem writes to global rank zero.
            dist.all_gather_object(rank_states, local_rng)
        else:
            rank_states[0] = local_rng
        checkpoint["rng_states_by_rank"] = rank_states
        checkpoint["world_size"] = world_size

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("stage") != self.stage:
            raise ValueError("Native resume must preserve stage; use portable initialization for a new stage")
        self.model.config.decoder_mode = checkpoint["model_config"]["decoder_mode"]
        self._set_active_head_gradients()
        self.microbatches_in_update = 0
        world_size = self.trainer.world_size if self._trainer is not None else 1
        if int(checkpoint.get("world_size", 1)) != world_size:
            raise ValueError("Native resume requires the same world size; use init_from for a fresh run at a different world size")
        rank = self.trainer.global_rank if self._trainer is not None else 0
        states = checkpoint.get("rng_states_by_rank")
        local_rng = states[rank] if states is not None else checkpoint
        self.restore_rng_state(local_rng)
