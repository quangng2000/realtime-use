"""Lightning lifecycle tests on small CPU models, without quality claims."""

import copy
import random
from types import SimpleNamespace

from lightning.pytorch import Trainer
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.aggregation import MeanMetric

from realtime_use.config import ModelConfig
from realtime_use.infer import load_model
from realtime_use.lightning_module import EnhancementModule


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(stage="regression", **training):
    return {
        "model": ModelConfig(sample_rate=800, depth=2, exits=(1, 2), width=4,
                             state_dim=2, expand=1, conv_kernel=2).to_dict(),
        "training": {"stage": stage, "steps": 2, "grad_accum": 2, "lr": 1e-3,
                     "sample_exit": False, "split_decoders_at": 1, **training},
        "loss": {"waveform": 1.0, "stft": 0.1, "fft_sizes": [16, 32], "kd": 0.0},
        "adversarial": {"channels": 4, "scales": 1, "weight": 0.01, "feature_matching": 0.1},
    }


def test_numpy_augmentation_rng_survives_weights_only_checkpoint(tmp_path):
    module = EnhancementModule(tiny_config())
    original = np.random.get_state()
    try:
        np.random.seed(87)
        np.random.normal(size=7)
        state = module.capture_rng_state()
        expected = np.random.normal(size=13)
        path = tmp_path / "rng.pt"
        torch.save(state, path)
        np.random.seed(123)
        module.restore_rng_state(torch.load(path, weights_only=True))
        np.testing.assert_array_equal(np.random.normal(size=13), expected)
    finally:
        np.random.set_state(original)


class TinyDataset(Dataset):
    def __init__(self, teacher=False, invalid_teacher=False):
        self.teacher, self.invalid_teacher = teacher, invalid_teacher

    def __len__(self):
        return 8

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        length = 73 + index % 3
        clean = 0.1 * torch.randn(128, generator=generator)
        noisy = clean + 0.02 * torch.randn(128, generator=generator)
        clean[length:] = 0
        noisy[length:] = 0
        row = {"noisy": noisy, "clean": clean, "lengths": torch.tensor(length)}
        if self.teacher:
            row["teacher"] = torch.full_like(clean, float("nan")) if self.invalid_teacher else clean.clone()
        return row


def trainer(tmp_path, updates, stage="regression"):
    return Trainer(accelerator="cpu", devices=1, precision="32-true", logger=False,
                   enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                   default_root_dir=tmp_path, num_sanity_val_steps=0, max_epochs=-1,
                   max_steps=updates * (2 if stage == "adversarial" else 1), accumulate_grad_batches=1)


def test_regression_kd_accumulation_transition_and_portable_checkpoint(tmp_path):
    config = tiny_config()
    config["loss"]["kd"] = 0.5
    module = EnhancementModule(config)
    parameter_ids = {name: id(parameter) for name, parameter in module.named_parameters()}
    initial = next(module.model.encoder.parameters()).detach().clone()
    runner = trainer(tmp_path, 2)
    loader = DataLoader(TinyDataset(teacher=True), batch_size=2)
    runner.fit(module, train_dataloaders=loader)
    assert module.completed_updates.item() == runner.global_step == 2
    assert module.microbatches_in_update == 0
    assert module.model.config.decoder_mode == "per_exit"
    assert parameter_ids == {name: id(parameter) for name, parameter in module.named_parameters()}
    assert not torch.equal(next(module.model.encoder.parameters()), initial)
    assert runner.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-4)
    assert "kd_loss" in runner.callback_metrics
    runner.validate(module, dataloaders=loader, verbose=False)
    assert runner.callback_metrics["validation_mean_clean_loss"] >= 0
    checkpoint = tmp_path / "native.ckpt"
    runner.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["step"] == 2 and payload["stage"] == "regression"
    assert not any(key.startswith("shared_decoder.") for key in payload["model_state"])
    restored = load_model(checkpoint)
    with torch.inference_mode():
        waveform = next(iter(loader))["noisy"]
        for depth, output in module.model.enhance(waveform).items():
            torch.testing.assert_close(restored.enhance(waveform)[depth], output)


def test_native_resume_after_split_keeps_optimizer_groups_and_update_count(tmp_path):
    config = tiny_config(split_decoders_at=0, steps=1, grad_accum=1)
    first = EnhancementModule(config)
    runner = trainer(tmp_path, 1)
    loader = DataLoader(TinyDataset(), batch_size=2)
    runner.fit(first, train_dataloaders=loader)
    checkpoint = tmp_path / "split.ckpt"
    runner.save_checkpoint(checkpoint)
    config["training"]["steps"] = 2
    config["model"]["decoder_mode"] = "per_exit"
    resumed = EnhancementModule(config)
    continued = trainer(tmp_path, 2)
    continued.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed.completed_updates.item() == continued.global_step == 2
    assert resumed.model.config.decoder_mode == "per_exit"
    assert len(continued.optimizers[0].param_groups) == 2
    assert continued.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-4)


def test_optimizer_honors_parameter_no_weight_decay_flags():
    module = EnhancementModule(tiny_config(weight_decay=0.2))
    named = dict(module.model.named_parameters())
    exempt = [next(parameter for name, parameter in named.items() if "decoder" not in name),
              next(parameter for name, parameter in named.items() if "decoder" in name)]
    for parameter in exempt:
        parameter._no_weight_decay = True
    optimizer = module.configure_optimizers()
    groups = optimizer.param_groups
    assert [(group["name"], group["weight_decay"]) for group in groups] == [
        ("backbone", 0.2), ("decoders", 0.2), ("backbone", 0.0), ("decoders", 0.0)]
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(named)
    assert set(all_ids) == {id(parameter) for parameter in named.values()}
    before = {id(parameter): parameter.detach().clone() for parameter in named.values()}
    for parameter in named.values():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    exempt_ids = {id(parameter) for parameter in exempt}
    for group in groups:
        for parameter in group["params"]:
            expected = before[id(parameter)]
            if id(parameter) not in exempt_ids:
                expected = expected * (1 - group["lr"] * group["weight_decay"])
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_decoder_transition_reduces_lr_for_decay_and_no_decay_backbone_groups():
    module = EnhancementModule(tiny_config(split_decoders_at=0))
    next(module.model.encoder.parameters())._no_weight_decay = True
    optimizer = module.configure_optimizers()
    identities = [[id(parameter) for parameter in group["params"]] for group in optimizer.param_groups]
    module._maybe_split_decoders(optimizer)
    assert module.model.config.decoder_mode == "per_exit"
    assert identities == [[id(parameter) for parameter in group["params"]] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        expected = module.base_lr * (0.1 if group["name"] == "backbone" else 1.0)
        assert group["lr"] == pytest.approx(expected)


def test_teacher_free_gan_counts_generator_updates_and_resumes(tmp_path):
    config = tiny_config(stage="adversarial", steps=1, split_decoders_at=None)
    config["loss"]["kd"] = 2.0  # Stage two must ignore even invalid supplied teacher tensors.
    module = EnhancementModule(config)
    initial = next(module.model.encoder.parameters()).detach().clone()
    runner = trainer(tmp_path, 1, "adversarial")
    loader = DataLoader(TinyDataset(teacher=True, invalid_teacher=True), batch_size=2)
    runner.fit(module, train_dataloaders=loader)
    assert module.completed_updates.item() == 1
    assert runner.global_step == 2
    assert not torch.equal(next(module.model.encoder.parameters()), initial)
    assert runner.optimizers[1].state
    assert "kd_loss" not in runner.callback_metrics
    checkpoint = tmp_path / "gan.ckpt"
    runner.save_checkpoint(checkpoint)
    config["training"]["steps"] = 2
    resumed = EnhancementModule(config)
    continued = trainer(tmp_path, 2, "adversarial")
    continued.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed.completed_updates.item() == 2
    assert continued.global_step == 4


def test_portable_initialization_copies_only_model_state_and_resets_updates():
    config = tiny_config(split_decoders_at=None)
    source = EnhancementModule(config)
    source.completed_updates.fill_(10)
    source.model.activate_prepared_decoders()
    payload = source.portable_checkpoint()
    config["training"]["stage"] = "adversarial"
    destination = EnhancementModule(config)
    destination.load_portable_weights(payload)
    assert destination.completed_updates.item() == 0
    assert destination.stage == "adversarial"
    assert destination.model.config.decoder_mode == "per_exit"
    for key, value in destination._portable_model_state().items():
        torch.testing.assert_close(value, payload["model_state"][key])
    bad = copy.deepcopy(payload)
    bad["model_config"]["width"] = 8
    with pytest.raises(ValueError, match="architecture"):
        destination.load_portable_weights(bad)


@pytest.mark.parametrize("stage", ["regression", "adversarial"])
def test_explicit_library_loss_trains_and_survives_native_resume(tmp_path, stage):
    config = tiny_config(stage=stage, steps=1, split_decoders_at=None)
    config["loss"]["implementation"] = "auraloss"
    module = EnhancementModule(config)
    assert module.criterion.implementation == "auraloss"
    initial = next(module.model.encoder.parameters()).detach().clone()
    runner = trainer(tmp_path, 1, stage)
    loader = DataLoader(TinyDataset(), batch_size=2)
    runner.fit(module, train_dataloaders=loader)
    assert not torch.equal(next(module.model.encoder.parameters()), initial)
    checkpoint = tmp_path / "library.ckpt"
    runner.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["hyper_parameters"]["config"]["loss"]["implementation"] == "auraloss"
    config["training"]["steps"] = 2
    resumed = EnhancementModule(config)
    continued = trainer(tmp_path, 2, stage)
    continued.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed.completed_updates.item() == 2
    assert resumed.criterion.implementation == "auraloss"


def test_partial_update_checkpoint_and_stage_resume_are_rejected():
    module = EnhancementModule(tiny_config())
    module.microbatches_in_update = 1
    with pytest.raises(RuntimeError, match="partial accumulated"):
        module.on_save_checkpoint({})
    with pytest.raises(ValueError, match="preserve stage"):
        module.on_load_checkpoint({"stage": "adversarial"})


def test_kd_requires_target_and_nonfinite_audio_fails(tmp_path):
    config = tiny_config(steps=1)
    config["loss"]["kd"] = 1.0
    module = EnhancementModule(config)
    with pytest.raises(ValueError, match="cached teacher"):
        trainer(tmp_path, 1).fit(module, train_dataloaders=DataLoader(TinyDataset(), batch_size=2))
    config["loss"]["kd"] = 0.0
    module = EnhancementModule(config)
    invalid = TinyDataset()[0]
    invalid["noisy"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="noisy input"):
        trainer(tmp_path, 1).fit(module, train_dataloaders=DataLoader([invalid], batch_size=1))


def test_all_future_decoder_parameters_require_grad_before_ddp_wrapping():
    module = EnhancementModule(tiny_config())
    assert all(parameter.requires_grad for parameter in module.model.parameters())
    identities = {id(parameter) for parameter in module.model.parameters()}
    module.model.activate_prepared_decoders()
    module._set_active_head_gradients()
    assert identities == {id(parameter) for parameter in module.model.parameters()}
    assert all(parameter.requires_grad for parameter in module.model.parameters())


def test_validation_excludes_distributed_padding_duplicates(tmp_path):
    module = EnhancementModule(tiny_config())
    rows = [TinyDataset()[index] for index in range(3)]
    for row in rows:
        row["distributed_valid"] = torch.tensor(True)
    rows[1]["distributed_valid"] = torch.tensor(False)
    rows[1]["clean"] = rows[1]["clean"] + 100  # Would dominate an incorrectly included duplicate.
    expected = []
    with torch.no_grad():
        for index in (0, 2):
            row = rows[index]
            predictions = module(row["noisy"].unsqueeze(0))
            expected.append(torch.stack([
                module.criterion(value, row["clean"].unsqueeze(0), row["lengths"].unsqueeze(0))[0]
                for value in predictions.values()]).mean())
    runner = trainer(tmp_path, 2)
    runner.validate(module, dataloaders=DataLoader(rows, batch_size=2), verbose=False)
    torch.testing.assert_close(runner.callback_metrics["validation_mean_clean_loss"].float(),
                               torch.stack(expected).mean())
    assert all(metric.weight.item() == 2 for metric in module.validation_metrics.values())


def test_zero_valid_validation_batch_executes_without_scoring_padding(tmp_path):
    module = EnhancementModule(tiny_config())
    row = TinyDataset()[0]
    row["distributed_valid"] = torch.tensor(False)
    runner = trainer(tmp_path, 2)
    runner.validate(module, dataloaders=DataLoader([row], batch_size=1), verbose=False)
    assert all(metric.weight.item() == 0 for metric in module.validation_metrics.values())
    assert all(metric.mean_value.item() == 0 for metric in module.validation_metrics.values())


def test_validation_uses_nonpersistent_float64_torchmetrics_without_checkpoint_keys():
    module = EnhancementModule(tiny_config())
    before = set(module.state_dict())
    assert all(isinstance(metric, MeanMetric) for metric in module.validation_metrics.values())
    for metric in module.validation_metrics.values():
        assert metric.mean_value.dtype == metric.weight.dtype == torch.float64
        metric.update(torch.tensor(1.0), weight=3)
        metric.update(torch.tensor(5.0), weight=1)
        assert metric.compute().item() == pytest.approx(2.0)
    assert set(module.state_dict()) == before
    assert not any(key.startswith("validation_metrics.") for key in before)
    fresh = EnhancementModule(tiny_config())
    fresh.load_state_dict(module.state_dict(), strict=True)
    assert all(metric.weight.item() == 0 for metric in fresh.validation_metrics.values())


def test_validation_metric_resets_between_epochs(tmp_path):
    module = EnhancementModule(tiny_config())
    runner = trainer(tmp_path, 2)
    loader = DataLoader(TinyDataset(), batch_size=2)
    runner.validate(module, dataloaders=loader, verbose=False)
    first = runner.callback_metrics["validation_mean_clean_loss"].clone()
    assert all(metric.weight.item() == 8 for metric in module.validation_metrics.values())
    runner.validate(module, dataloaders=loader, verbose=False)
    torch.testing.assert_close(runner.callback_metrics["validation_mean_clean_loss"], first)
    assert all(metric.weight.item() == 8 for metric in module.validation_metrics.values())


def test_native_resume_rejects_world_size_change():
    module = EnhancementModule(tiny_config())
    checkpoint = module.portable_checkpoint()
    checkpoint["world_size"] = 2
    with pytest.raises(ValueError, match="same world size"):
        module.on_load_checkpoint(checkpoint)


def test_cuda_rng_uses_strategy_device_before_transfer_and_after_teardown(monkeypatch):
    module = EnhancementModule(tiny_config())
    assigned_device = torch.device("cuda", 1)
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(root_device=assigned_device),
                                      world_size=1, global_rank=0)
    assert module.device.type == "cpu"
    saved_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    called = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: (called.append(("get", device)) or saved_state))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: called.append(("set", device)))
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    assert called == [("get", assigned_device), ("set", assigned_device)]


def test_local_rng_helpers_preserve_python_and_torch_progression():
    module = EnhancementModule(tiny_config())
    saved = module.capture_rng_state()
    expected_python = [random.randrange(1000) for _ in range(5)]
    expected_torch = torch.randn(7)
    module.restore_rng_state(saved)
    assert [random.randrange(1000) for _ in range(5)] == expected_python
    torch.testing.assert_close(torch.randn(7), expected_torch, rtol=0, atol=0)
