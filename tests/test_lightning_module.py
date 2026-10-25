"""Lightning lifecycle tests on small CPU models, without quality claims."""

import copy

from lightning.pytorch import Trainer
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

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
