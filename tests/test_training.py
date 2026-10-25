"""Exercise optimizer transitions, loss gradients, and real checkpoint continuation."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from realtime_use.config import ModelConfig
from realtime_use.discriminators import MultiScaleDiscriminator
from realtime_use.losses import (ReconstructionLoss, discriminator_audio_loss,
                                 discriminator_hinge_loss, feature_matching_loss,
                                 generator_audio_losses, generator_hinge_loss)
from realtime_use.model import Enhancer
from realtime_use.train import (build_optimizer, make_parser, run_training,
                                transition_decoders)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model_config(**overrides):
    values = dict(sample_rate=800, window_ms=40.0, hop_ms=20.0,
                  depth=2, exits=(1, 2), width=4, state_dim=2,
                  expand=1, conv_kernel=2, scan_backend="reference")
    values.update(overrides)
    return ModelConfig(**values)


def test_reconstruction_loss_is_finite_and_backpropagates_for_short_audio():
    prediction = torch.randn(2, 19, requires_grad=True)
    target = torch.randn(2, 19)
    criterion = ReconstructionLoss(fft_sizes=(16, 32))
    loss, components = criterion(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert set(components) == {"waveform_l1", "stft_loss"}
    identity, _ = criterion(target, target)
    assert identity.item() == pytest.approx(0)


def test_adversarial_losses_propagate_to_generator_when_discriminator_frozen():
    discriminator = MultiScaleDiscriminator(channels=4, scales=2)
    clean = torch.randn(2, 127)
    prediction = torch.randn(2, 127, requires_grad=True)
    d_loss = discriminator_hinge_loss(discriminator(clean), discriminator(prediction.detach()))
    d_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in discriminator.parameters())
    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    real, fake = discriminator(clean), discriminator(prediction)
    loss = generator_hinge_loss(fake) + feature_matching_loss(real, fake)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert all(p.grad is None for p in discriminator.parameters())


def test_reconstruction_and_distillation_ignore_padding_and_average_per_utterance():
    lengths = torch.tensor([19, 31, 19])
    predicted = torch.randn(3, 47, requires_grad=True)
    reference = torch.randn(3, 47)
    criterion = ReconstructionLoss(fft_sizes=(16, 32))
    loss, _ = criterion(predicted, reference, lengths=lengths)
    expected = torch.stack([criterion(predicted[i:i+1, :length], reference[i:i+1, :length])[0]
                            for i, length in enumerate(lengths)]).mean()
    torch.testing.assert_close(loss, expected)
    appended_prediction = torch.cat([predicted, torch.randn(3, 13)], dim=-1)
    appended_reference = torch.cat([reference, torch.randn(3, 13)], dim=-1)
    padded_loss, _ = criterion(appended_prediction, appended_reference, lengths=lengths)
    torch.testing.assert_close(padded_loss, loss)
    padded_loss.backward()
    for index, length in enumerate(lengths):
        assert torch.count_nonzero(predicted.grad[index, length:]) == 0
    with pytest.raises(ValueError, match="positive"):
        criterion(predicted, reference, lengths=torch.tensor([0, 31, 19]))


def test_adversarial_objectives_ignore_padding_and_match_per_utterance_losses():
    discriminator = MultiScaleDiscriminator(channels=4, scales=2)
    prediction = torch.randn(3, 70, requires_grad=True)
    reference = torch.randn(3, 70)
    lengths = torch.tensor([39, 53, 39])
    loss = discriminator_audio_loss(discriminator, reference, prediction, lengths)
    separate = torch.stack([discriminator_audio_loss(discriminator, reference[i:i+1, :length],
                                                     prediction[i:i+1, :length])
                            for i, length in enumerate(lengths)]).mean()
    torch.testing.assert_close(loss, separate)
    padded_prediction = torch.cat([prediction, torch.randn(3, 23)], dim=-1)
    padded_reference = torch.cat([reference, torch.randn(3, 23)], dim=-1)
    torch.testing.assert_close(discriminator_audio_loss(discriminator, padded_reference,
                                                        padded_prediction, lengths), loss)
    discriminator.requires_grad_(False)
    adversarial, matching = generator_audio_losses(discriminator, reference, prediction, lengths)
    padded_adversarial, padded_matching = generator_audio_losses(
        discriminator, padded_reference, padded_prediction, lengths)
    torch.testing.assert_close(padded_adversarial, adversarial)
    torch.testing.assert_close(padded_matching, matching)
    (padded_adversarial + padded_matching).backward()
    for index, length in enumerate(lengths):
        assert torch.count_nonzero(prediction.grad[index, length:]) == 0


def test_split_decoders_preserves_outputs_and_backbone_optimizer_moments():
    model = Enhancer(tiny_model_config())
    training = {"lr": 2e-4, "predecoder_lr_scale": 0.1}
    optimizer = build_optimizer(model, training)
    spectrum = torch.randn(1, 3, 17, dtype=torch.complex64)
    sum(value.abs().mean() for value in model(spectrum).values()).backward()
    optimizer.step()
    before = {depth: value.detach().clone() for depth, value in model(spectrum).items()}
    backbone_parameter = next(model.encoder.parameters())
    moments = optimizer.state[backbone_parameter]["exp_avg"].clone()
    optimizer = transition_decoders(model, optimizer, training)
    assert model.config.decoder_mode == "per_exit"
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([2e-5, 2e-4])
    torch.testing.assert_close(optimizer.state[backbone_parameter]["exp_avg"], moments)
    for depth, value in model(spectrum).items():
        torch.testing.assert_close(value, before[depth])
    assert model.decoders["1"].output.weight is not model.decoders["2"].output.weight


def make_training_files(tmp_path: Path, kd: float = 0.0, teacher: bool = False):
    cfg = {"model": tiny_model_config(depth=1, exits=(1,)).to_dict(),
           "training": {"batch_size": 1, "crop_seconds": 0.16, "steps": 1,
                        "precision": "fp32", "lr": 1e-3, "checkpoint_every": 1,
                        "log_every": 1, "validation_batches": 1, "grad_accum": 2},
           "loss": {"waveform": 1.0, "stft": 0.1, "fft_sizes": [16, 32], "kd": kd},
           "adversarial": {"channels": 4, "scales": 1, "weight": 0.01,
                           "feature_matching": 0.1}}
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    x = np.arange(128, dtype=np.float32) / 800
    clean = 0.2 * np.sin(2 * np.pi * 120 * x)
    noisy = clean + 0.02 * np.random.default_rng(0).normal(size=128).astype(np.float32)
    sf.write(tmp_path / "clean.wav", clean, 800, subtype="FLOAT")
    sf.write(tmp_path / "noisy.wav", noisy, 800, subtype="FLOAT")
    row = {"noisy": "noisy.wav", "clean": "clean.wav", "sample_rate": 800}
    if teacher:
        row["teacher"] = "clean.wav"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return config, manifest


def test_training_checkpoint_kd_validation_and_teacher_free_adversarial_resume(tmp_path):
    config, manifest = make_training_files(tmp_path, kd=0.5, teacher=True)
    parser = make_parser()
    regression = run_training(parser.parse_args([
        "--config", str(config), "--manifest", str(manifest),
        "--validation-manifest", str(manifest), "--output", str(tmp_path / "regression"),
    ]))
    initial = torch.load(regression, weights_only=True)
    assert initial["stage"] == "regression" and initial["step"] == 1
    assert (regression.parent / "best.pt").is_file()
    metrics = json.loads((regression.parent / "metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["kd_loss"] >= 0
    assert "validation_mean_clean_loss" in metrics
    # Deliberately leave loss.kd enabled but remove cached targets: stage two must ignore KD.
    row = json.loads(manifest.read_text())
    row.pop("teacher")
    manifest.write_text(json.dumps(row) + "\n")
    adversarial = run_training(parser.parse_args([
        "--config", str(config), "--manifest", str(manifest),
        "--output", str(tmp_path / "adversarial"), "--stage", "adversarial",
        "--init-from", str(regression),
    ]))
    state = torch.load(adversarial, weights_only=True)
    assert state["stage"] == "adversarial" and state["step"] == 1
    assert state["discriminator_optimizer_state"]["state"]
    assert any(not torch.equal(value, initial["model_state"][key])
               for key, value in state["model_state"].items())
    resumed = run_training(parser.parse_args([
        "--config", str(config), "--manifest", str(manifest),
        "--output", str(tmp_path / "continued"), "--stage", "adversarial",
        "--resume", str(adversarial), "--steps", "2",
    ]))
    assert torch.load(resumed, weights_only=True)["step"] == 2


def test_kd_requires_cached_targets_and_stage_two_requires_initialization(tmp_path):
    config, manifest = make_training_files(tmp_path, kd=1)
    parser = make_parser()
    base = ["--config", str(config), "--manifest", str(manifest), "--output", str(tmp_path / "failed")]
    with pytest.raises(ValueError, match="cached teacher"):
        run_training(parser.parse_args(base))
    with pytest.raises(ValueError, match="requires --init-from"):
        run_training(parser.parse_args(base + ["--stage", "adversarial"]))


def test_resume_rejects_stage_change(tmp_path):
    config, manifest = make_training_files(tmp_path)
    parser = make_parser()
    checkpoint = run_training(parser.parse_args([
        "--config", str(config), "--manifest", str(manifest), "--output", str(tmp_path / "first")]))
    with pytest.raises(ValueError, match="preserve stage"):
        run_training(parser.parse_args([
            "--config", str(config), "--manifest", str(manifest), "--output", str(tmp_path / "bad"),
            "--resume", str(checkpoint), "--stage", "adversarial", "--steps", "2"]))


def test_resume_after_decoder_transition_restores_per_exit_architecture(tmp_path):
    config, manifest = make_training_files(tmp_path)
    settings = yaml.safe_load(config.read_text())
    settings["model"] = tiny_model_config().to_dict()
    settings["training"].update(steps=2, split_decoders_at=1, grad_accum=1)
    config.write_text(yaml.safe_dump(settings))
    parser = make_parser()
    checkpoint = run_training(parser.parse_args([
        "--config", str(config), "--manifest", str(manifest), "--output", str(tmp_path / "split")]))
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["model_config"]["decoder_mode"] == "per_exit"
    assert saved["training_config"]["model"]["decoder_mode"] == "per_exit"
    assert len(saved["optimizer_state"]["param_groups"]) == 2
    resumed = run_training(parser.parse_args([
        "--config", str(config), "--manifest", str(manifest), "--output", str(tmp_path / "split_resume"),
        "--resume", str(checkpoint), "--steps", "3"]))
    saved = torch.load(resumed, weights_only=True)
    assert saved["step"] == 3
    assert saved["model_config"]["decoder_mode"] == "per_exit"
