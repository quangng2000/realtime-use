"""Numerical training objectives independent of the Lightning orchestration."""

import pytest
import torch
from realtime_use.discriminators import MultiScaleDiscriminator
from realtime_use.losses import (ReconstructionLoss, discriminator_audio_loss,
    discriminator_hinge_loss, feature_matching_loss, generator_audio_losses,
    generator_hinge_loss)

@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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
