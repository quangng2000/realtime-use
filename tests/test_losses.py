"""Library objective integration, valid-length behavior, and legacy selection."""

import pytest
import torch
from auraloss.freq import MultiResolutionSTFTLoss as AuralossSTFTLoss
from torch.nn import functional as F

from realtime_use.losses import MultiResolutionSTFTLoss, ReconstructionLoss


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def library_loss():
    return AuralossSTFTLoss(
        fft_sizes=[16, 32], hop_sizes=[4, 8], win_lengths=[16, 32],
        window="hann_window", w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0,
        w_phs=0.0, sample_rate=None, scale=None, n_bins=None,
        perceptual_weighting=False, scale_invariance=False, eps=1e-8,
        output="loss", reduction="mean", mag_distance="L1")


@pytest.mark.parametrize("samples", [1, 9, 17, 65])
def test_library_objective_and_gradients_equal_direct_auraloss_calls(samples):
    generator = torch.Generator().manual_seed(samples)
    prediction = torch.randn(1, samples, generator=generator, requires_grad=True)
    target = torch.randn(1, samples, generator=generator)
    criterion = ReconstructionLoss(waveform=0.7, stft=1.3, fft_sizes=(16, 32),
                                   implementation="auraloss")
    assert isinstance(criterion.waveform, torch.nn.L1Loss)
    assert isinstance(criterion.spectral, AuralossSTFTLoss)
    actual, components = criterion(prediction, target)
    padding = max(0, 17 - samples)
    direct_spectral = library_loss()(F.pad(prediction[:, None], (0, padding)),
                                     F.pad(target[:, None], (0, padding)))
    direct_waveform = torch.nn.L1Loss()(prediction, target)
    expected = 0.7 * direct_waveform + 1.3 * direct_spectral
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(components["stft_loss"], direct_spectral.detach())
    torch.testing.assert_close(components["waveform_l1"], direct_waveform.detach())
    actual_gradient, = torch.autograd.grad(actual, prediction, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, prediction)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all()
    assert actual_gradient.abs().sum() > 0


def test_library_loss_ignores_padding_and_weights_utterances_independently():
    generator = torch.Generator().manual_seed(91)
    lengths = torch.tensor([7, 41, 41])
    prediction = torch.randn(3, 60, generator=generator, requires_grad=True)
    target = torch.randn(3, 60, generator=generator) * torch.tensor([0.01, 0.1, 10.0])[:, None]
    criterion = ReconstructionLoss(fft_sizes=(16, 32), implementation="auraloss")
    actual, _ = criterion(prediction, target, lengths)
    separate = torch.stack([
        criterion(prediction[index:index + 1, :length], target[index:index + 1, :length])[0]
        for index, length in enumerate(lengths)
    ]).mean()
    torch.testing.assert_close(actual, separate)
    gradient, = torch.autograd.grad(actual, prediction)
    padded_prediction = torch.cat([prediction.detach(), 1000 * torch.randn(3, 23)], -1).requires_grad_()
    padded_target = torch.cat([target, 1000 * torch.randn(3, 23)], -1)
    padded_loss, _ = criterion(padded_prediction, padded_target, lengths)
    padded_loss.backward()
    torch.testing.assert_close(padded_loss, actual)
    torch.testing.assert_close(padded_prediction.grad[:, :60], gradient)
    for index, length in enumerate(lengths):
        assert torch.count_nonzero(padded_prediction.grad[index, length:]) == 0
    # Equal-length grouping must not introduce the library's cross-example
    # spectral-convergence normalization: singleton calls define this recipe.
    batched, _ = criterion(prediction[1:, :41], target[1:, :41])
    independently = torch.stack([criterion(prediction[i:i + 1, :41], target[i:i + 1, :41])[0]
                                 for i in (1, 2)]).mean()
    torch.testing.assert_close(batched, independently)


def test_library_objective_silence_and_cpu_autocast_are_finite():
    criterion = ReconstructionLoss(fft_sizes=(16, 32), implementation="auraloss")
    target = torch.zeros(2, 13)
    identity, _ = criterion(target, target)
    assert identity.item() == 0
    prediction = torch.randn(2, 13, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, _ = criterion(prediction, target)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_missing_implementation_preserves_legacy_objective_and_checkpoint_keys():
    criterion = ReconstructionLoss(fft_sizes=(16, 32))
    assert criterion.implementation == "legacy"
    assert isinstance(criterion.spectral, MultiResolutionSTFTLoss)
    assert not criterion.state_dict()
    assert not ReconstructionLoss(fft_sizes=(16, 32), implementation="auraloss").state_dict()
    with pytest.raises(ValueError, match="loss.implementation"):
        ReconstructionLoss(implementation="unknown")
