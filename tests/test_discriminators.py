"""Exercise the real DAC package without downloading pretrained weights."""

from importlib.util import find_spec

import pytest
import torch
from torch.nn import functional as F

from realtime_use.discriminators import (
    DACMultiResolutionDiscriminator,
    MultiScaleDiscriminator,
    build_discriminator,
)
from realtime_use.losses import discriminator_audio_loss, generator_audio_losses


requires_dac = pytest.mark.skipif(find_spec("dac") is None, reason="Install the gan extra to test official DAC")


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_missing_implementation_keeps_legacy_discriminator():
    discriminator = build_discriminator({"channels": 4, "scales": 2}, sample_rate=16000)
    assert isinstance(discriminator, MultiScaleDiscriminator)
    assert len(discriminator(torch.randn(1, 97))) == 2
    with pytest.raises(ValueError, match="implementation"):
        build_discriminator({"implementation": "unknown"}, sample_rate=16000)


@requires_dac
@pytest.mark.parametrize("sample_rate,expected", [
    (8000, [(0.0, 1.0)]),
    (16000, [(0.0, 0.5), (0.5, 1.0)]),
    (22050, [(0.0, 8000 / 22050), (8000 / 22050, 16000 / 22050), (16000 / 22050, 1.0)]),
])
def test_dac_factory_uses_official_mrd_and_four_khz_bands(sample_rate, expected):
    from dac.model.discriminator import MRD

    discriminator = build_discriminator({"implementation": "dac_mrd", "fft_sizes": [32, 64]}, sample_rate)
    assert isinstance(discriminator, DACMultiResolutionDiscriminator)
    assert discriminator.bands == expected
    assert all(isinstance(module, MRD) for module in discriminator.discriminators)
    assert [module.window_length for module in discriminator.discriminators] == [32, 64]


@requires_dac
@pytest.mark.parametrize("length", [1, 17, 97])
def test_adapter_preserves_direct_mrd_outputs_and_short_crop_gradients(length):
    torch.manual_seed(71)
    discriminator = DACMultiResolutionDiscriminator(16000, (32, 64))
    waveform = (0.01 * torch.randn(2, length)).requires_grad_()
    outputs = discriminator(waveform)
    for size, upstream, actual in zip(discriminator.fft_sizes, discriminator.discriminators, outputs):
        expected = upstream(F.pad(waveform, (0, max(0, size - length))).unsqueeze(1))
        assert len(actual) == 5 * len(discriminator.bands) + 1
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
            assert torch.isfinite(value).all()
    sum(resolution[-1].square().mean() for resolution in outputs).backward()
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all()
    assert waveform.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in discriminator.parameters())


@requires_dac
def test_dac_gan_losses_respect_valid_prefixes_and_detach_the_right_graph():
    torch.manual_seed(19)
    discriminator = DACMultiResolutionDiscriminator(16000, (32, 64))
    reference = torch.randn(2, 97) * 0.1
    prediction = (reference + torch.randn_like(reference) * 0.05).requires_grad_()
    lengths = torch.tensor([17, 97])
    loss_d = discriminator_audio_loss(discriminator, reference, prediction, lengths)
    assert torch.isfinite(loss_d)
    loss_d.backward()
    assert prediction.grad is None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in discriminator.parameters())

    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    adversarial, matching = generator_audio_losses(discriminator, reference, prediction, lengths)
    assert torch.isfinite(adversarial) and torch.isfinite(matching)
    assert matching > 0
    (adversarial + matching).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, :17].abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[0, 17:]) == 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())

    altered_reference, altered_prediction = reference.clone(), prediction.detach().clone()
    altered_reference[0, 17:] = 1000
    altered_prediction[0, 17:] = -1000
    actual = generator_audio_losses(discriminator, altered_reference, altered_prediction, lengths)
    for unchanged, changed in zip((adversarial, matching), actual):
        torch.testing.assert_close(unchanged, changed, rtol=0, atol=0)


@requires_dac
def test_silent_one_sample_dac_input_and_losses_are_finite():
    discriminator = DACMultiResolutionDiscriminator(16000, (32, 64))
    silent = torch.zeros(1, 1, requires_grad=True)
    adversarial, matching = generator_audio_losses(discriminator, silent.detach(), silent)
    critic = discriminator_audio_loss(discriminator, silent.detach(), silent)
    assert all(torch.isfinite(value) for value in (adversarial, matching, critic))
    assert matching == 0
    (adversarial + matching).backward()
    assert silent.grad is not None and torch.isfinite(silent.grad).all()


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_rate": True}, "sample_rate"),
    ({"sample_rate": 0}, "sample_rate"),
    ({"sample_rate": 16000, "band_width_hz": 0}, "band_width_hz"),
    ({"sample_rate": 16000, "fft_sizes": []}, "fft_sizes"),
    ({"sample_rate": 16000, "fft_sizes": [32.5]}, "fft_sizes"),
    ({"sample_rate": 48000, "fft_sizes": [8]}, "empty frequency band"),
])
def test_invalid_dac_config_fails_before_loading_optional_dependency(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DACMultiResolutionDiscriminator(**kwargs)
