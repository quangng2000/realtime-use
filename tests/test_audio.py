import pytest
import torch

from realtime_use.audio import SpectralTransform, StreamingEnhancer
from realtime_use.config import ModelConfig
from realtime_use.model import Enhancer


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("length", [1, 17, 319, 320, 321, 639, 640, 641, 960, 1921])
def test_transform_roundtrip_and_gradient(length):
    transform = SpectralTransform(16000, 40, 20)
    audio = torch.randn(2, length, requires_grad=True)
    reconstructed = transform.synthesis(transform.analysis(audio), length)
    torch.testing.assert_close(audio, reconstructed, atol=3e-4, rtol=3e-4)
    reconstructed.square().mean().backward()
    assert torch.isfinite(audio.grad).all()


@pytest.mark.parametrize("length", [1, 15, 31, 32, 33, 48, 49, 157])
@pytest.mark.parametrize("exit_depth", [4, 8, 12])
def test_chunked_stream_matches_batch_and_exact_length(length, exit_depth):
    cfg = ModelConfig(sample_rate=800, depth=12, exits=(4, 8, 12), width=4,
                      state_dim=2, expand=1, conv_kernel=2)
    model = Enhancer(cfg).eval()
    audio = torch.randn(1, length)
    with torch.no_grad():
        expected = model.enhance(audio, (exit_depth,))[exit_depth]
    stream = StreamingEnhancer(model, exit_depth)
    outputs = []
    for chunk in audio.split(7, dim=-1):
        outputs.append(stream.push(chunk))
        assert stream._input.shape[-1] < stream.transform.window_samples
        assert stream._input.untyped_storage().nbytes() <= stream.transform.window_samples * 4
    outputs.append(stream.flush())
    actual = torch.cat(outputs, -1)
    assert actual.shape == audio.shape
    torch.testing.assert_close(actual, expected, atol=8e-4, rtol=2e-4)
    with pytest.raises(RuntimeError, match="reset"):
        stream.push(audio)
    stream.reset()
    reset_output = torch.cat([stream.push(audio), stream.flush()], -1)
    assert stream._input.untyped_storage().nbytes() <= stream.transform.window_samples * 4
    torch.testing.assert_close(actual, reset_output, atol=8e-4, rtol=2e-4)


def test_stream_does_not_emit_before_full_window():
    model = Enhancer(ModelConfig(sample_rate=800, depth=1, exits=(1,), width=4,
                                  expand=1, state_dim=2)).eval()
    stream = StreamingEnhancer(model, 1)
    assert stream.push(torch.zeros(1, 31)).shape[-1] == 0
    assert stream.push(torch.zeros(1, 1)).shape[-1] == 16
    assert stream.flush().shape[-1] == 16


def test_stream_checks_training_mode_and_batch_shape():
    model = Enhancer(ModelConfig(depth=1, exits=(1,), width=4, expand=1, state_dim=2))
    with pytest.raises(ValueError, match="eval"):
        StreamingEnhancer(model, 1)
    stream = StreamingEnhancer(model.eval(), 1)
    stream.push(torch.zeros(1, 10))
    with pytest.raises(ValueError, match="Batch"):
        stream.push(torch.zeros(2, 10))


def test_window_boundaries_do_not_amplify_mild_spectral_edits():
    torch.manual_seed(7)
    transform = SpectralTransform(16000, 40, 20)
    signal = torch.randn(1, 16000) * 0.05
    spectrum = transform.analysis(signal)
    mask = 1 + 0.1 * torch.cos(torch.arange(spectrum.shape[-1]) * torch.pi)
    restored = transform.synthesis(spectrum * mask, signal.shape[-1])
    assert restored.abs().max() < 1.2 * signal.abs().max()
