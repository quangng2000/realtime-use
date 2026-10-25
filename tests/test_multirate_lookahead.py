"""Forward/backward and inference contracts only; no optimizer or training loop."""

from copy import deepcopy

import pytest
import torch

from realtime_use.audio import SpectralTransform, StreamingEnhancer
from realtime_use.author_codec import AuthorEncoder
from realtime_use.benchmark import benchmark_stream
from realtime_use.config import ModelConfig
from realtime_use.export import export_exit
from realtime_use.infer import enhance_waveform, load_model
from realtime_use.model import Enhancer


RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(97)
    yield
    torch.set_num_threads(previous)


def configuration(**overrides):
    values = dict(sample_rate=800, sample_rates=(800, 1600), depth=1, exits=(1,), width=4,
                  state_dim=2, expand=1, conv_kernel=2, codec_impl="nemotron_causal",
                  mixer_impl="legacy", scan_backend="reference", lookahead_frames=(0, 1, 2), window_type="hann")
    values.update(overrides)
    return ModelConfig(**values)


def model(**overrides):
    result = Enhancer(configuration(**overrides)).eval()
    with torch.no_grad():
        for decoder in [result.shared_decoder, *result.decoders.values()]:
            if decoder is not None:
                decoder.magnitude.final_conv.bias.fill_(0.8)
    return result


@pytest.mark.parametrize("lookahead", [0, 1, 2])
def test_encoder_branches_have_separate_parameters_and_exact_lookahead_boundary(lookahead):
    encoder = AuthorEncoder(4, (0, 1, 2)).double()
    branches = encoder.codec.dense_conv_1.branches
    assert len({branches[str(ahead)][0].conv.weight.data_ptr() for ahead in (0, 1, 2)}) == 3
    spectrum = torch.randn(1, 8, 6, dtype=torch.complex128, requires_grad=True)
    actual = encoder(spectrum, lookahead)
    changed = spectrum.detach().clone()
    changed[:, 3 + lookahead:] += 100 * torch.randn_like(changed[:, 3 + lookahead:])
    torch.testing.assert_close(actual[:, :3], encoder(changed, lookahead)[:, :3], rtol=0, atol=0)
    actual[:, 2].square().sum().backward()
    assert torch.count_nonzero(spectrum.grad[:, 2 + lookahead]) > 0
    assert torch.count_nonzero(spectrum.grad[:, 3 + lookahead:]) == 0
    for ahead in (0, 1, 2):
        gradient = branches[str(ahead)][0].conv.weight.grad
        assert (gradient is not None) == (ahead == lookahead)


@pytest.mark.parametrize("lookahead", [0, 1, 2])
def test_encoder_streaming_delay_flush_and_gradients_match_full_sequence(lookahead):
    encoder = AuthorEncoder(4, (0, 1, 2)).double()
    recurrent = deepcopy(encoder)
    left = torch.randn(1, 5, 6, dtype=torch.complex128, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected = encoder(left, lookahead)
    state, outputs = None, []
    frames = [*right.unbind(1), *[torch.zeros_like(right[:, 0]) for _ in range(lookahead)]]
    for index, frame in enumerate(frames):
        output, state = recurrent.step(frame, state, lookahead)
        assert (output is None) == (index < lookahead)
        if output is not None:
            outputs.append(output)
    actual = torch.stack(outputs, 1)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=2e-11, rtol=2e-10)
    for parameter, counterpart in zip(encoder.parameters(), recurrent.parameters()):
        assert (parameter.grad is None) == (counterpart.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, counterpart.grad, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("lookahead", [0, 1, 2])
@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 65, 149])
def test_audio_startup_short_clips_tails_and_reset(lookahead, length):
    enhancer = model()
    waveform = 0.1 * torch.randn(1, length)
    expected = enhance_waveform(enhancer, waveform, lookahead=lookahead)
    stream = StreamingEnhancer(enhancer, 1, lookahead=lookahead)
    chunks = []
    received = 0
    for chunk in waveform.split(7, -1):
        output = stream.push(chunk)
        received += chunk.shape[-1]
        if received < (2 + lookahead) * stream.transform.hop_samples:
            assert output.shape[-1] == 0
        chunks.append(output)
        assert stream._input.shape[-1] < stream.transform.window_samples
        assert stream._input.untyped_storage().nbytes() <= 4 * stream.transform.window_samples
    chunks.append(stream.flush())
    actual = torch.cat(chunks, -1)
    assert actual.shape == waveform.shape
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    assert stream._frames == stream.transform.frame_count(length)
    assert stream._output_frames == stream._frames
    stream.reset()
    torch.testing.assert_close(torch.cat((stream.push(waveform), stream.flush()), -1), expected,
                               atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("lookahead", [0, 1, 2])
@pytest.mark.parametrize("sample_rate", RATES)
def test_all_native_rates_use_exact_40ms_20ms_hann_and_match_streaming(sample_rate, lookahead):
    enhancer = model(sample_rate=16000, sample_rates=RATES, width=2, state_dim=1)
    transform = enhancer.transform_for(sample_rate)
    assert transform.window_samples == round(sample_rate * 0.040)
    assert transform.hop_samples == round(sample_rate * 0.020)
    torch.testing.assert_close(transform.window, torch.hann_window(transform.window_samples), rtol=0, atol=0)
    waveform = 0.1 * torch.randn(1, (3 + lookahead) * transform.hop_samples + 13)
    expected = enhance_waveform(enhancer, waveform, sample_rate=sample_rate, lookahead=lookahead)
    stream = StreamingEnhancer(enhancer, 1, sample_rate=sample_rate, lookahead=lookahead)
    outputs = []
    first_output_at = None
    received = 0
    for chunk in waveform.split(transform.hop_samples, -1):
        part = stream.push(chunk)
        received += chunk.shape[-1]
        if part.shape[-1] and first_output_at is None:
            first_output_at = received
        outputs.append(part)
    assert first_output_at == transform.window_samples + lookahead * transform.hop_samples
    outputs.append(stream.flush())
    torch.testing.assert_close(torch.cat(outputs, -1), expected, atol=3e-5, rtol=3e-4)
    assert enhancer.config.sample_rate == 16000
    assert enhancer.transform.window_samples == 640


@pytest.mark.parametrize("sample_rate", RATES)
def test_hann_transform_roundtrip_and_gradients_include_short_audio(sample_rate):
    transform = SpectralTransform(sample_rate, 40, 20, "hann")
    for length in (1, transform.hop_samples + 1):
        audio = torch.randn(1, length, requires_grad=True)
        recovered = transform.synthesis(transform.analysis(audio), length)
        torch.testing.assert_close(recovered, audio, atol=1e-6, rtol=1e-5)
        recovered.square().mean().backward()
        assert torch.isfinite(audio.grad).all()


def test_two_native_rate_lookahead_streams_can_be_interleaved():
    enhancer = model()
    streams = [StreamingEnhancer(enhancer, 1, 800, 0), StreamingEnhancer(enhancer, 1, 1600, 2)]
    waves = [torch.randn(1, 171) * 0.05, torch.randn(1, 263) * 0.05]
    pieces = [[], []]
    for offset in range(0, max(w.shape[-1] for w in waves), 11):
        for index, (stream, wave) in enumerate(zip(streams, waves)):
            if offset < wave.shape[-1]:
                pieces[index].append(stream.push(wave[:, offset:offset + 11]))
    for index, (stream, wave) in enumerate(zip(streams, waves)):
        pieces[index].append(stream.flush())
        expected = enhance_waveform(enhancer, wave, sample_rate=stream.sample_rate, lookahead=stream.lookahead)
        torch.testing.assert_close(torch.cat(pieces[index], -1), expected, atol=3e-5, rtol=3e-4)


def test_old_checkpoint_missing_new_fields_preserves_sqrt_hann_and_parameter_layout(tmp_path):
    old = model(sample_rates=(), lookahead_frames=(0,), window_type="sqrt_hann")
    configuration = old.config.to_dict()
    for key in ("sample_rates", "lookahead_frames", "window_type"):
        configuration.pop(key)
    checkpoint = tmp_path / "old.pt"
    torch.save({"model_config": configuration, "model_state": old.state_dict()}, checkpoint)
    restored = load_model(checkpoint)
    assert restored.config.supported_sample_rates == (800,)
    assert restored.config.lookahead_frames == (0,)
    assert restored.config.window_type == "sqrt_hann"
    assert set(old.state_dict()) == set(restored.state_dict())
    torch.testing.assert_close(restored.transform.window, torch.hann_window(32).sqrt(), rtol=0, atol=0)
    waveform = torch.randn(1, 73)
    torch.testing.assert_close(enhance_waveform(restored, waveform), enhance_waveform(old, waveform), rtol=0, atol=0)


@pytest.mark.parametrize("lookahead,retain_all_rates", [(0, True), (1, True), (2, False)])
def test_export_prunes_lookahead_and_exit_and_keeps_selected_native_rates(tmp_path, lookahead, retain_all_rates):
    source = model(depth=2, exits=(1, 2), decoder_mode="per_exit")
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_config": source.config.to_dict(), "model_state": source.state_dict()}, checkpoint)
    path = export_exit(checkpoint, 1, tmp_path / "selected.pt", lookahead=lookahead,
                       sample_rate=1600, retain_all_rates=retain_all_rates)
    exported = load_model(path)
    assert exported.config.lookahead_frames == (lookahead,)
    assert exported.config.sample_rate == 1600
    assert exported.config.supported_sample_rates == ((800, 1600) if retain_all_rates else (1600,))
    assert len(exported.blocks) == 1
    assert not any("branches." in key for key in exported.state_dict())
    assert sum(parameter.numel() for parameter in exported.parameters()) == source.active_parameter_count(1, lookahead)
    assert source.active_parameter_count(1, lookahead) < sum(parameter.numel() for parameter in source.parameters())
    payload = torch.load(path, weights_only=True)
    assert payload["algorithmic_latency_ms"] == 40 + 20 * lookahead
    assert payload["verification"]["sample_rates_checked"] == list(exported.config.supported_sample_rates)
    for rate in exported.config.supported_sample_rates:
        wave = torch.randn(1, 137) * 0.05
        expected = enhance_waveform(source, wave, 1, sample_rate=rate, lookahead=lookahead)
        actual = enhance_waveform(exported, wave, 1, streaming=True, chunk_samples=11,
                                  sample_rate=rate, lookahead=lookahead)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("lookahead", [0, 1, 2])
def test_benchmark_reports_actual_startup_and_algorithmic_latency(lookahead):
    enhancer = model()
    report = benchmark_stream(enhancer, torch.randn(256) * 0.05, warmup_seconds=0,
                              sample_rate=1600, lookahead=lookahead)
    assert report["sample_rate"] == 1600
    assert report["algorithmic_latency_ms"] == 40 + 20 * lookahead
    assert report["input_ms_before_first_output"] == 40 + 20 * lookahead
    assert report["startup_hops_excluded_from_steady_state"] == 2 + lookahead


def test_unsupported_rates_and_lookahead_fail_without_state_reuse():
    enhancer = model()
    with pytest.raises(ValueError, match="sample_rate"):
        enhancer.enhance(torch.ones(1, 20), sample_rate=22050)
    with pytest.raises(ValueError, match="lookahead"):
        StreamingEnhancer(enhancer, 1, lookahead=3)
    frame = torch.randn(1, 17, dtype=torch.complex64)
    _, state = enhancer.step(frame, lookahead=1)
    with pytest.raises(ValueError, match="reset state"):
        enhancer.step(frame, state, lookahead=2)
    for settings in ({"sample_rates": (1600,)}, {"sample_rates": (800, 800)},
                     {"lookahead_frames": (0, 3)}, {"lookahead_frames": ()}, {"window_type": "unknown"}):
        with pytest.raises(ValueError):
            configuration(**settings)
