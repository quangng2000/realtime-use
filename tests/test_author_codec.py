"""Author-source reuse and causal adaptation, tested without Mamba/CUDA."""

from copy import deepcopy
import hashlib
import io
from pathlib import Path

import pytest
import torch
from torch import nn

from realtime_use.author_codec import AuthorDecoder, AuthorEncoder, CausalConv, ChannelLayerNorm
from realtime_use._vendor.nemotron_codec import codec_module_time_d4 as author


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(previous)


def codec(width=4, *, dtype=torch.float64):
    encoder, decoder = AuthorEncoder(width).to(dtype), AuthorDecoder(width).to(dtype)
    # Exercise both heads' gradients, avoiding the source ReLU's dead negative
    # region in a randomly initialized, tiny test network.
    with torch.no_grad():
        decoder.magnitude.final_conv.bias.fill_(1.0)
    return encoder, decoder


def stream(encoder, decoder, spectrum, encoder_state=None, decoder_state=None):
    encoded, outputs = [], []
    for frame in spectrum.unbind(1):
        features, encoder_state = encoder.step(frame, encoder_state)
        output, decoder_state = decoder.step(features, frame, decoder_state)
        encoded.append(features)
        outputs.append(output)
    return torch.stack(encoded, 1), torch.stack(outputs, 1), encoder_state, decoder_state


def test_vendored_source_and_license_are_exact_pinned_copies():
    root = Path(author.__file__).parent
    expected = {
        "codec_module_time_d4.py": "02bd1d478f36a9b38dce7d44368805298f42798ea239304580f1ccffafe766a0",
        "LICENSE": "a3d882953f026d3e2fdd57842d9c522c8320d4c0388af2278bcf4d438d255a2c",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert "Third-Party Notices" in (root / "LICENSE").read_text()


def test_actual_author_classes_and_forward_methods_are_reused():
    encoder, decoder = codec()
    assert type(encoder.codec) is author.DenseEncoder
    assert type(encoder.codec.dense_block) is author.DenseBlock
    assert type(decoder.magnitude) is author.MagDecoder
    assert type(decoder.phase) is author.PhaseDecoder
    assert encoder.codec.forward.__func__ is author.DenseEncoder.forward
    assert decoder.magnitude.forward.__func__ is author.MagDecoder.forward
    assert decoder.phase.forward.__func__ is author.PhaseDecoder.forward
    assert encoder.codec.dense_conv_2[0].stride == (1, 2)
    assert isinstance(encoder.codec.dense_conv_1[0], CausalConv)
    assert encoder.codec.dense_conv_1[0].conv.kernel_size == (3, 3)
    for head in (decoder.magnitude, decoder.phase):
        assert type(head.up_conv1[0]) is author.SPConvTranspose2d
        assert type(head.up_conv2[0]) is author.SPConvTranspose2d
        assert head.up_conv1[0].r == 2 and head.up_conv2[0].r == 1
        assert isinstance(head.up_conv2[0].conv, CausalConv)
        assert head.up_conv2[0].conv.time_axis == 3
    assert not any(isinstance(m, nn.InstanceNorm2d) for root in (encoder, decoder) for m in root.modules())


@pytest.mark.parametrize("bins", [1, 2, 5, 6, 321, 442])
def test_sequence_and_one_frame_streaming_match_for_odd_and_even_bin_counts(bins):
    encoder, decoder = codec()
    spectrum = torch.randn(2, 7, bins, dtype=torch.complex128)
    with torch.no_grad():
        encoded = encoder(spectrum)
        full = decoder(encoded, spectrum)
        streamed_features, streamed, _, _ = stream(encoder, decoder, spectrum)
    assert encoded.shape == (2, 7, (bins + 1) // 2, 4)
    assert full.shape == spectrum.shape
    torch.testing.assert_close(encoded, streamed_features, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(full, streamed, atol=2e-10, rtol=2e-10)


def test_future_frames_cannot_affect_encoder_or_decoder_prefix():
    encoder, decoder = codec()
    spectrum = torch.randn(2, 24, 8, dtype=torch.complex128)
    changed = spectrum.clone()
    changed[:, 11:] = torch.randn_like(changed[:, 11:]) * 100
    original_features, changed_features = encoder(spectrum), encoder(changed)
    torch.testing.assert_close(original_features[:, :11], changed_features[:, :11], rtol=0, atol=0)
    original, perturbed = decoder(original_features, spectrum), decoder(changed_features, changed)
    torch.testing.assert_close(original[:, :11], perturbed[:, :11], rtol=0, atol=0)


def test_channel_norm_uses_neither_time_nor_frequency_statistics():
    norm = ChannelLayerNorm(4).double()
    x = torch.randn(2, 4, 3, 5, dtype=torch.float64)
    changed = x.clone()
    changed[:, :, 1:] *= 100
    changed[:, :, 0, 1:] -= 200
    torch.testing.assert_close(norm(x)[:, :, 0, 0], norm(changed)[:, :, 0, 0], rtol=0, atol=0)


def test_full_and_streaming_gradients_match_and_remain_finite():
    encoder, decoder = codec()
    stream_encoder, stream_decoder = deepcopy(encoder), deepcopy(decoder)
    left = torch.randn(2, 19, 7, dtype=torch.complex128, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    target = torch.randn_like(left)
    full = decoder(encoder(left), left)
    _, streamed, _, _ = stream(stream_encoder, stream_decoder, right)
    for output in (full, streamed):
        (output - target).abs().square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=2e-9, rtol=2e-8)
    assert torch.isfinite(left.grad).all() and torch.count_nonzero(left.grad) > 0
    for original, recurrent in ((encoder, stream_encoder), (decoder, stream_decoder)):
        for (name, parameter), (_, counterpart) in zip(original.named_parameters(), recurrent.named_parameters()):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, counterpart.grad, atol=2e-9, rtol=2e-8, msg=name)


def test_external_states_are_isolated_bounded_serializable_and_not_mutated():
    encoder, decoder = codec()
    first = torch.randn(1, 40, 6, dtype=torch.complex128)
    second = torch.randn_like(first)
    encoder_states, decoder_states, outputs = [None, None], [None, None], [[], []]
    shapes = None
    with torch.no_grad():
        expected = [decoder(encoder(x), x) for x in (first, second)]
        for time in range(first.shape[1]):
            for index, audio in enumerate((first, second)):
                old_encoder = deepcopy(encoder_states[index])
                old_decoder = deepcopy(decoder_states[index])
                features, new_encoder = encoder.step(audio[:, time], encoder_states[index])
                output, new_decoder = decoder.step(features, audio[:, time], decoder_states[index])
                for old, state in ((old_encoder, encoder_states[index]), (old_decoder, decoder_states[index])):
                    if old is not None:
                        for key, values in old["convolutions"].items():
                            torch.testing.assert_close(values, state["convolutions"][key], rtol=0, atol=0)
                encoder_states[index], decoder_states[index] = new_encoder, new_decoder
                outputs[index].append(output)
                current_shapes = [tuple(t.shape) for state in (new_encoder, new_decoder)
                                  for t in state["convolutions"].values()]
                if shapes is None:
                    shapes = current_shapes
                assert current_shapes == shapes
            if time == 12:
                buffer = io.BytesIO()
                torch.save((encoder_states, decoder_states), buffer)
                buffer.seek(0)
                encoder_states, decoder_states = torch.load(buffer, weights_only=True)
    assert not list(encoder.named_buffers()) and not list(decoder.named_buffers())
    assert len(encoder_states[0]["convolutions"]) == 5
    assert len(decoder_states[0]["convolutions"]) == 10
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(torch.stack(actual, 1), reference, atol=2e-10, rtol=2e-10)


def test_stream_context_is_restored_on_failure_and_shape_change_requires_reset():
    encoder, decoder = codec()
    frame = torch.randn(1, 6, dtype=torch.complex128)
    features, state = encoder.step(frame)
    key = next(iter(state["convolutions"]))
    state["convolutions"][key] = state["convolutions"][key].float()
    with pytest.raises(ValueError, match="reset state"):
        encoder.step(frame, state)
    # A failed step must not leave ambient streaming mode enabled for a batch.
    full = encoder(frame.unsqueeze(1).expand(-1, 3, -1))
    assert full.shape[1] == 3
    _, state = encoder.step(frame)
    with pytest.raises(ValueError, match="shape changed"):
        encoder.step(frame[:, :-1], state)
    _, decoder_state = decoder.step(features, frame)
    with pytest.raises(ValueError, match="shape changed"):
        decoder.step(features, frame[:, :-1], decoder_state)


def test_relu_log1p_mapping_matches_recipe_and_can_generate_missing_bins():
    encoder, decoder = codec()
    silence = torch.zeros(1, 3, 6, dtype=torch.complex128, requires_grad=True)
    with torch.no_grad():
        decoder.magnitude.final_conv.weight.zero_()
        decoder.magnitude.final_conv.bias.fill_(0.2)
    output = decoder(encoder(silence), silence)
    torch.testing.assert_close(output.abs(), torch.full_like(output.real, 0.2).expm1())
    output.real.sum().backward()
    assert torch.isfinite(silence.grad).all()
    assert torch.count_nonzero(output) == output.numel()
