import copy

import pytest
import torch

from realtime_use.config import ModelConfig
from realtime_use.model import Enhancer
from realtime_use.ssm import SelectiveMixer


@pytest.fixture(autouse=True)
def deterministic_threads():
    torch.manual_seed(7)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kwargs):
    return ModelConfig(**dict(dict(sample_rate=800, window_ms=40, hop_ms=20,
        depth=12, exits=(4, 8, 12), width=4, state_dim=2, expand=1,
        conv_kernel=3, scan_backend="reference"), **kwargs))


def test_old_checkpoint_config_retains_legacy_mixer_and_parameter_layout():
    settings = tiny_config().to_dict()
    settings.pop("mixer_impl")
    restored = Enhancer(ModelConfig.from_dict(settings))
    assert restored.config.mixer_impl == "legacy"
    assert isinstance(restored.blocks[0].temporal, SelectiveMixer)
    assert "blocks.0.temporal.in_projection.weight" in restored.state_dict()
    restored.load_state_dict(Enhancer(tiny_config()).state_dict(), strict=True)


@pytest.mark.parametrize("values", [
    {"mixer_impl": "unknown"},
    {"mixer_impl": "mamba_ssm", "scan_backend": "reference"},
    {"mixer_impl": "mamba_ssm", "scan_backend": "mamba_ssm"},
])
def test_mixer_selection_does_not_silently_change_architecture(values):
    with pytest.raises(ValueError):
        tiny_config(**values)


@pytest.mark.parametrize("kernel", [1, 4])
def test_selective_scan_matches_steps_and_has_finite_gradients(kernel):
    layer = SelectiveMixer(4, 3, 2, kernel, "reference")
    values = torch.randn(2, 7, 4, requires_grad=True)
    full = layer(values)
    state, pieces = None, []
    for token in values.unbind(1):
        output, state = layer.step(token, state)
        pieces.append(output)
    torch.testing.assert_close(full, torch.stack(pieces, 1), atol=2e-6, rtol=2e-5)
    full.square().mean().backward()
    assert torch.isfinite(values.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("decoder_mode", ["shared", "per_exit"])
def test_all_exits_match_frame_streaming_and_are_causal(decoder_mode):
    model = Enhancer(tiny_config(decoder_mode=decoder_mode)).eval()
    spectrum = torch.randn(2, 5, 5, dtype=torch.complex64)
    with torch.no_grad():
        full = model(spectrum)
        perturbed = spectrum.clone()
        perturbed[:, 3:] += 50 * torch.randn_like(perturbed[:, 3:])
        future_changed = model(perturbed)
        for depth in model.config.exits:
            state, frames = None, []
            for frame in spectrum.unbind(1):
                output, state = model.step(frame, state, depth)
                frames.append(output)
            torch.testing.assert_close(full[depth], torch.stack(frames, 1), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(full[depth][:, :3], future_changed[depth][:, :3])
            assert len(state["blocks"]) == depth


def test_decoder_split_preserves_predictions_but_not_parameter_sharing():
    model = Enhancer(tiny_config()).eval()
    spectrum = torch.randn(1, 3, 5, dtype=torch.complex64)
    before = model(spectrum)
    model.split_decoders()
    after = model(spectrum)
    for depth in model.config.exits:
        torch.testing.assert_close(before[depth], after[depth])
    assert model.config.decoder_mode == "per_exit"
    assert model.decoders["4"].output.weight.data_ptr() != model.decoders["8"].output.weight.data_ptr()
    assert sum(p.numel() for p in model.parameters()) > model.active_parameter_count(12)


@pytest.mark.parametrize("depth", [4, 8, 12])
def test_fixed_depth_has_only_its_own_active_parameters(depth):
    model = Enhancer(tiny_config(depth=depth, exits=(depth,)))
    assert model.active_parameter_count(depth) == sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError, match="subset"):
        model.active_parameter_count(1)


def test_exit_change_requires_stream_reset():
    model = Enhancer(tiny_config()).eval()
    frame = torch.randn(1, 5, dtype=torch.complex64)
    _, state = model.step(frame, exit_depth=4)
    with pytest.raises(ValueError, match="reset"):
        model.step(frame, state, exit_depth=8)


def test_decoder_can_generate_content_in_missing_input_bins():
    model = Enhancer(tiny_config(depth=4, exits=(4,))).eval()
    with torch.no_grad():
        model.shared_decoder.output.weight.zero_()
        model.shared_decoder.output.bias.copy_(torch.tensor([0.1, -0.2]))
        generated = model(torch.zeros(1, 3, 5, dtype=torch.complex64))[4]
    torch.testing.assert_close(generated, torch.full_like(generated, 0.1 - 0.2j))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity requires a GPU")
def test_cuda_selective_scan_forward_and_backward_parity():
    pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    reference = SelectiveMixer(8, 4, 2, 4, "reference").cuda()
    accelerated = copy.deepcopy(reference)
    accelerated.backend = "mamba_ssm"
    left = torch.randn(2, 11, 8, device="cuda", requires_grad=True)
    right = left.detach().clone().requires_grad_()
    a, b = reference(left), accelerated(right)
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)
    for first, second in zip(reference.parameters(), accelerated.parameters()):
        torch.testing.assert_close(first.grad, second.grad, atol=5e-5, rtol=5e-4)
