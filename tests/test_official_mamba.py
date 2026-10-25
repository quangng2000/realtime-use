"""Official-backend interface checks and explicitly GPU-gated numerical tests.

CPU stubs verify selection/errors only. They do not claim Mamba numerical parity.
"""

import builtins
import io

import pytest
import torch
from torch import nn

from realtime_use.audio import StreamingEnhancer
from realtime_use.config import ModelConfig
from realtime_use.export import export_exit
from realtime_use.infer import enhance_waveform, load_model
from realtime_use.model import Enhancer
from realtime_use import ssm


class _InterfaceOnlyMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.constructor_arguments = kwargs
        self.placeholder = nn.Parameter(torch.zeros(1))

    def forward(self, value):
        raise AssertionError("CPU interface stub must never perform numerical execution")

    def step(self, *args):
        raise AssertionError("CPU interface stub must never perform numerical execution")


def test_official_model_constructs_complete_upstream_modules(monkeypatch):
    monkeypatch.setattr(ssm, "_load_official_mamba", lambda: _InterfaceOnlyMamba)
    cfg = ModelConfig(depth=1, exits=(1,), width=8, state_dim=4, expand=2,
                      conv_kernel=4, mixer_impl="mamba_ssm", scan_backend="auto")
    model = Enhancer(cfg)
    block = model.blocks[0]
    for mixer in (block.frequency_forward, block.frequency_backward, block.temporal):
        assert isinstance(mixer, ssm.OfficialMambaMixer)
        assert isinstance(mixer.mamba, _InterfaceOnlyMamba)
        assert mixer.mamba.constructor_arguments == {"d_model": 8, "d_state": 4, "d_conv": 4, "expand": 2}
    assert "blocks.0.temporal.mamba.placeholder" in model.state_dict()


@pytest.mark.parametrize("operation", ["forward", "step"])
def test_official_backend_explicitly_rejects_cpu_execution(monkeypatch, operation):
    monkeypatch.setattr(ssm, "_load_official_mamba", lambda: _InterfaceOnlyMamba)
    mixer = ssm.OfficialMambaMixer(8, 4, 2, 4)
    value = torch.zeros(1, 3, 8) if operation == "forward" else torch.zeros(1, 8)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        getattr(mixer, operation)(value)


def test_missing_official_dependency_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def missing_mamba(name, *args, **kwargs):
        if name == "mamba_ssm":
            raise ModuleNotFoundError("test dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_mamba)
    with pytest.raises(RuntimeError, match="Install this project's cuda extra"):
        ssm.OfficialMambaMixer(8, 4, 2, 4)


def test_legacy_model_never_imports_official_package(monkeypatch):
    def forbidden_import():
        raise AssertionError("legacy selection must not import the full Mamba package")

    monkeypatch.setattr(ssm, "_load_official_mamba", forbidden_import)
    model = Enhancer(ModelConfig(depth=1, exits=(1,), width=4, state_dim=2, expand=1))
    assert isinstance(model.blocks[0].temporal, ssm.SelectiveMixer)


@pytest.fixture
def official_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Official Mamba numerical tests require CUDA; CPU stubs are interface checks only")
    # A CUDA host is the production preflight: a missing or broken required
    # dependency must fail visibly instead of turning that preflight green.
    official_class = ssm._load_official_mamba()
    torch.manual_seed(47)
    return official_class


def test_official_cuda_matches_direct_class_and_backward(official_cuda):
    adapter = ssm.OfficialMambaMixer(8, 4, 2, 4).cuda().eval()
    direct = official_cuda(d_model=8, d_state=4, d_conv=4, expand=2).cuda().eval()
    direct.load_state_dict(adapter.mamba.state_dict(), strict=True)
    left = torch.randn(2, 11, 8, device="cuda", requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    actual, expected = adapter(left), direct(right)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)
    for ours, upstream in zip(adapter.mamba.parameters(), direct.parameters()):
        assert ours.grad is not None and torch.isfinite(ours.grad).all()
        torch.testing.assert_close(ours.grad, upstream.grad, atol=2e-5, rtol=2e-4)
    assert adapter.mamba.A_log._no_weight_decay
    assert adapter.mamba.D._no_weight_decay


def test_official_cuda_sequence_step_and_serialized_stream_state(official_cuda):
    mixer = ssm.OfficialMambaMixer(8, 4, 2, 4).cuda().eval()
    values = torch.randn(2, 11, 8, device="cuda")
    with torch.inference_mode():
        full = mixer(values)
        state, outputs = None, []
        for token in values[:, :5].unbind(1):
            output, state = mixer.step(token, state)
            outputs.append(output)
        saved = io.BytesIO()
        torch.save(state, saved)
        saved.seek(0)
        restored = torch.load(saved, weights_only=True, map_location="cuda")
        for token in values[:, 5:].unbind(1):
            output, state = mixer.step(token, state)
            resumed, restored = mixer.step(token, restored)
            torch.testing.assert_close(output, resumed, atol=2e-5, rtol=2e-4)
            outputs.append(output)
        torch.testing.assert_close(torch.stack(outputs, 1), full, atol=2e-5, rtol=2e-4)
        _, independent = mixer.step(values[:, 0])
        assert state["conv"].data_ptr() != independent["conv"].data_ptr()
        assert state["ssm"].data_ptr() != independent["ssm"].data_ptr()
        with pytest.raises(ValueError, match="reset"):
            mixer.step(values[:1, 0], state)


@pytest.mark.parametrize("decoder_mode", ["shared", "per_exit"])
def test_official_cuda_enhancer_reload_streaming_and_export(official_cuda, decoder_mode, tmp_path):
    cfg = ModelConfig(sample_rate=800, depth=2, exits=(1, 2), width=8, state_dim=4,
                      expand=2, conv_kernel=4, decoder_mode=decoder_mode,
                      mixer_impl="mamba_ssm", scan_backend="auto", codec_impl="nemotron_causal")
    original = Enhancer(cfg).cuda().eval()
    restored = Enhancer(ModelConfig.from_dict(cfg.to_dict())).cuda().eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    waveform = torch.randn(1, 3 * original.transform.hop_samples + 7, device="cuda")
    with torch.inference_mode():
        expected = original.enhance(waveform)
        reloaded = restored.enhance(waveform)
        for depth in cfg.exits:
            torch.testing.assert_close(reloaded[depth], expected[depth], atol=2e-5, rtol=2e-4)
            stream = StreamingEnhancer(restored, depth)
            parts = [stream.push(chunk) for chunk in waveform.split(7, dim=-1)]
            parts.append(stream.flush())
            actual = torch.cat(parts, dim=-1)
            assert actual.shape == waveform.shape
            torch.testing.assert_close(actual, expected[depth], atol=3e-4, rtol=3e-4)
            stream.reset()
            torch.testing.assert_close(torch.cat([stream.push(waveform), stream.flush()], -1), actual,
                                       atol=3e-4, rtol=3e-4)
    checkpoint = tmp_path / "official.pt"
    torch.save({"model_config": cfg.to_dict(), "model_state": original.state_dict()}, checkpoint)
    extracted = export_exit(checkpoint, 1, tmp_path / "exit1.pt", device="cuda")
    selected = load_model(extracted, device="cuda")
    assert selected.config.mixer_impl == "mamba_ssm"
    assert selected.config.depth == 1 and selected.config.exits == (1,)
    assert selected.config.decoder_mode == "shared"
    torch.testing.assert_close(enhance_waveform(selected, waveform, 1), expected[1], atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(enhance_waveform(selected, waveform, 1, streaming=True, chunk_samples=7),
                               expected[1], atol=3e-4, rtol=3e-4)
    saved = torch.load(extracted, map_location="cpu", weights_only=True)
    assert saved["verification"]["backend"] == "mamba_ssm"
    assert saved["verification"]["device"].startswith("cuda")


def test_official_cuda_native_regression_gan_and_resume_preflight(official_cuda, tmp_path):
    """Synthetic execution preflight on the real backend, without a quality claim."""
    from hydra import compose, initialize_config_module
    from realtime_use.train import run_training

    with initialize_config_module(version_base="1.3", config_module="realtime_use.conf"):
        cfg = compose(config_name="config", overrides=[
            f"output_dir={tmp_path / 'regression'}", "data.smoke=true",
            # Do not enable top-level smoke: it explicitly selects the legacy CPU
            # architecture and would skip the production backend being tested.
            "model.mixer_impl=mamba_ssm", "model.scan_backend=auto",
            "model.sample_rate=800", "model.window_ms=40", "model.hop_ms=20",
            "model.depth=2", "model.exits=[1,2]", "model.width=8", "model.state_dim=4",
            "model.expand=2", "model.conv_kernel=4", "training.steps=2",
            "training.batch_size=1", "training.crop_seconds=0.16", "training.workers=0",
            "training.grad_accum=2", "training.sample_exit=false", "training.split_decoders_at=1",
            "training.checkpoint_every=1", "training.validate_every=2",
            "training.validation_batches=1", "training.log_every=1",
            "loss.fft_sizes=[16,32]", "loss.kd=0", "adversarial.channels=4",
            "adversarial.scales=1", "trainer.accelerator=gpu", "trainer.devices=1",
            "trainer.strategy=auto", "trainer.precision=32-true",
            "trainer.enable_progress_bar=false", "trainer.enable_model_summary=false",
            "trainer.num_sanity_val_steps=0",
        ])
    regression = run_training(cfg)
    pretrained = torch.load(regression, map_location="cpu", weights_only=True)
    assert pretrained["step"] == pretrained["global_step"] == 2
    assert pretrained["model_config"]["mixer_impl"] == "mamba_ssm"
    assert pretrained["model_config"]["decoder_mode"] == "per_exit"
    assert "cuda_rng_state" in pretrained["rng_states_by_rank"][0]
    assert any(group["name"] == "backbone" and group["weight_decay"] == 0
               for group in pretrained["optimizer_states"][0]["param_groups"])

    cfg.training.stage = "adversarial"
    cfg.training.steps = 1
    cfg.init_from = str(regression)
    cfg.output_dir = str(tmp_path / "adversarial")
    gan = run_training(cfg)
    trained = torch.load(gan, map_location="cpu", weights_only=True)
    assert trained["step"] == 1 and trained["global_step"] == 2
    assert len(trained["optimizer_states"]) == 2
    assert trained["optimizer_states"][1]["state"]
    assert "cuda_rng_state" in trained["rng_states_by_rank"][0]
    cfg.init_from = None
    cfg.resume = str(gan)
    cfg.training.steps = 2
    resumed = run_training(cfg)
    final = torch.load(resumed, map_location="cpu", weights_only=True)
    assert final["step"] == 2 and final["global_step"] == 4
    assert final["model_config"]["mixer_impl"] == "mamba_ssm"
    assert any(not torch.equal(trained["model_state"][name], value)
               for name, value in final["model_state"].items())
    model = load_model(resumed, device="cuda")
    assert torch.isfinite(enhance_waveform(model, torch.randn(128), 1)).all()
