"""Causal author-codec integration using explicitly selected CPU reference mixers.

These tests exercise architecture/state contracts, not official Mamba CUDA
kernels, speech quality, or equivalence to NVIDIA's trained checkpoints.
"""

from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.strategies import DDPStrategy
import pytest
import torch
from torch.utils.data import DataLoader

from realtime_use.config import ModelConfig
from realtime_use.export import export_exit
from realtime_use.infer import enhance_waveform, load_model
from realtime_use.lightning_module import EnhancementModule
from realtime_use.model import Enhancer
from realtime_use.train import run_training


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def author_config(**overrides):
    settings = dict(sample_rate=800, depth=2, exits=(1, 2), width=4, state_dim=2,
                    expand=1, conv_kernel=2, mixer_impl="legacy", scan_backend="reference",
                    codec_impl="nemotron_causal")
    settings.update(overrides)
    return ModelConfig(**settings)


def hydra_config(output_dir):
    directory = Path(__file__).resolve().parents[1] / "src/realtime_use/conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        return compose(config_name="config", overrides=[
            f"output_dir={output_dir}", "data.smoke=true", "training.steps=1",
            "training.grad_accum=2", "training.batch_size=1", "training.crop_seconds=0.16",
            "training.validate_every=1", "training.validation_batches=1",
            "training.checkpoint_every=10", "training.log_every=1",
            "training.sample_exit=true", "training.split_decoders_at=1",
            "model.codec_impl=nemotron_causal", "model.mixer_impl=legacy",
            "model.scan_backend=reference", "model.sample_rate=800", "model.depth=2",
            "model.exits=[1,2]", "model.width=4", "model.state_dim=2",
            "model.expand=1", "model.conv_kernel=2", "loss.fft_sizes=[16,32]",
            "adversarial.fft_sizes=[64]", "trainer.accelerator=cpu",
            "trainer.enable_progress_bar=false", "trainer.enable_model_summary=false",
        ])


def test_hydra_presets_select_author_codec_and_official_mamba_by_default():
    directory = Path(__file__).resolve().parents[1] / "src/realtime_use/conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        for name in ("fixed4", "fixed8", "fixed12", "flexible12"):
            config = compose(config_name="config", overrides=[f"experiment={name}"])
            assert config.model.codec_impl == "nemotron_causal"
            assert config.model.mixer_impl == "mamba_ssm"


@pytest.mark.parametrize("frequency_bins", [16, 17, 18])
@pytest.mark.parametrize("decoder_mode", ["shared", "per_exit"])
def test_author_model_full_stream_causality_and_gradients(frequency_bins, decoder_mode):
    model = Enhancer(author_config(decoder_mode=decoder_mode))
    spectrum = torch.randn(1, 6, frequency_bins, dtype=torch.complex64, requires_grad=True)
    outputs = model(spectrum)
    changed = spectrum.detach().clone()
    changed[:, 4:] += 30 * torch.randn_like(changed[:, 4:])
    changed_outputs = model(changed)
    for depth, expected in outputs.items():
        state, pieces = None, []
        for frame in spectrum.unbind(1):
            actual, state = model.step(frame, state, depth)
            pieces.append(actual)
        assert expected.shape == spectrum.shape
        torch.testing.assert_close(torch.stack(pieces, 1), expected, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(changed_outputs[depth][:, :4], expected[:, :4],
                                   atol=2e-5, rtol=2e-4)
        reset, _ = model.step(spectrum[:, 0], None, depth)
        torch.testing.assert_close(reset, expected[:, 0], atol=2e-5, rtol=2e-4)
    objective = sum(value.real.square().mean() + 0.7 * value.imag.square().mean()
                    for value in outputs.values())
    objective.backward()
    assert torch.isfinite(spectrum.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in model.encoder.parameters())


def test_author_shared_decoder_transition_preserves_outputs_and_registration():
    model = Enhancer(author_config())
    model.prepare_decoder_transition()
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    spectrum = torch.randn(1, 4, 17, dtype=torch.complex64)
    before = model(spectrum)
    model.activate_prepared_decoders()
    after = model(spectrum)
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    for depth in model.config.exits:
        torch.testing.assert_close(after[depth], before[depth])
        for name, value in model.shared_decoder.state_dict().items():
            torch.testing.assert_close(model.decoders[str(depth)].state_dict()[name], value)
    first, second = (next(model.decoders[str(depth)].parameters()) for depth in model.config.exits)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("decoder_mode", ["shared", "per_exit"])
def test_author_exit_export_preserves_selected_head_and_streaming(tmp_path, decoder_mode):
    model = Enhancer(author_config(depth=3, exits=(1, 2, 3), decoder_mode=decoder_mode)).eval()
    if decoder_mode == "per_exit":
        with torch.no_grad():
            for depth, head in model.decoders.items():
                for parameter in head.parameters():
                    parameter.add_(0.003 * int(depth))
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_config": model.config.to_dict(), "model_state": model.state_dict()}, checkpoint)
    path = export_exit(checkpoint, 2, tmp_path / "exit2.pt")
    exported = load_model(path)
    assert exported.config.codec_impl == "nemotron_causal"
    assert exported.config.exits == (2,)
    assert len(exported.blocks) == 2
    assert sum(parameter.numel() for parameter in exported.parameters()) == model.active_parameter_count(2)
    source_head = model.shared_decoder if decoder_mode == "shared" else model.decoders["2"]
    for key, value in exported.shared_decoder.state_dict().items():
        torch.testing.assert_close(value, source_head.state_dict()[key], rtol=0, atol=0)
    waveform = 0.05 * torch.randn(1, 117)
    expected = enhance_waveform(model, waveform, 2)
    for chunk in (1, 13, 32):
        actual = enhance_waveform(exported, waveform, 2, streaming=True, chunk_samples=chunk)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("stage", ["regression", "adversarial"])
@pytest.mark.training_loop
def test_author_training_native_resume_split_and_architecture_guards(tmp_path, stage):
    config = hydra_config(tmp_path / "regression")
    path = run_training(config)
    if stage == "adversarial":
        config.training.stage = stage
        config.init_from = str(path)
        config.output_dir = str(tmp_path / "adversarial")
        path = run_training(config)
        config.init_from = None
    first = torch.load(path, weights_only=True)
    assert first["model_config"]["codec_impl"] == "nemotron_causal"
    assert first["model_config"]["decoder_mode"] == "shared"
    assert first["step"] == 1
    config.resume = str(path)
    config.training.steps = 2
    resumed = torch.load(run_training(config), weights_only=True)
    assert resumed["step"] == 2
    assert resumed["global_step"] == (4 if stage == "adversarial" else 2)
    assert resumed["model_config"]["decoder_mode"] == "per_exit"
    assert any(not torch.equal(first["model_state"][key], value)
               for key, value in resumed["model_state"].items() if key in first["model_state"])
    config.training.steps = 3
    if stage == "adversarial":
        config.adversarial.implementation = "legacy"
        with pytest.raises(ValueError, match="preserve adversarial"):
            run_training(config)
        config.adversarial.implementation = "dac_mrd"
    incompatible = EnhancementModule({
        "model": author_config(codec_impl="legacy").to_dict(), "training": {"stage": stage},
    })
    with pytest.raises(ValueError, match="architecture"):
        incompatible.load_portable_weights(first)
    config.model.codec_impl = "legacy"
    config.training.steps = 3
    with pytest.raises(ValueError, match="codec_impl"):
        run_training(config)


def _ddp_worker(output_dir):
    """Executed only by torchrun, not during pytest collection."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    module = EnhancementModule({
        "model": author_config().to_dict(),
        "training": {"stage": "regression", "steps": 2, "grad_accum": 2,
                     "sample_exit": True, "split_decoders_at": 1},
        "loss": {"implementation": "auraloss", "fft_sizes": [16, 32]},
    })
    rows = []
    for index in range(8):
        clean = 0.05 * torch.randn(93, generator=torch.Generator().manual_seed(index))
        rows.append({"noisy": clean + 0.01, "clean": clean, "lengths": torch.tensor(93)})

    class Capture(Callback):
        def on_train_end(self, trainer, model):
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(destination / "native.ckpt")
            torch.save({"state": model.state_dict(), "updates": int(model.completed_updates),
                        "optimizers": [optimizer.state_dict() for optimizer in trainer.optimizers],
                        "decoder_mode": model.model.config.decoder_mode},
                       destination / f"rank{trainer.global_rank}.pt")

    runner = Trainer(accelerator="cpu", devices=2,
                     strategy=DDPStrategy(find_unused_parameters=True, static_graph=False,
                                          broadcast_buffers=False),
                     max_steps=2, max_epochs=-1, accumulate_grad_batches=1,
                     enable_checkpointing=False, enable_progress_bar=False,
                     enable_model_summary=False, logger=False, callbacks=[Capture()])
    runner.fit(module, train_dataloaders=DataLoader(rows, batch_size=1))


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="Requires two-process CPU/Gloo support")
@pytest.mark.training_loop
def test_author_codec_two_rank_training_split_and_optimizer_agreement(tmp_path):
    from test_ddp import assert_nested_close, run_process, torchrun_prefix

    run_process(torchrun_prefix() + [str(Path(__file__).resolve()), "--ddp-worker", str(tmp_path)])
    first, second = (torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in (0, 1))
    assert first["updates"] == second["updates"] == 2
    assert first["decoder_mode"] == second["decoder_mode"] == "per_exit"
    assert_nested_close(first["state"], second["state"], rtol=0, atol=0)
    assert_nested_close(first["optimizers"], second["optimizers"], rtol=0, atol=0)
    checkpoint = torch.load(tmp_path / "native.ckpt", weights_only=True)
    assert checkpoint["world_size"] == 2
    assert checkpoint["model_config"]["codec_impl"] == "nemotron_causal"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--ddp-worker":
        _ddp_worker(sys.argv[2])
    else:
        raise SystemExit("This file only supports --ddp-worker OUTPUT_DIRECTORY")
