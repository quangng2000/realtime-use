"""Native checkpoints remain resumable after explicit final validation."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch

from realtime_use.infer import enhance_waveform, load_model
from realtime_use.train import run_training


def tiny_config(output_dir):
    config_dir = Path(__file__).resolve().parents[1] / "src" / "realtime_use" / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="config", overrides=[
            f"output_dir={output_dir}",
            # Keep top-level smoke=false so prepare_config does not replace the
            # non-multiple validation interval or accumulation under test.
            "data.smoke=true", "training.steps=1", "training.grad_accum=2",
            "training.validate_every=2", "training.checkpoint_every=10",
            "training.validation_batches=1", "training.log_every=1",
            "training.batch_size=1", "training.crop_seconds=0.008",
            "model.sample_rate=8000", "model.window_ms=4", "model.hop_ms=2",
            "model.depth=1", "model.exits=[1]", "model.width=4", "model.state_dim=2",
            "model.expand=1", "model.conv_kernel=2", "model.scan_backend=reference",
            "loss.fft_sizes=[32,64]", "adversarial.channels=2", "adversarial.scales=1",
            "trainer.accelerator=cpu", "trainer.enable_progress_bar=false",
            "trainer.enable_model_summary=false", "trainer.num_sanity_val_steps=0",
        ])


@pytest.mark.parametrize("stage", ["regression", "adversarial"])
def test_final_validation_checkpoint_retains_native_resume_state(tmp_path, stage):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = tiny_config(tmp_path / "regression")
        path = run_training(cfg)
        if stage == "adversarial":
            cfg.training.stage = "adversarial"
            cfg.init_from = str(path)
            cfg.output_dir = str(tmp_path / "adversarial")
            path = run_training(cfg)
            cfg.init_from = None
        saved = torch.load(path, map_location="cpu", weights_only=True)
        optimizer_count = 2 if stage == "adversarial" else 1
        assert saved["step"] == 1
        assert saved["global_step"] == optimizer_count
        assert len(saved["optimizer_states"]) == optimizer_count
        assert "fit_loop" in saved["loops"]
        best = torch.load(path.parent / "best.ckpt", map_location="cpu", weights_only=True)
        assert best["step"] == 1
        assert len(best["optimizer_states"]) == optimizer_count
        # Native .ckpt also carries the portable model metadata used by inference.
        model = load_model(path, "cpu")
        assert torch.isfinite(enhance_waveform(model, torch.randn(64), 1)).all()

        cfg.resume = str(path)
        cfg.training.steps = 2
        resumed_path = run_training(cfg)
        resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)
        assert resumed["step"] == 2
        assert resumed["global_step"] == 2 * optimizer_count
        assert len(resumed["optimizer_states"]) == optimizer_count
        assert any(not torch.equal(saved["model_state"][key], value)
                   for key, value in resumed["model_state"].items())
    finally:
        torch.set_num_threads(previous_threads)
