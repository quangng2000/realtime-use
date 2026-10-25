"""Lightning selects checkpoints and validates the final completed update."""

from pathlib import Path
import shutil

from hydra import compose, initialize_config_dir
import pytest
import torch

from realtime_use.infer import enhance_waveform, load_model
from lightning.pytorch.callbacks import ModelCheckpoint
from realtime_use.train import RunArtifacts, checkpoint_callbacks, run_training


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
            "model.expand=1", "model.conv_kernel=2", "model.scan_backend=reference", "model.mixer_impl=legacy", "model.codec_impl=legacy",
            "loss.fft_sizes=[32,64]", "adversarial.channels=2", "adversarial.scales=1", "adversarial.implementation=legacy",
            "trainer.accelerator=cpu", "trainer.enable_progress_bar=false",
            "trainer.enable_model_summary=false", "trainer.num_sanity_val_steps=0",
        ])


@pytest.mark.parametrize("stage", ["regression", "adversarial"])
@pytest.mark.training_loop
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


@pytest.mark.training_loop
def test_final_validation_stays_inside_fit_and_preserves_rng(tmp_path, monkeypatch):
    before = {}
    original = RunArtifacts.on_train_end

    def capture(callback, trainer, module):
        original(callback, trainer, module)
        before.update(module.capture_rng_state())

    monkeypatch.setattr(RunArtifacts, "on_train_end", capture)
    def reject_extra_validate(*args, **kwargs):
        raise AssertionError("Final validation must use Lightning's fit loop")
    monkeypatch.setattr("lightning.Trainer.validate", reject_extra_validate)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        path = run_training(tiny_config(tmp_path / "final_validation"))
        saved = torch.load(path, weights_only=True)["rng_states_by_rank"][0]
        assert saved["python_rng_state"] == before["python_rng_state"]
        torch.testing.assert_close(saved["torch_rng_state"], before["torch_rng_state"], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.training_loop
def test_pre_library_native_checkpoint_resumes_only_with_legacy_objective(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = tiny_config(tmp_path / "legacy_recipe")
        cfg.loss.implementation = "legacy"
        path = run_training(cfg)
        saved = torch.load(path, weights_only=True)
        saved["model_config"].pop("mixer_impl")
        saved["model_config"].pop("codec_impl")
        saved["hyper_parameters"]["config"]["model"].pop("mixer_impl")
        saved["hyper_parameters"]["config"]["model"].pop("codec_impl")
        saved["hyper_parameters"]["config"]["loss"].pop("implementation")
        for key in ("implementation", "fft_sizes", "band_width_hz"):
            saved["hyper_parameters"]["config"]["adversarial"].pop(key)
        torch.save(saved, path)
        cfg.resume = str(path)
        cfg.training.steps = 2
        cfg.loss.implementation = "auraloss"
        with pytest.raises(ValueError, match="resume must preserve loss"):
            run_training(cfg)
        cfg.loss.implementation = "legacy"
        resumed = torch.load(run_training(cfg), weights_only=True)
        assert resumed["step"] == 2
        assert resumed["model_config"]["mixer_impl"] == "legacy"
    finally:
        torch.set_num_threads(previous_threads)


def test_checkpoint_policy_uses_lightning_model_checkpoint(tmp_path):
    cfg = tiny_config(tmp_path)
    artifacts, checkpoints = checkpoint_callbacks(cfg)
    assert isinstance(artifacts, RunArtifacts)
    assert all(isinstance(callback, ModelCheckpoint) for callback in checkpoints)
    assert checkpoints[1].monitor == "validation_mean_clean_loss"


@pytest.mark.parametrize("stage", ["regression", "adversarial"])
@pytest.mark.training_loop
def test_periodic_names_count_generator_updates_and_resume_between_boundaries(tmp_path, stage):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = tiny_config(tmp_path / "initial")
        cfg.training.checkpoint_every = 1
        first = run_training(cfg)
        if stage == "adversarial":
            cfg.training.stage = stage
            cfg.init_from = str(first)
            cfg.output_dir = str(tmp_path / "gan")
            first = run_training(cfg)
            cfg.init_from = None
        assert (first.parent / "step_0000001.ckpt").is_file()
        cfg.resume = str(first)
        cfg.training.steps = 2
        last = run_training(cfg)
        assert (last.parent / "step_0000002.ckpt").is_file()
        saved = torch.load(last, weights_only=True)
        assert saved["step"] == 2
        assert any("ModelCheckpoint" in key for key in saved["callbacks"])
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("destination", ["same", "different", "missing_best"])
@pytest.mark.training_loop
def test_legacy_best_score_migrates_only_with_original_best_file(tmp_path, destination):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = tiny_config(tmp_path / "original")
        path = run_training(cfg)
        best_path = path.parent / "best.ckpt"
        original_best = best_path.read_bytes()
        legacy = torch.load(path, weights_only=True)
        # Emulate the old callback in an otherwise native, resumable checkpoint.
        # Zero beats the positive reconstruction loss from subsequent updates.
        legacy["callbacks"] = {"TrainingCheckpoints": {
            "best_score": 0.0, "last_periodic_update": -1}}
        torch.save(legacy, path)
        cfg.resume = str(path)
        cfg.training.steps = 2
        if destination == "different":
            cfg.output_dir = str(tmp_path / "continued")
            copied_best = Path(cfg.output_dir) / "checkpoints" / "best.ckpt"
            copied_best.parent.mkdir(parents=True)
            shutil.copyfile(best_path, copied_best)
            best_path = copied_best
        elif destination == "missing_best":
            best_path.unlink()

        resumed_path = run_training(cfg)
        resumed = torch.load(resumed_path, weights_only=True)
        selector = next(value for value in resumed["callbacks"].values()
                        if value.get("monitor") == "validation_mean_clean_loss")
        assert resumed["step"] == 2
        assert resumed["global_step"] == 2
        assert resumed["optimizer_states"]
        assert "TrainingCheckpoints" not in resumed["callbacks"]
        if destination == "same":
            assert selector["best_model_score"] == 0
            assert selector["best_k_models"] == {str(best_path): torch.tensor(0.0)}
            assert best_path.read_bytes() == original_best
            # Once migrated, the next resume uses Lightning's ordinary state.
            cfg.resume = str(resumed_path)
            cfg.training.steps = 3
            native = torch.load(run_training(cfg), weights_only=True)
            native_selector = next(value for value in native["callbacks"].values()
                                   if value.get("monitor") == "validation_mean_clean_loss")
            assert native_selector["best_model_score"] == 0
            assert best_path.read_bytes() == original_best
        else:
            assert selector["best_model_score"] > 0
            assert torch.load(best_path, weights_only=True)["step"] == 2
    finally:
        torch.set_num_threads(previous_threads)
