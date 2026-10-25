"""Paper profile wiring checks; no training loop or optimizer update is run."""

from copy import deepcopy

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
import pytest
import torch

from realtime_use.config import ModelConfig
from realtime_use.discriminators import DACMultiResolutionDiscriminator
from realtime_use.lightning_module import EnhancementModule
from realtime_use.train import checkpoint_callbacks, convergence_callbacks, prepare_config, _checkpoint_config


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_settings():
    return {
        "model": ModelConfig(sample_rate=800, sample_rates=(800, 1600),
            depth=2, exits=(1, 2), width=4, state_dim=2, expand=1, conv_kernel=2,
            codec_impl="nemotron_causal", lookahead_frames=(0, 1, 2), window_type="hann").to_dict(),
        "training": {"stage": "regression", "steps": 3, "sample_exit": True,
                     "sample_lookahead": True, "grad_accum": 1},
        "loss": {"implementation": "auraloss", "fft_sizes": [16, 32], "sample_rate_scaling": True},
        "adversarial": {"implementation": "dac_mrd", "fft_sizes": [32], "sample_rate_scaling": True},
        "validation": {"metrics": ["utmos"], "lookaheads": None},
    }


def test_paper_profile_resolves_without_starting_a_run(tmp_path):
    with initialize_config_module(version_base="1.3", config_module="realtime_use.conf"):
        cfg = compose(config_name="config", overrides=["experiment=paper_flexible12",
            f"output_dir={tmp_path / 'unused'}", f"data.validation_manifest={tmp_path / 'validation.jsonl'}"])
    values = prepare_config(cfg)
    assert values["model"]["exits"] == [4, 8, 12]
    assert values["model"]["sample_rates"] == [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    assert values["model"]["lookahead_frames"] == [0, 1, 2]
    assert values["model"]["window_type"] == "hann"
    assert values["trainer"]["accelerator"] == "gpu"
    assert values["training"]["sample_lookahead"] and values["training"]["sample_exit"]
    assert values["training"]["split_decoders_at"] is None
    assert values["training"]["validation_batches"] is None
    assert values["data"]["validation_full_utterance"]
    _, callbacks = checkpoint_callbacks(values)
    assert callbacks[1].monitor == "validation_mean_utmos" and callbacks[1].mode == "max"
    assert not (tmp_path / "unused").exists()


def test_paper_profile_rejects_missing_validation_or_wrong_metric(tmp_path):
    with initialize_config_module(version_base="1.3", config_module="realtime_use.conf"):
        cfg = compose(config_name="config", overrides=["experiment=paper_flexible12", f"output_dir={tmp_path}"])
    with pytest.raises(ValueError, match="validation_manifest"):
        prepare_config(cfg)
    cfg.data.validation_manifest = str(tmp_path / "validation.jsonl")
    cfg.validation.metrics = []
    with pytest.raises(ValueError, match="metrics"):
        prepare_config(cfg)
    cfg.validation.metrics = ["utmos"]
    cfg.data.validation_full_utterance = False
    with pytest.raises(ValueError, match="full_utterance"):
        prepare_config(cfg)


def test_multirate_module_forward_and_loss_use_native_rate_without_updates():
    module = EnhancementModule(tiny_settings())
    assert module._criterion_for_rate(800) is module.criterion
    assert module._criterion_for_rate(1600).minimum_spectral_length == 33
    before = {name: value.clone() for name, value in module.model.state_dict().items()}
    for rate in (800, 1600):
        batch = {"noisy": torch.randn(1, rate // 10) * 0.02, "sample_rate": rate}
        assert module._batch_rate(batch) == rate
        for lookahead in (0, 1, 2):
            prediction = module(batch["noisy"], sample_rate=rate, lookahead=lookahead)
            for output in prediction.values():
                loss, _ = module._criterion_for_rate(rate)(output, batch["noisy"])
                assert torch.isfinite(loss)
    assert module.completed_updates.item() == 0
    for name, value in module.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="homogeneous"):
        module._batch_rate({"sample_rate": torch.tensor([800, 1600])})


def test_validation_scores_full_valid_prefixes_all_exits_and_lookaheads():
    module = EnhancementModule(tiny_settings()).eval()
    calls = []

    class Scorer:
        def score(self, estimate, reference, sample_rate):
            calls.append((len(estimate), len(reference), sample_rate))
            return {"utmos": float(len(reference)) / 100}

    module._quality_evaluator = Scorer()
    logged = {}
    module.log_dict = lambda values, **kwargs: logged.update(values)
    batch = {"noisy": torch.randn(3, 111) * 0.02, "clean": torch.randn(3, 111) * 0.02,
             "lengths": torch.tensor([73, 111, 97]), "sample_rate": 1600,
             "distributed_valid": torch.tensor([True, False, True])}
    with torch.inference_mode():
        module.on_validation_epoch_start()
        module.validation_step(batch, 0)
        module.on_validation_epoch_end()
    assert len(calls) == 2 * 2 * 3
    assert set(calls) == {(73, 73, 1600), (97, 97, 1600)}
    assert logged["validation_mean_utmos"].item() == pytest.approx(0.85)
    assert all(metric.weight.item() == 2 for metric in module.validation_quality_metrics.values())
    assert len([key for key in logged if key.startswith("validation_utmos_exit_")]) == 6
    assert torch.isfinite(logged["validation_mean_clean_loss"])


def test_lookahead_selection_samples_only_configured_independent_branches():
    module = EnhancementModule(tiny_settings())
    selected = {module._select_lookahead() for _ in range(80)}
    assert selected == {0, 1, 2}
    module.training_config["sample_lookahead"] = False
    module.training_config["lookahead"] = 2
    assert module._select_lookahead() == 2


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000])
def test_official_dac_multirate_uses_shared_registered_bands_and_native_stft(rate):
    pytest.importorskip("dac")
    rates = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
    discriminator = DACMultiResolutionDiscriminator(16000, (256,), sample_rates=rates,
                                                   sample_rate_scaling=True)
    ids = {name: id(parameter) for name, parameter in discriminator.named_parameters()}
    original_geometry = [(item.sample_rate, item.window_length, item.stft_params, item.bands)
                         for item in discriminator.discriminators]
    waveform = (torch.randn(1, rate // 100) * 0.01).requires_grad_()
    outputs = discriminator(waveform, sample_rate=rate)
    assert len(outputs[0]) == 5 * ((rate + 7999) // 8000) + 1
    outputs[0][-1].square().mean().backward()
    assert torch.isfinite(waveform.grad).all()
    assert {name: id(parameter) for name, parameter in discriminator.named_parameters()} == ids
    assert [(item.sample_rate, item.window_length, item.stft_params, item.bands)
            for item in discriminator.discriminators] == original_geometry
    assert discriminator.fft_sizes_for_rate(rate) == (round(256 * rate / 16000),)


def test_adversarial_refinement_can_initialize_from_an_adversarial_checkpoint(tmp_path):
    with initialize_config_module(version_base="1.3", config_module="realtime_use.conf"):
        cfg = compose(config_name="config", overrides=[f"output_dir={tmp_path}"])
    cfg.training.stage = "adversarial"
    cfg.init_from = str(tmp_path / "previous.ckpt")
    cfg.training.split_decoders_at = 0
    settings = prepare_config(cfg)
    checkpoint = {"model_config": deepcopy(settings["model"]), "stage": "adversarial", "step": 10}
    _checkpoint_config(settings, checkpoint)
    assert settings["training"]["split_decoders_at"] == 0


def test_signal_selection_remains_legacy_compatible(tmp_path):
    settings = {"output_dir": str(tmp_path), "training": {"stage": "regression", "checkpoint_every": 100}}
    _, callbacks = checkpoint_callbacks(settings)
    assert callbacks[1].monitor == "validation_mean_clean_loss" and callbacks[1].mode == "min"


def test_optional_convergence_uses_lightning_early_stopping():
    assert convergence_callbacks({}) == []
    callbacks = convergence_callbacks({"validation": {"monitor": "utmos",
        "early_stopping": {"patience": 8, "min_delta": 0.01}}})
    assert len(callbacks) == 1
    assert callbacks[0].monitor == "validation_mean_utmos" and callbacks[0].mode == "max"
    assert callbacks[0].patience == 8 and callbacks[0].min_delta == 0.01


def test_preflight_does_not_fit_or_write_a_checkpoint(tmp_path, monkeypatch):
    import lightning
    from realtime_use.preflight import run_preflight

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must never train or update parameters")

    monkeypatch.setattr(lightning.Trainer, "fit", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    with initialize_config_module(version_base="1.3", config_module="realtime_use.conf"):
        cfg = compose(config_name="config", overrides=[f"output_dir={tmp_path / 'unused'}"])
    cfg.model = OmegaConf.create(tiny_settings()["model"])
    cfg.data.smoke = True
    cfg.training.crop_seconds = 0.1
    cfg.loss.fft_sizes = [16, 32]
    report = run_preflight(cfg)
    assert report["optimizer_updates"] == 0 and report["training_started"] is False
    assert len(report["checks"]) == 12
    assert report["configured_sample_rates"] == [800, 1600]
    assert not (tmp_path / "unused").exists()
