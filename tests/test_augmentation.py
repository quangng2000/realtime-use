"""Exercise real audiomentations transforms, without replacing their DSP."""

import builtins
from copy import deepcopy
import json

from lightning import seed_everything
import numpy as np
import pytest
import soundfile as sf
import torch

from realtime_use.augmentation import build_input_augmentation
from realtime_use.data import PairedAudioDataset
from realtime_use.datamodule import PairedAudioDataModule


@pytest.fixture
def library():
    return pytest.importorskip("audiomentations")


def settings(transform="gaussian_snr"):
    result = {"enabled": True, "gaussian_snr": {"p": 0}, "clipping": {"p": 0}, "low_pass": {"p": 0}}
    result[transform]["p"] = 1
    if transform == "clipping":
        result[transform].update(min_percentile_threshold=20, max_percentile_threshold=20)
    return result


def manifest(tmp_path, length=512, teacher=False):
    time = np.arange(length, dtype=np.float32) / 16000
    clean = (0.2 * np.sin(2 * np.pi * 800 * time) + 0.1 * np.sin(2 * np.pi * 7000 * time)).astype(np.float32)
    sf.write(tmp_path / "clean.wav", clean, 16000, subtype="FLOAT")
    sf.write(tmp_path / "noisy.wav", clean, 16000, subtype="FLOAT")
    row = {"noisy": "noisy.wav", "clean": "clean.wav", "sample_rate": 16000}
    if teacher:
        row["teacher"] = "clean.wav"
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path, torch.from_numpy(clean)


@pytest.mark.parametrize("name,class_name", [
    ("gaussian_snr", "AddGaussianSNR"), ("clipping", "ClippingDistortion"), ("low_pass", "LowPassFilter"),
])
def test_real_transforms_preserve_clean_length_and_input_storage(tmp_path, library, name, class_name):
    path, clean = manifest(tmp_path)
    dataset = PairedAudioDataset(path, 16000, load_teacher=False, augmentation=settings(name))
    assert isinstance(dataset.augmentation.pipeline, library.Compose)
    assert isinstance(dataset.augmentation.pipeline.transforms[0], getattr(library, class_name))
    before = clean.clone()
    direct = dataset.augmentation(clean)
    torch.testing.assert_close(clean, before, rtol=0, atol=0)
    assert not torch.equal(direct, clean)
    item = dataset[0]
    torch.testing.assert_close(item["clean"], clean, rtol=0, atol=0)
    assert item["noisy"].shape == item["clean"].shape == (512,)
    assert item["noisy"].dtype == torch.float32
    assert item["length"] == 512
    assert torch.isfinite(item["noisy"]).all()
    assert not torch.equal(item["noisy"], item["clean"])


def test_augmentation_follows_shared_crop_and_leaves_padding_zero(tmp_path, library, monkeypatch):
    path, clean = manifest(tmp_path, length=64)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(7))
    cropped = PairedAudioDataset(path, 16000, crop_samples=32, load_teacher=False,
                                 augmentation=settings())[0]
    torch.testing.assert_close(cropped["clean"], clean[7:39], rtol=0, atol=0)
    assert cropped["length"] == 32
    padded = PairedAudioDataset(path, 16000, crop_samples=128, load_teacher=False,
                                augmentation=settings())[0]
    torch.testing.assert_close(padded["clean"][:64], clean, rtol=0, atol=0)
    assert padded["length"] == 64
    assert torch.count_nonzero(padded["noisy"][64:]) == 0
    assert torch.count_nonzero(padded["clean"][64:]) == 0
    assert not torch.equal(padded["noisy"][:64], clean)


def test_lightning_seed_repeats_real_library_noise_and_random_crop(tmp_path, library):
    path, _ = manifest(tmp_path)
    dataset = PairedAudioDataset(path, 16000, crop_samples=128, load_teacher=False,
                                 augmentation=settings())
    seed_everything(321, workers=True)
    first = dataset[0]
    seed_everything(321, workers=True)
    second = dataset[0]
    torch.testing.assert_close(first["noisy"], second["noisy"], rtol=0, atol=0)
    torch.testing.assert_close(first["clean"], second["clean"], rtol=0, atol=0)
    seed_everything(322, workers=True)
    assert not torch.equal(first["noisy"], dataset[0]["noisy"])


def test_disabled_and_validation_paths_do_not_import_optional_library(tmp_path, monkeypatch):
    path, clean = manifest(tmp_path, teacher=True)
    original = builtins.__import__

    def no_augmentation_import(name, *args, **kwargs):
        if name == "audiomentations":
            raise AssertionError("Optional augmentation dependency imported on a disabled path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_augmentation_import)
    assert build_input_augmentation({"enabled": False}, 16000, training=True, load_teacher=True) is None
    validation = PairedAudioDataset(path, 16000, training=False, augmentation=settings())[0]
    for key in ("noisy", "clean", "teacher"):
        torch.testing.assert_close(validation[key], clean, rtol=0, atol=0)


def test_cached_teacher_and_augmentation_rejected_at_dataset_setup(tmp_path):
    path, _ = manifest(tmp_path, teacher=True)
    with pytest.raises(ValueError, match="incompatible with cached teacher"):
        PairedAudioDataset(path, 16000, augmentation=settings())


def test_datamodule_passes_augmentation_to_training_only_and_guards_kd(tmp_path, library):
    path, clean = manifest(tmp_path, teacher=True)
    cfg = {
        "model": {"sample_rate": 16000, "depth": 1, "exits": [1], "width": 4, "state_dim": 2, "expand": 1},
        "training": {"stage": "regression", "batch_size": 1, "crop_seconds": len(clean) / 16000, "workers": 0},
        "data": {"train_manifest": str(path), "validation_manifest": str(path), "augmentation": settings()},
        "loss": {"kd": 0.0},
    }
    dm = PairedAudioDataModule(cfg)
    dm.setup("fit")
    train, validation = next(iter(dm.train_dataloader())), next(iter(dm.val_dataloader()))
    assert not torch.equal(train["noisy"], train["clean"])
    torch.testing.assert_close(validation["noisy"], validation["clean"], rtol=0, atol=0)
    assert "teacher" not in train and "teacher" not in validation
    bad = deepcopy(cfg)
    bad["loss"]["kd"] = 1.0
    with pytest.raises(ValueError, match="incompatible with cached teacher"):
        PairedAudioDataModule(bad).setup("fit")
    # Stage 2 does not load teacher targets, even when a reused config retains kd.
    bad["training"]["stage"] = "adversarial"
    PairedAudioDataModule(bad).setup("fit")
    cfg["data"]["smoke"] = True
    with pytest.raises(ValueError, match="requires real paired data"):
        PairedAudioDataModule(cfg)


def test_low_pass_rejects_silent_nyquist_clamping_and_phase_delay(library):
    with pytest.raises(ValueError, match="below Nyquist"):
        build_input_augmentation(settings("low_pass"), 8000, training=True, load_teacher=False)
    cfg = settings("low_pass")
    cfg["low_pass"]["zero_phase"] = False
    with pytest.raises(ValueError, match="phase alignment"):
        build_input_augmentation(cfg, 16000, training=True, load_teacher=False)


def test_too_short_low_pass_clip_reports_library_padding_requirement(tmp_path, library):
    path, _ = manifest(tmp_path, length=3)
    dataset = PairedAudioDataset(path, 16000, crop_samples=128, load_teacher=False,
                                 augmentation=settings("low_pass"))
    with pytest.raises(ValueError, match="3-sample clip.*filter padding"):
        dataset[0]
