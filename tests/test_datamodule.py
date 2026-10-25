from copy import deepcopy
import json

from lightning import LightningDataModule, seed_everything
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from realtime_use.datamodule import PairedAudioDataModule


def configuration(**training):
    return {
        "model": {"sample_rate": 8000, "depth": 1, "exits": [1], "width": 4, "state_dim": 2, "expand": 1},
        "training": {"stage": "regression", "batch_size": 2, "crop_seconds": 0.01, "workers": 0, **training},
        "data": {"smoke": False, "pin_memory": False},
        "loss": {"kd": 0.0},
    }


def manifest(tmp_path, name="pairs", *, lengths=(20, 30), teacher=None):
    rows = []
    for index, samples in enumerate(lengths):
        clean = np.linspace(-0.2, 0.2, samples, dtype=np.float32)
        row = {"sample_rate": 8000, "id": f"{name}-{index}"}
        for kind, values in (("clean", clean), ("noisy", clean + 0.01)):
            filename = f"{name}-{index}-{kind}.wav"
            sf.write(tmp_path / filename, values, 8000, subtype="FLOAT")
            row[kind] = filename
        if teacher == "present":
            row["teacher"] = row["clean"]
        elif teacher == "missing-file":
            row["teacher"] = "missing-teacher.wav"
        rows.append(row)
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_strict_paired_setup_preserves_lengths_and_loader_policies(tmp_path):
    cfg = configuration()
    cfg["data"].update(train_manifest=manifest(tmp_path), validation_manifest=manifest(tmp_path, "val"))
    dm = PairedAudioDataModule(cfg)
    assert isinstance(dm, LightningDataModule)
    dm.setup("fit")
    first_dataset = dm.train_dataset
    dm.setup("fit")
    assert dm.train_dataset is first_dataset
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert not train_loader.persistent_workers
    assert not train_loader.pin_memory
    assert train_loader.drop_last is False
    batch = next(iter(train_loader))
    assert batch["noisy"].shape == (2, 80)
    assert sorted(batch["lengths"].tolist()) == [20, 30]
    assert "teacher" not in batch
    validation = next(iter(val_loader))
    assert validation["ids"] == ["val-0", "val-1"]
    assert validation["lengths"].tolist() == [20, 30]


@pytest.mark.parametrize("stage,kd", [("regression", 0.0), ("adversarial", 1.0)])
def test_teacher_free_stages_never_open_teacher_paths(tmp_path, stage, kd):
    cfg = configuration(stage=stage)
    cfg["loss"]["kd"] = kd
    cfg["data"]["train_manifest"] = manifest(tmp_path, teacher="missing-file")
    cfg["data"]["validation_manifest"] = cfg["data"]["train_manifest"]
    dm = PairedAudioDataModule(cfg)
    dm.setup("fit")
    assert "teacher" not in next(iter(dm.train_dataloader()))
    assert "teacher" not in next(iter(dm.val_dataloader()))


def test_kd_requires_targets_and_validation_ignores_them(tmp_path):
    cfg = configuration()
    cfg["loss"]["kd"] = 0.5
    cfg["data"].update(train_manifest=manifest(tmp_path, teacher="present"),
                        validation_manifest=manifest(tmp_path, "val", teacher="missing-file"))
    dm = PairedAudioDataModule(cfg)
    dm.setup("fit")
    assert "teacher" in next(iter(dm.train_dataloader()))
    assert "teacher" not in next(iter(dm.val_dataloader()))
    cfg["data"]["train_manifest"] = manifest(tmp_path, "no-target")
    with pytest.raises(ValueError, match="every training row"):
        PairedAudioDataModule(cfg).setup("fit")


def test_smoke_is_explicit_and_nonempty_even_with_large_batch():
    cfg = configuration(batch_size=16)
    cfg["data"]["smoke"] = True
    dm = PairedAudioDataModule(cfg)
    train_loader = dm.train_dataloader()
    assert len(train_loader) == 1
    batch = next(iter(train_loader))
    assert batch["noisy"].shape == (8, 80)
    assert batch["lengths"].tolist() == [80] * 8
    assert all(value.startswith("synthetic-") for value in batch["ids"])
    assert dm.has_validation
    assert next(iter(dm.val_dataloader()))["noisy"].shape == (2, 80)
    cfg["loss"]["kd"] = 1
    with pytest.raises(ValueError, match="no teacher"):
        PairedAudioDataModule(cfg)


def test_real_training_cannot_silently_fall_back_to_smoke(tmp_path):
    with pytest.raises(ValueError, match="train_manifest"):
        PairedAudioDataModule(configuration()).train_dataloader()
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    cfg = configuration()
    cfg["data"]["train_manifest"] = str(empty)
    with pytest.raises(ValueError, match="no audio"):
        PairedAudioDataModule(cfg).setup("fit")


def test_validation_can_run_without_training_manifest_and_is_optional(tmp_path):
    cfg = configuration()
    dm = PairedAudioDataModule(cfg)
    assert not dm.has_validation
    assert dm.val_dataloader() == []
    cfg["data"]["validation_manifest"] = manifest(tmp_path)
    validation_only = PairedAudioDataModule(cfg)
    validation_only.setup("validate")
    assert validation_only.train_dataset is None
    assert len(next(iter(validation_only.val_dataloader()))["ids"]) == 2


def test_worker_persistence_follows_worker_count_and_pin_memory_setting():
    cfg = configuration(workers=2)
    cfg["data"].update(smoke=True, pin_memory=True)
    dm = PairedAudioDataModule(cfg)
    loader = dm.train_dataloader()
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.pin_memory is True
    # Do not start subprocesses here; Lightning injects the worker seed function
    # when attaching this loader to a Trainer seeded with workers=True.
    assert loader.worker_init_fn is None


def test_fresh_seeded_runs_repeat_shuffle_and_crops(tmp_path):
    cfg = configuration()
    cfg["data"]["train_manifest"] = manifest(tmp_path, lengths=(200, 210, 220, 230))
    seed_everything(123, workers=True)
    first = next(iter(PairedAudioDataModule(cfg).train_dataloader()))
    seed_everything(123, workers=True)
    second = next(iter(PairedAudioDataModule(cfg).train_dataloader()))
    assert first["ids"] == second["ids"]
    torch.testing.assert_close(first["noisy"], second["noisy"])
    torch.testing.assert_close(first["clean"], second["clean"])


@pytest.mark.parametrize("section,key,value", [
    ("training", "batch_size", 0),
    ("training", "workers", -1),
    ("training", "workers", 1.5),
    ("training", "crop_seconds", 0),
    ("training", "crop_seconds", float("nan")),
    ("training", "crop_seconds", 1e-10),
    ("training", "stage", "typo"),
    ("data", "smoke", "false"),
    ("loss", "kd", -1),
])
def test_invalid_data_settings_fail_early(section, key, value):
    cfg = deepcopy(configuration())
    cfg[section][key] = value
    with pytest.raises(ValueError):
        PairedAudioDataModule(cfg)
