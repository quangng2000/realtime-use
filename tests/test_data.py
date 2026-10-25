import json

import numpy as np
import pytest
import soundfile as sf
import torch

from realtime_use.data import PairedAudioDataset, SyntheticDataset, collate_audio


def write_manifest(tmp_path, arrays, rates=None, row_rate=16000):
    row = {"sample_rate": row_rate}
    for name, array in arrays.items():
        sf.write(tmp_path / f"{name}.wav", np.asarray(array, dtype=np.float32), (rates or {}).get(name, 16000), subtype="FLOAT")
        row[name] = f"{name}.wav"
    manifest = tmp_path / "paired.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_shared_crop_includes_teacher(tmp_path, monkeypatch):
    clean = np.arange(12, dtype=np.float32) / 20
    manifest = write_manifest(tmp_path, {"clean": clean, "noisy": clean + 0.1, "teacher": clean + 0.2})
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(3))
    item = PairedAudioDataset(manifest, 16000, crop_samples=5)[0]
    torch.testing.assert_close(item["clean"], torch.from_numpy(clean[3:8]))
    torch.testing.assert_close(item["noisy"] - item["clean"], torch.full((5,), 0.1))
    torch.testing.assert_close(item["teacher"] - item["clean"], torch.full((5,), 0.2))
    assert item["length"] == 5


def test_padding_and_valid_lengths(tmp_path):
    clean = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    manifest = write_manifest(tmp_path, {"noisy": clean, "clean": clean})
    item = PairedAudioDataset(manifest, 16000, crop_samples=7)[0]
    assert item["length"] == 3
    assert torch.count_nonzero(item["clean"][3:]) == 0
    torch.testing.assert_close(item["clean"], item["noisy"])
    batch = collate_audio([item, {"noisy": torch.ones(5), "clean": torch.ones(5), "length": 5}])
    assert batch["noisy"].shape == (2, 7)
    assert batch["lengths"].tolist() == [3, 5]


def test_evaluation_crop_starts_at_zero(tmp_path):
    audio = np.arange(10, dtype=np.float32) / 10
    manifest = write_manifest(tmp_path, {"noisy": audio, "clean": audio})
    item = PairedAudioDataset(manifest, 16000, crop_samples=4, training=False)[0]
    torch.testing.assert_close(item["clean"], torch.from_numpy(audio[:4]))


@pytest.mark.parametrize("problem", ["length", "teacher_length", "rate", "stereo", "manifest_rate"])
def test_invalid_pairs_rejected_before_training(tmp_path, problem):
    arrays = {"noisy": np.ones(12), "clean": np.ones(12)}
    rates, row_rate = {}, 16000
    if problem == "length":
        arrays["clean"] = np.ones(11)
    elif problem == "teacher_length":
        arrays["teacher"] = np.ones(10)
    elif problem == "rate":
        rates["clean"] = 8000
    elif problem == "stereo":
        arrays["clean"] = np.ones((12, 2))
    elif problem == "manifest_rate":
        row_rate = 8000
    manifest = write_manifest(tmp_path, arrays, rates, row_rate)
    with pytest.raises(ValueError):
        PairedAudioDataset(manifest, 16000)


def test_nonfinite_samples_rejected_when_decoded(tmp_path):
    manifest = write_manifest(tmp_path, {"noisy": [0, float("nan"), 0], "clean": [0, 0, 0]})
    dataset = PairedAudioDataset(manifest, 16000)
    with pytest.raises(ValueError, match="non-finite"):
        dataset[0]


def test_mixed_teacher_batch_is_not_silently_dropped():
    plain = {"noisy": torch.ones(3), "clean": torch.ones(3)}
    teacher = {**plain, "teacher": torch.ones(3)}
    with pytest.raises(ValueError, match="consistently"):
        collate_audio([plain, teacher])


def test_missing_sample_rate_is_rejected(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"noisy": "x.wav", "clean": "y.wav"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sample_rate"):
        PairedAudioDataset(manifest, 16000)


def test_teacher_free_loading_never_accesses_teacher_path(tmp_path):
    manifest = write_manifest(tmp_path, {"noisy": np.ones(5), "clean": np.ones(5)})
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["teacher"] = "not-present.wav"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        PairedAudioDataset(manifest, 16000)
    item = PairedAudioDataset(manifest, 16000, load_teacher=False)[0]
    assert "teacher" not in item
    assert item["noisy"].shape == (5,)


def test_synthetic_smoke_is_deterministic_and_paired():
    dataset = SyntheticDataset(16000, 800, length=2)
    torch.testing.assert_close(dataset[0]["noisy"], dataset[0]["noisy"])
    assert not torch.equal(dataset[0]["noisy"], dataset[1]["noisy"])
    assert dataset[0]["noisy"].shape == dataset[0]["clean"].shape == (800,)
    assert dataset[0]["id"].startswith("synthetic-")
