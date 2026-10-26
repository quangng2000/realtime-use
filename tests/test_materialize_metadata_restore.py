"""Regenerate omitted extracted audio from verified sources; no training."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pytest
import soundfile as sf

from realtime_use import materialize_sources as materializer
from realtime_use.preparation_backup import PreparationBackup, restore
from test_materialize_sources import mp3_pool, pool, stereo_pool


def metadata_files(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*')
            if p.is_file() and 'audio' not in p.relative_to(directory).parts}


@pytest.mark.parametrize('kind', ['mono-wav', 'stereo-wav', 'mono-mp3', 'stereo-mp3'])
def test_metadata_only_restore_reextracts_exact_bytes_samples_and_inventory(tmp_path, kind):
    if kind == 'mono-wav':
        root, manifest, _ = pool(tmp_path)
        role = 'speech'
    elif kind == 'stereo-wav':
        root, manifest, _, expected = stereo_pool(tmp_path)
        role = 'speech'
    else:
        channels = 2 if kind == 'stereo-mp3' else 1
        root, manifest, _, _, expected = mp3_pool(tmp_path, channels)
        role = 'noise'
    output = tmp_path / 'materialized'
    plan = materializer.make_plan(root, manifest, output, role, channel_policy='first')
    initial = materializer.materialize(plan)
    first_second = int(time.time())
    before = metadata_files(output)
    audio = {p.name: (materializer.file_sha256(p), sf.read(p, dtype='float32', always_2d=True)[0])
             for p in (output / 'audio').iterdir()}
    if kind.startswith('stereo'):
        for name, (_, samples) in audio.items():
            np.testing.assert_array_equal(samples, expected[:, :1])
            with sf.SoundFile(output / 'audio' / name) as decoded:
                assert 'PEAK :' not in decoded.extra_info
    sink = PreparationBackup(output, tmp_path / 'persistent')
    paths = [p for p in output.iterdir() if p.name != 'audio']
    identities = [output / 'request.json', output / 'records.jsonl',
                  *sorted((output / 'verification').glob('*.json'))]
    sink.checkpoint('materialized-metadata/fixture', paths, identities)
    shutil.rmtree(output)
    restore(sink.backup_dir, output)
    assert not (output / 'audio').exists()
    assert metadata_files(output) == before
    # libsndfile's old PEAK timestamp came from C wall-clock time, so a real
    # second boundary is necessary to expose this regression (no mocked clock).
    time.sleep(1.05)
    assert int(time.time()) > first_second
    result = materializer.materialize(materializer.make_plan(
        root, manifest, output, role, channel_policy='first'))
    assert result['extracted'] == initial['extracted'] and result['reused'] == 0
    assert metadata_files(output) == before
    for name, (sha256, samples) in audio.items():
        regenerated = output / 'audio' / name
        assert materializer.file_sha256(regenerated) == sha256
        np.testing.assert_array_equal(sf.read(regenerated, dtype='float32', always_2d=True)[0], samples)


def test_missing_legacy_normalized_audio_cannot_silently_change_saved_hash(tmp_path, monkeypatch):
    root, manifest, _, _ = stereo_pool(tmp_path)
    output = tmp_path / 'materialized'
    plan = materializer.make_plan(root, manifest, output, 'speech', channel_policy='first')
    # Generate the historical library-default container, including its PEAK
    # timestamp. Existing intact files must remain valid under the newer code.
    with monkeypatch.context() as legacy:
        legacy.setattr(materializer, '_disable_peak_chunk', lambda destination: None)
        materializer.materialize(plan)
    resumed = materializer.materialize(materializer.make_plan(
        root, manifest, output, 'speech', channel_policy='first'))
    assert resumed['reused'] == 1
    before = metadata_files(output)
    shutil.rmtree(output / 'audio')
    with pytest.raises(ValueError, match='Recreated first-channel audio differs from saved verification'):
        materializer.materialize(materializer.make_plan(
            root, manifest, output, 'speech', channel_policy='first'))
    assert metadata_files(output) == before
    assert not list((output / 'audio').iterdir())


def test_changed_verification_is_not_overwritten_when_audio_is_missing(tmp_path):
    root, manifest, _, _ = stereo_pool(tmp_path)
    output = tmp_path / 'materialized'
    materializer.materialize(materializer.make_plan(root, manifest, output, 'speech', channel_policy='first'))
    sidecar = next((output / 'verification').iterdir())
    metadata = json.loads(sidecar.read_text())
    metadata['source_sha256'] = '0' * 64
    sidecar.write_text(json.dumps(metadata))
    before = metadata_files(output)
    shutil.rmtree(output / 'audio')
    with pytest.raises(ValueError, match='verification identity mismatch'):
        materializer.materialize(materializer.make_plan(
            root, manifest, output, 'speech', channel_policy='first'))
    assert metadata_files(output) == before
    assert not list((output / 'audio').iterdir())
