"""Metadata-only source recovery retains generated pairs and never trains."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest
import soundfile as sf

from realtime_use import prepare_release as release
from realtime_use.preparation_backup import PreparationBackup, restore
from test_prepare_release import backup_receipts, fake_renderer, fixture_release


def stereo_source(root):
    index = root / 'release/speech_train.jsonl'
    rows = list(release._rows(index))
    shard = root / rows[0]['shard']
    with tarfile.open(shard) as archive:
        contents = [(member.name, archive.extractfile(member).read()) for member in archive]
    wave, rate = sf.read(io.BytesIO(contents[0][1]), dtype='float32')
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([wave, wave * .25], axis=1), rate, format='WAV', subtype='FLOAT')
    raw = buffer.getvalue()
    contents[0] = contents[0][0], raw
    rows[0].update(channels=2, sha256=hashlib.sha256(raw).hexdigest())
    with tarfile.open(shard, 'w') as archive:
        for name, data in contents:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    payload = ''.join(json.dumps(row) + '\n' for row in rows)
    index.write_text(payload)
    (shard.parent / 'train.jsonl').write_text(payload)
    for path in [*root.glob('sources/*/SOURCE_READY.json'), root / 'release/RELEASE.json']:
        report = json.loads(path.read_text())
        for entry in report['artifacts']:
            file = root / entry['path']
            entry.update(bytes=file.stat().st_size, sha256=release.prep.sha256(file))
        path.write_text(json.dumps(report))


def paired_fixture_renderer(monkeypatch):
    _, render = fake_renderer(monkeypatch)
    def paired(plan, **kwargs):
        manifest = render(plan, **kwargs)
        rows = list(release._rows(manifest))
        for index, row in enumerate(rows):
            for role in ('clean', 'noisy'):
                path = Path(plan['output_dir']) / role / f'{index}.wav'
                path.parent.mkdir(exist_ok=True)
                if Path(row[role]) != path:
                    path.write_bytes(Path(row[role]).read_bytes())
                row[role] = str(path)
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (manifest.parent / 'render-progress.jsonl').write_bytes(manifest.read_bytes())
        return manifest
    monkeypatch.setattr(release.prep, 'execute_preparation', paired)


@pytest.mark.parametrize('overlap', [False, True])
def test_metadata_mode_retains_all_metadata_and_pairs_then_regenerates_exact_sources(tmp_path, monkeypatch, overlap):
    root, source = fixture_release(tmp_path, monkeypatch)
    stereo_source(root)
    paired_fixture_renderer(monkeypatch)
    output, backup = tmp_path / 'out', tmp_path / 'backup'
    plan = release.release_plan(root, source, output, chunk_size=1)
    result = release.execute_release(plan, backup_dir=backup, materialized_backup_mode='metadata',
                                     overlap_materialized_backups=overlap)
    request = (output / 'request.json').read_bytes()
    assert 'materialized_backup_mode' not in json.loads(request)
    assert json.loads((output / 'execution.json').read_text())['materialized_backup_mode'] == 'metadata'
    originals = {p.relative_to(output): p.read_bytes() for p in (output / 'materialized').rglob('*') if p.is_file()}
    pairs = {p.relative_to(output): p.read_bytes() for p in (output / 'chunks').rglob('*.wav')}
    assert pairs and any('/verification/' in str(path) for path in originals)
    receipts = backup_receipts(backup)
    materialized = [item for item in receipts if item['unit'].startswith('materialized-metadata/')]
    assert len(materialized) == len(plan['groups'])
    assert not any(item['unit'].startswith('materialized/') for item in receipts)
    archived = set()
    for item in materialized:
        with tarfile.open(backup / item['archive']['path']) as archive:
            names = {member.name for member in archive if member.isfile()}
        assert not any('/audio/' in name for name in names)
        archived.update(names)
    metadata = {path: data for path, data in originals.items() if 'audio' not in path.parts}
    assert {str(path) for path in metadata} <= archived
    final = next(item for item in receipts if item['unit'] == 'final')
    assert {item['unit'] for item in materialized} <= set(final['dependencies'])
    shutil.rmtree(output)
    assert restore(backup, output)['ready_restored'] is True
    assert not list((output / 'materialized').glob('*/audio/*'))
    assert all((output / path).read_bytes() == data for path, data in metadata.items())
    assert all((output / path).read_bytes() == data for path, data in pairs.items())
    assert json.loads((output / 'READY.json').read_text()) == result
    # The retained source root/release hash and partitioned original records
    # reproduce every intermediate byte, including first-channel WAV sidecars.
    assert json.loads(request)['root'] == str(root)
    assert json.loads(request)['release_sha256'] == release.prep.sha256(root / 'release/RELEASE.json')
    for key, group in plan['groups'].items():
        source_plan = release.bridge.make_plan(root, output / 'indices' / f'{key}.jsonl',
            output / 'materialized' / key, group['role'], channel_policy='first',
            wind_id_scope=group.get('wind_id_scope', 'global'))
        release.bridge.materialize(source_plan)
    assert originals == {p.relative_to(output): p.read_bytes()
                         for p in (output / 'materialized').rglob('*') if p.is_file()}
    assert release.execute_release(plan, backup_dir=backup, materialized_backup_mode='metadata') == result
    assert (output / 'request.json').read_bytes() == request


def unit_fixture(tmp_path):
    root, persistent = tmp_path / 'output', tmp_path / 'backup'
    group = root / 'materialized/group'
    (group / 'audio').mkdir(parents=True)
    (root / 'request.json').write_text('{"recipe": "fixture"}')
    for name, data in {'request.json': '{}', 'records.jsonl': '{}\n', 'source.scp': 'fixture',
                       'text': 'words', 'utt2spk': 'speaker', 'materialization.json': '{"reused":0}'}.items():
        (group / name).write_text(data)
    (group / 'audio/original.wav').write_bytes(b'extracted fixture bytes')
    backup = PreparationBackup(root, persistent)
    backup.checkpoint('root', [root / 'request.json'], [root / 'request.json'])
    return root, group, backup


@pytest.mark.parametrize('last_mode', ['metadata', 'full'])
def test_mixed_modes_restore_only_latest_scope_and_reactivation_reuses_archive(tmp_path, last_mode):
    root, group, backup = unit_fixture(tmp_path)
    full = release._checkpoint_materialized(backup, root, 'group', 'full')
    (group / 'materialization.json').write_text('{"reused":1}')
    metadata = release._checkpoint_materialized(backup, root, 'group', 'metadata')
    assert full['unit'] != metadata['unit'] and full['archive']['path'] != metadata['archive']['path']
    objects = set((backup.backup_dir / 'objects').iterdir())
    if last_mode == 'full':
        active = release._checkpoint_materialized(backup, root, 'group', 'full')
        assert active['receipt_sha256'] != full['receipt_sha256']
        assert active['created_ns'] > metadata['created_ns']
        assert active['archive'] == full['archive']
        assert set((backup.backup_dir / 'objects').iterdir()) == objects
    (root / 'READY.pending.json').write_text('{"ready": true}')
    backup.checkpoint('final', [root / 'READY.pending.json'], [root / 'READY.pending.json'])
    shutil.rmtree(root)
    report = restore(backup.backup_dir, root)
    assert report['ready_restored'] is True
    assert (group / 'audio/original.wav').exists() == (last_mode == 'full')
    assert json.loads((group / 'materialization.json').read_text())['reused'] == (last_mode == 'metadata')
    assert ('materialized/group' in report['units']) == (last_mode == 'full')
    assert ('materialized-metadata/group' in report['units']) == (last_mode == 'metadata')


def test_superseded_scope_is_still_verified_and_corrupt_archive_cannot_reactivate(tmp_path):
    root, group, backup = unit_fixture(tmp_path)
    full = release._checkpoint_materialized(backup, root, 'group', 'full')
    release._checkpoint_materialized(backup, root, 'group', 'metadata')
    before = set((backup.backup_dir / 'commits').iterdir())
    archive = backup.backup_dir / full['archive']['path']
    archive.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity mismatch'):
        release._checkpoint_materialized(backup, root, 'group', 'full')
    assert set((backup.backup_dir / 'commits').iterdir()) == before
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='integrity mismatch'):
        restore(backup.backup_dir, root)
    assert not root.exists()


@pytest.mark.parametrize('mode', ['unknown', 'metadata'])
def test_invalid_mode_or_missing_backup_fails_before_mutation(tmp_path, monkeypatch, mode):
    root, source = fixture_release(tmp_path, monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output)
    with pytest.raises(ValueError, match='full or metadata|backup directory'):
        release.execute_release(plan, materialized_backup_mode=mode)
    assert not output.exists()
