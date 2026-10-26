"""Optional ISA-L gzip uses unchanged recovery/integrity; never trains."""
import builtins
import json
from pathlib import Path
import shutil

import pytest

from realtime_use import preparation_backup as backup
from realtime_use import prepare_release as release
from test_preparation_backup import chunk, output
from test_prepare_release import backup_receipts, fake_renderer, fixture_release


def require_isal():
    module = pytest.importorskip('isal.igzip')
    if backup.metadata.version('isal') != '1.8.0':
        pytest.skip('ISA-L checkpoint tests require the pinned optional isal==1.8.0')
    return module


def test_isal_uses_measured_level2_with_ordinary_gzip_restore(tmp_path, monkeypatch):
    module = require_isal()
    original, levels = module.IGzipFile, []
    def observed(**kwargs):
        levels.append(kwargs['compresslevel'])
        return original(**kwargs)
    monkeypatch.setattr(module, 'IGzipFile', observed)
    root, old = output(tmp_path)
    sink = backup.PreparationBackup(root, old.backup_dir, compression='gzip-isal')
    path, identity = chunk(root, content=b'exact lossless bytes' * 4096)
    sink.checkpoint('chunks/train/000000', [path], identity)
    assert levels == [2]
    shutil.rmtree(root)
    backup.restore(sink.backup_dir, root)
    assert (path / 'audio.wav').read_bytes() == b'exact lossless bytes' * 4096


def test_isal_and_legacy_archives_mix_reuse_and_restore_exactly(tmp_path, monkeypatch):
    module = require_isal()
    assert backup.validate_compression('gzip-isal') is module.IGzipFile
    root, original = output(tmp_path)  # Legacy uncompressed root receipt.
    stdlib = backup.PreparationBackup(root, original.backup_dir, compression='gzip')
    legacy, legacy_identity = chunk(root, name='000000', content=b'legacy gzip bytes' * 100)
    legacy_receipt = stdlib.checkpoint('chunks/train/000000', [legacy], legacy_identity)
    accelerated = backup.PreparationBackup(root, original.backup_dir, compression='gzip-isal')
    assert accelerated.checkpoint('chunks/train/000000', [legacy], legacy_identity) == legacy_receipt
    paths = [chunk(root, name=f'{index:06d}', content=bytes(range(256)) * (256 + index))
             for index in (1, 2)]
    with release._ChunkBackups(accelerated, True, 2) as checkpoints:
        for index, (path, identity) in enumerate(paths, 1):
            checkpoints.checkpoint(f'chunks/train/{index:06d}', [path], identity)
    current = backup.PreparationBackup(root, original.backup_dir, compression='gzip')
    for index, (path, identity) in enumerate(paths, 1):
        receipt = current.checkpoint(f'chunks/train/{index:06d}', [path], identity)
        archive = original.backup_dir / receipt['archive']['path']
        assert archive.name.endswith('.tar.gz')
        assert backup._digest(archive) == receipt['archive']['sha256']
        assert archive.stat().st_size == receipt['archive']['bytes']
        with archive.open('rb') as handle:
            header = handle.read(8)
        assert header[:4] == b'\x1f\x8b\x08\x00' and header[4:] == b'\0' * 4
    assert len(list((original.backup_dir / 'objects').iterdir())) == 4
    (root / 'READY.pending.json').write_text('{"training_started": false}\n')
    accelerated.checkpoint('final', [root / 'READY.pending.json'], [root / 'READY.pending.json'])
    expected = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    expected[Path('READY.json')] = expected.pop(Path('READY.pending.json'))
    shutil.rmtree(root)
    # Recovery must not select or import any compression implementation.
    def no_optional_backend(*args):
        pytest.fail('Restore tried to resolve an optional compression writer')
    monkeypatch.setattr(backup, 'validate_compression', no_optional_backend)
    assert backup.restore(original.backup_dir, root)['ready_restored'] is True
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()} == expected


def test_release_records_explicit_isal_choice_without_changing_recipe_or_audio(tmp_path, monkeypatch):
    require_isal()
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    destination, persistent = tmp_path / 'pairs', tmp_path / 'persistent'
    plan = release.release_plan(root, source, destination, chunk_size=1)
    result = release.execute_release(plan, backup_dir=persistent, backup_compression='gzip-isal',
                                     overlap_backups=True, backup_workers=2)
    request = (destination / 'request.json').read_bytes()
    assert 'backup_compression' not in json.loads(request)
    assert json.loads((destination / 'execution.json').read_text())['backup_compression'] == 'gzip-isal'
    audio = {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*.wav')}
    receipts = backup_receipts(persistent)
    assert all(item['archive']['path'].endswith('.tar.gz') for item in receipts)
    assert release.execute_release(plan, backup_dir=persistent, backup_compression='gzip') == result
    assert (destination / 'request.json').read_bytes() == request
    assert len(backup_receipts(persistent)) == len(receipts)
    shutil.rmtree(destination)
    assert backup.restore(persistent, destination)['ready_restored'] is True
    assert {relative: (destination / relative).read_bytes() for relative in audio} == audio


@pytest.mark.parametrize('failure', ['missing', 'wrong_version', 'unloadable'])
def test_unavailable_isal_fails_before_any_output_or_ready_mutation(tmp_path, monkeypatch, failure):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    destination, persistent = tmp_path / 'pairs', tmp_path / 'persistent'
    plan = release.release_plan(root, source, destination)
    release.execute_release(plan)
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    original_version, original_import = backup.metadata.version, builtins.__import__

    def version(name):
        if name != 'isal':
            return original_version(name)
        if failure == 'missing':
            raise backup.metadata.PackageNotFoundError('isal')
        return '0.0.0' if failure == 'wrong_version' else '1.8.0'

    def imported(name, *args, **kwargs):
        if name == 'isal.igzip' and failure == 'unloadable':
            raise OSError('optional compiled library could not load')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(backup.metadata, 'version', version)
    monkeypatch.setattr(builtins, '__import__', imported)
    with pytest.raises(RuntimeError, match='requires isal==1.8.0'):
        backup.PreparationBackup(destination, persistent, compression='gzip-isal')
    with pytest.raises(RuntimeError, match='requires isal==1.8.0'):
        release.execute_release(plan, backup_dir=persistent, backup_compression='gzip-isal')
    assert not persistent.exists()
    assert {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()} == before
    new = tmp_path / 'new-output'
    new_plan = release.release_plan(root, source, new)
    with pytest.raises(RuntimeError, match='requires isal==1.8.0'):
        release.execute_release(new_plan, backup_dir=persistent, backup_compression='gzip-isal')
    assert not new.exists() and not persistent.exists()
    # The default implementations never depend on ISA-L or silently select it.
    assert backup.validate_compression('none') is None
    assert backup.validate_compression('gzip') is backup.gzip.GzipFile
