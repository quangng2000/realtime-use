"""Lossless stdlib gzip checkpoint transport; no audio processing or training."""
import gzip
import json
from pathlib import Path
import shutil

import pytest

from realtime_use import preparation_backup as backup
from realtime_use import prepare_release as release
from test_preparation_backup import chunk
from test_prepare_release import backup_receipts, fake_renderer, fixture_release


def sink_fixture(tmp_path, compression):
    root = tmp_path / 'local'
    root.mkdir()
    (root / 'request.json').write_text('{"recipe": "fixed"}\n')
    sink = backup.PreparationBackup(root, tmp_path / 'backup', compression=compression)
    sink.checkpoint('root', [root / 'request.json'], [root / 'request.json'])
    return root, sink


@pytest.mark.parametrize('initial', ['none', 'gzip'])
def test_mixed_and_gzip_archives_restore_exact_bytes_and_final_ready(tmp_path, initial):
    root, original = sink_fixture(tmp_path, initial)
    sink = backup.PreparationBackup(root, original.backup_dir, compression='gzip')
    path, identities = chunk(root, content=bytes(range(256)) * 4096)
    receipt = sink.checkpoint('chunks/train/000000', [path], identities)
    archive = sink.backup_dir / receipt['archive']['path']
    compressed = archive.read_bytes()
    assert archive.name.endswith('.tar.gz')
    assert receipt['archive']['bytes'] == len(compressed)
    assert receipt['archive']['sha256'] == backup._digest(archive)
    assert len(gzip.decompress(compressed)) > len(compressed)
    assert compressed[:4] == b'\x1f\x8b\x08\x00'  # No filename field.
    assert compressed[4:8] == b'\0' * 4  # Stable gzip mtime.
    (root / 'READY.pending.json').write_text('{"training_started": false}\n')
    sink.checkpoint('final', [root / 'READY.pending.json'], [root / 'READY.pending.json'])
    expected = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    expected[Path('READY.json')] = expected.pop(Path('READY.pending.json'))
    shutil.rmtree(root)
    assert backup.restore(sink.backup_dir, root)['ready_restored'] is True
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()} == expected


@pytest.mark.parametrize('initial,following', [('none', 'gzip'), ('gzip', 'none')])
def test_compression_change_reuses_existing_verified_receipt(tmp_path, initial, following):
    root, sink = sink_fixture(tmp_path, initial)
    path, identities = chunk(root)
    first = sink.checkpoint('chunks/train/000000', [path], identities)
    other = backup.PreparationBackup(root, sink.backup_dir, compression=following)
    assert other.checkpoint('chunks/train/000000', [path], identities) == first
    assert len(list((sink.backup_dir / 'objects').iterdir())) == 2


def test_gzip_output_is_reproducible_and_corruption_blocks_reuse_and_restore(tmp_path):
    root, sink = sink_fixture(tmp_path, 'gzip')
    path, identities = chunk(root, content=b'lossless fixture' * 10000)
    first = sink.checkpoint('chunks/train/000000', [path], identities)
    second_sink = backup.PreparationBackup(root, tmp_path / 'other', compression='gzip')
    second = second_sink.checkpoint('chunks/train/000000', [path], identities)
    assert first['archive']['sha256'] == second['archive']['sha256']
    archive = sink.backup_dir / first['archive']['path']
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ValueError, match='integrity mismatch'):
        sink.checkpoint('chunks/train/000000', [path], identities)
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='integrity mismatch'):
        backup.restore(sink.backup_dir, root)
    assert not root.exists()


@pytest.mark.parametrize('compression,wrong_suffix', [('gzip', '.tar'), ('none', '.tar.gz')])
def test_descriptor_format_must_match_archive_bytes(tmp_path, compression, wrong_suffix):
    root, sink = sink_fixture(tmp_path, compression)
    receipt = sink._receipts[0]
    old = sink.backup_dir / receipt['archive']['path']
    new = old.with_name('mismatched' + wrong_suffix)
    old.rename(new)
    receipt['archive']['path'] = str(new.relative_to(sink.backup_dir))
    receipt['receipt_sha256'] = backup._hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    next((sink.backup_dir / 'commits').iterdir()).write_bytes(backup._json(receipt))
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='differs from'):
        backup.restore(sink.backup_dir, root)
    assert not (root / 'request.json').exists()


def test_gzip_failure_syncs_no_receipt_and_invalid_option_is_read_only(tmp_path, monkeypatch):
    root, sink = sink_fixture(tmp_path, 'gzip')
    path, identities = chunk(root)
    with pytest.raises(ValueError, match='none or gzip'):
        backup.PreparationBackup(root, tmp_path / 'invalid', compression='zip')
    assert not (tmp_path / 'invalid').exists()
    def fail_sync(fd):
        raise OSError('gzip archive sync failed')
    monkeypatch.setattr(backup.os, 'fsync', fail_sync)
    with pytest.raises(OSError, match='gzip archive sync failed'):
        sink.checkpoint('chunks/train/000000', [path], identities)
    assert len(list((sink.backup_dir / 'commits').iterdir())) == 1


def test_release_gzip_option_is_runtime_only_and_combines_with_overlap(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    output, destination = tmp_path / 'out', tmp_path / 'persistent'
    plan = release.release_plan(root, source, output, chunk_size=1)
    result = release.execute_release(plan, backup_dir=destination,
                                     backup_compression='gzip', overlap_backups=True)
    request = (output / 'request.json').read_bytes()
    assert 'backup_compression' not in json.loads(request)
    assert json.loads((output / 'execution.json').read_text())['backup_compression'] == 'gzip'
    receipts = backup_receipts(destination)
    assert all(item['archive']['path'].endswith('.tar.gz') for item in receipts)
    assert release.execute_release(plan, backup_dir=destination, backup_compression='none') == result
    assert (output / 'request.json').read_bytes() == request
    assert len(backup_receipts(destination)) == len(receipts)
    ready = (output / 'READY.json').read_bytes()
    for options in ({'backup_compression': 'zip'}, {'backup_compression': 'gzip'}):
        with pytest.raises(ValueError, match='none or gzip|backup directory'):
            release.execute_release(plan, **options)
        assert (output / 'READY.json').read_bytes() == ready
    shutil.rmtree(output)
    assert backup.restore(destination, output)['ready_restored'] is True
    assert (output / 'READY.json').read_bytes() == ready
