"""Real filesystem backup/recovery checks; no model or training invocation."""
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from realtime_use import preparation_backup as backup


def output(tmp_path):
    root = tmp_path / 'local' / 'paired'
    root.mkdir(parents=True)
    (root / 'request.json').write_text('{"recipe": "fixed"}\n')
    sink = backup.PreparationBackup(root, tmp_path / 'persistent')
    sink.checkpoint('root', [root / 'request.json'], [root / 'request.json'])
    return root, sink


def chunk(root, name='000000', content=b'float-wave'):
    path = root / 'chunks' / 'train' / name
    path.mkdir(parents=True, exist_ok=True)
    (path / 'audio.wav').write_bytes(content)
    (path / 'plan.json').write_text('{"seed": 1}\n')
    (path / 'manifest.jsonl').write_text(json.dumps({'audio_sha256': backup._digest(path / 'audio.wav')}) + '\n')
    (path / 'render-progress.jsonl').write_text('{"complete": true}\n')
    (path / 'provenance.json').write_text('{"seconds": 1}\n')
    return path, [path / name for name in ('plan.json', 'manifest.jsonl', 'render-progress.jsonl')]


def test_checkpoint_restore_final_pending_and_ready_is_promoted_last(tmp_path, monkeypatch):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    sink.checkpoint('chunks/train/000000', [path], identities)
    (root / 'READY.pending.json').write_text('{"training_started": false}\n')
    final = sink.checkpoint('final', [root / 'READY.pending.json'], [root / 'READY.pending.json'])
    assert not (root / 'READY.json').exists()
    with tarfile.open(sink.backup_dir / final['archive']['path']) as archive:
        assert 'READY.json' in archive.getnames()
        assert 'READY.pending.json' not in archive.getnames()
    expected = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    expected[Path('READY.json')] = expected.pop(Path('READY.pending.json'))
    shutil.rmtree(root)
    linked = []
    original_link = backup.os.link
    def link(source, destination, **kwargs):
        linked.append(Path(destination).relative_to(root).as_posix())
        return original_link(source, destination, **kwargs)
    monkeypatch.setattr(backup.os, 'link', link)
    result = backup.restore(sink.backup_dir, root)
    assert result['ready_restored'] and result['training_started'] is False
    assert linked[-1] == 'READY.json'
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()} == expected
    again = backup.restore(sink.backup_dir, root)
    assert again['restored_files'] == 0 and again['reused_files'] == len(expected)


def test_semantic_identity_deduplicates_but_archive_is_reverified(tmp_path):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    first = sink.checkpoint('chunks/train/000000', [path], identities)
    (path / 'provenance.json').write_text('{"seconds": 999, "reused": 1}\n')
    assert sink.checkpoint('chunks/train/000000', [path], identities) == first
    assert len(list((sink.backup_dir / 'objects').glob('*.tar'))) == 2
    archive = sink.backup_dir / first['archive']['path']
    with archive.open('r+b') as handle:
        handle.seek(1024)
        handle.write(b'X')
    with pytest.raises(ValueError, match='integrity mismatch'):
        sink.checkpoint('chunks/train/000000', [path], identities)


def test_latest_unit_snapshot_and_stale_ready_are_handled(tmp_path):
    root, sink = output(tmp_path)
    path, identities = chunk(root, content=b'old-audio')
    first = sink.checkpoint('chunks/train/000000', [path], identities)
    (root / 'READY.json').write_text('{"ready": true}')
    sink.checkpoint('final', [root / 'READY.json'], [root / 'READY.json'])
    chunk(root, content=b'new-audio')
    latest = sink.checkpoint('chunks/train/000000', [path], identities)
    assert latest['receipt_sha256'] != first['receipt_sha256']
    shutil.rmtree(root)
    result = backup.restore(sink.backup_dir, root)
    assert not result['ready_restored'] and not (root / 'READY.json').exists()
    assert (path / 'audio.wav').read_bytes() == b'new-audio'


def test_interrupted_objects_and_receipts_are_not_commits(tmp_path):
    root, sink = output(tmp_path)
    (sink.backup_dir / 'objects' / 'interrupted.tar').write_bytes(b'incomplete')
    (sink.backup_dir / 'commits' / 'interrupted.json').write_text('{"status": "committed",')
    (sink.backup_dir / 'commits' / 'incomplete.json').write_text('{"status": "committed"}')
    (sink.backup_dir / 'commits' / 'uncommitted.json').write_text('{"status": "writing"}')
    shutil.rmtree(root)
    assert backup.restore(sink.backup_dir, root)['units'] == ['root']


def test_tampered_archive_blocks_all_restore_and_ready(tmp_path):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    receipt = sink.checkpoint('chunks/train/000000', [path], identities)
    (root / 'READY.json').write_text('{"ready": true}')
    sink.checkpoint('final', [root / 'READY.json'], [root / 'READY.json'])
    (sink.backup_dir / receipt['archive']['path']).write_bytes(b'truncated')
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='integrity mismatch'):
        backup.restore(sink.backup_dir, root)
    assert not root.exists()


def test_conflicting_output_and_wrong_root_are_rejected(tmp_path):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    sink.checkpoint('chunks/train/000000', [path], identities)
    (path / 'audio.wav').write_bytes(b'conflicting-existing-data')
    with pytest.raises(FileExistsError, match='conflicting existing file'):
        backup.restore(sink.backup_dir, root)
    assert (path / 'audio.wav').read_bytes() == b'conflicting-existing-data'
    with pytest.raises(ValueError, match='exact absolute output root'):
        backup.restore(sink.backup_dir, tmp_path / 'different')
    (root / 'request.json').write_text('{"recipe": "different"}')
    with pytest.raises(ValueError, match='different output root or preparation request'):
        sink.checkpoint('root', [root / 'request.json'], [root / 'request.json'])


def test_backup_rejects_symlinks_and_paths_outside_output(tmp_path):
    root, sink = output(tmp_path)
    outside = tmp_path / 'outside'
    outside.write_text('must stay outside')
    link = root / 'linked'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        sink.checkpoint('bad', [link], [root / 'request.json'])
    with pytest.raises(ValueError, match='outside output root'):
        sink.checkpoint('bad', [outside], [root / 'request.json'])


@pytest.mark.parametrize('kind', ['traversal', 'absolute', 'symlink', 'hardlink', 'fifo'])
def test_restore_rejects_unsafe_tar_members_even_with_valid_archive_hash(tmp_path, kind):
    root, sink = output(tmp_path)
    receipt = dict(sink._receipts[0])
    archive_path = sink.backup_dir / receipt['archive']['path']
    with tarfile.open(archive_path, mode='w') as archive:
        request = (root / 'request.json').read_bytes()
        info = tarfile.TarInfo('request.json'); info.size = len(request)
        archive.addfile(info, io.BytesIO(request))
        info = tarfile.TarInfo('../escape' if kind == 'traversal' else '/escape' if kind == 'absolute' else 'bad')
        if kind == 'symlink':
            info.type, info.linkname = tarfile.SYMTYPE, '../escape'
        elif kind == 'hardlink':
            info.type, info.linkname = tarfile.LNKTYPE, 'request.json'
        elif kind == 'fifo':
            info.type = tarfile.FIFOTYPE
        archive.addfile(info)
    receipt['archive'] = {'path': receipt['archive']['path'], 'bytes': archive_path.stat().st_size,
                          'sha256': backup._digest(archive_path)}
    receipt['receipt_sha256'] = backup._hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    next((sink.backup_dir / 'commits').glob('*.json')).write_bytes(backup._json(receipt))
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='Unconfined|forbidden'):
        backup.restore(sink.backup_dir, root)
    assert not (root.parent / 'escape').exists()
    assert not (root / 'request.json').exists()


def test_readback_failure_publishes_no_commit(tmp_path, monkeypatch):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    original = backup._digest
    def damaged(path):
        value = original(path)
        return '0' * 64 if path.suffix == '.tar' else value
    monkeypatch.setattr(backup, '_digest', damaged)
    with pytest.raises(ValueError, match='read-back integrity'):
        sink.checkpoint('chunks/train/000000', [path], identities)
    assert len(list((sink.backup_dir / 'commits').glob('*.json'))) == 1


def test_corrupted_committed_receipt_is_rejected(tmp_path):
    root, sink = output(tmp_path)
    path = next((sink.backup_dir / 'commits').glob('*.json'))
    receipt = json.loads(path.read_text())
    receipt['created_ns'] += 1  # Complete JSON, but not the committed descriptor.
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='receipt checksum mismatch'):
        backup.restore(sink.backup_dir, root)


def test_ready_cannot_be_saved_in_an_ordinary_unit(tmp_path):
    root, sink = output(tmp_path)
    (root / 'READY.json').write_text('{"ready": true}')
    with pytest.raises(ValueError, match='only appear in the final'):
        sink.checkpoint('root-with-ready', [root], [root / 'request.json'])


def test_restore_verifies_semantic_identity_inside_validated_archive(tmp_path):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    receipt = sink.checkpoint('chunks/train/000000', [path], identities)
    archive_path = sink.backup_dir / receipt['archive']['path']
    with tarfile.open(archive_path) as archive:
        members = [(item, archive.extractfile(item).read() if item.isfile() else None) for item in archive]
    with tarfile.open(archive_path, 'w') as archive:
        for item, content in members:
            if item.name.endswith('/plan.json'):
                content = b'{"seed": 999}\n'
                item.size = len(content)
            archive.addfile(item, io.BytesIO(content) if content is not None else None)
    # Even a self-consistent container hash cannot replace the separately
    # committed semantic file identity. This is an integrity check, not a signature.
    receipt['archive']['bytes'] = archive_path.stat().st_size
    receipt['archive']['sha256'] = backup._digest(archive_path)
    receipt['receipt_sha256'] = backup._hash_json({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    for marker in (sink.backup_dir / 'commits').glob('*.json'):
        if json.loads(marker.read_text())['unit'] == receipt['unit']:
            marker.write_bytes(backup._json(receipt))
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='Restored semantic identity mismatch'):
        backup.restore(sink.backup_dir, root)
    assert not (root / 'request.json').exists()
