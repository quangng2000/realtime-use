"""Bounded concurrent durable checkpoints use real archives, never training."""
import json
import os
import shutil
import threading

import pytest

from realtime_use import preparation_backup as backup
from realtime_use import prepare_release as release
from test_preparation_backup import chunk, output
from test_prepare_release import backup_receipts, fake_renderer, fixture_release


@pytest.mark.parametrize('compression', ['none', 'gzip'])
def test_two_units_checkpoint_concurrently_then_restore_all_final_dependencies(tmp_path, monkeypatch, compression):
    root, sink = output(tmp_path)
    sink = backup.PreparationBackup(root, sink.backup_dir, compression=compression)
    paths = [chunk(root, name=f'{index:06d}', content=bytes([index]) * 1024)
             for index in range(2)]
    original = sink._checkpoint
    gate, both_started = threading.Event(), threading.Event()
    mutex, started = threading.Lock(), []

    def blocked(unit, paths, identity_paths):
        with mutex:
            started.append(unit)
            if len(started) == 2:
                both_started.set()
        assert gate.wait(5)
        return original(unit, paths, identity_paths)

    monkeypatch.setattr(sink, '_checkpoint', blocked)
    with release._ChunkBackups(sink, True, 2) as queue:
        for index, (path, identity) in enumerate(paths):
            queue.checkpoint(f'chunks/train/{index:06d}', [path], identity)
        try:
            assert both_started.wait(5), 'Checkpoint payloads were serialized'
            assert len(queue.pending) == 2
            with pytest.raises(RuntimeError, match='already active'):
                sink.checkpoint('chunks/train/000000', [paths[0][0]], paths[0][1])
            with pytest.raises(RuntimeError, match='all other checkpoints'):
                sink.checkpoint('final', [], [])
        finally:
            gate.set()
    assert not sink._active_units
    monkeypatch.setattr(sink, '_checkpoint', original)
    (root / 'READY.pending.json').write_text('{"ready": true}\n')
    final = sink.checkpoint('final', [root / 'READY.pending.json'], [root / 'READY.pending.json'])
    assert set(final['dependencies']) == {'root', 'chunks/train/000000', 'chunks/train/000001'}
    shutil.rmtree(root)
    assert backup.restore(sink.backup_dir, root)['ready_restored'] is True
    for index, (path, _) in enumerate(paths):
        assert (path / 'audio.wav').read_bytes() == bytes([index]) * 1024


def test_second_future_failure_is_seen_without_waiting_for_first_and_all_join(tmp_path):
    first_started, second_failed, release_first = (threading.Event() for _ in range(3))
    first_finished = threading.Event()

    class FailingBackup:
        def checkpoint(self, unit, *args):
            if unit == 'first':
                first_started.set()
                assert release_first.wait(5)
                first_finished.set()
            else:
                assert first_started.wait(5)
                second_failed.set()
                raise OSError('second checkpoint failed')

    with pytest.raises(OSError, match='second checkpoint failed'):
        with release._ChunkBackups(FailingBackup(), True, 2) as queue:
            queue.checkpoint('first', [], [])
            queue.checkpoint('second', [], [])
            try:
                assert second_failed.wait(5)
                # result() makes the failure's completion deterministic; keep
                # it queued so check() must inspect beyond the blocked first.
                with pytest.raises(OSError, match='second checkpoint failed'):
                    queue.pending[1].result(timeout=5)
                assert not first_finished.is_set()
                queue.check()
                pytest.fail('A completed second failure was not observed')
            finally:
                release_first.set()
    assert first_finished.is_set()
    assert not any(t.name.startswith('chunk-backup') for t in threading.enumerate())


def test_release_two_backups_overlap_third_render_and_keep_recipe_identity(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, render = fake_renderer(monkeypatch)
    destination, persistent = tmp_path / 'pairs', tmp_path / 'persistent'
    plan = release.release_plan(root, source, destination, chunk_size=1)
    original = backup.PreparationBackup._checkpoint
    mutex, first_started = threading.Lock(), []
    both_started, third_rendered = threading.Event(), threading.Event()

    def checkpoint(self, unit, paths, identity_paths):
        if unit in ('chunks/validation/000000', 'chunks/validation/000001'):
            with mutex:
                first_started.append(threading.get_ident())
                if len(first_started) == 2:
                    both_started.set()
            assert third_rendered.wait(5)
        return original(self, unit, paths, identity_paths)

    def local_render(plan, **kwargs):
        if len(calls) == 2:
            assert both_started.wait(5)
            third_rendered.set()
        return render(plan, **kwargs)

    monkeypatch.setattr(backup.PreparationBackup, '_checkpoint', checkpoint)
    monkeypatch.setattr(release.prep, 'execute_preparation', local_render)
    result = release.execute_release(plan, backup_dir=persistent, overlap_backups=True, backup_workers=2)
    assert result['training_started'] is False
    assert len(set(first_started)) == 2
    request = (destination / 'request.json').read_bytes()
    assert 'backup_workers' not in json.loads(request)
    assert json.loads((destination / 'execution.json').read_text())['backup_workers'] == 2
    final = next(item for item in backup_receipts(persistent) if item['unit'] == 'final')
    assert len([unit for unit in final['dependencies'] if unit.startswith('chunks/')]) == 6
    monkeypatch.setattr(backup.PreparationBackup, '_checkpoint', original)
    assert release.execute_release(plan, backup_dir=persistent) == result
    assert (destination / 'request.json').read_bytes() == request


def test_failed_receipt_flush_is_retried_before_archive_reuse(tmp_path, monkeypatch):
    root, sink = output(tmp_path)
    path, identities = chunk(root)
    original_sync = backup.os.fsync
    calls = 0

    def fail_receipt(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('receipt flush failed')
        return original_sync(fd)

    monkeypatch.setattr(backup.os, 'fsync', fail_receipt)
    with pytest.raises(OSError, match='receipt flush failed'):
        sink.checkpoint('chunks/train/000000', [path], identities)
    assert len(sink._receipts) == 1
    receipts = {json.loads(p.read_text())['unit']: p for p in (sink.backup_dir / 'commits').iterdir()}
    receipt_path = receipts['chunks/train/000000']
    receipt = json.loads(receipt_path.read_text())
    archive = sink.backup_dir / receipt['archive']['path']

    def unavailable(fd):
        raise OSError('persistent flush still unavailable')

    monkeypatch.setattr(backup.os, 'fsync', unavailable)
    with pytest.raises(OSError, match='flush still unavailable'):
        backup.PreparationBackup(root, sink.backup_dir)
    synced = []

    def record_sync(fd):
        synced.append(os.fstat(fd).st_ino)
        return original_sync(fd)

    monkeypatch.setattr(backup.os, 'fsync', record_sync)
    resumed = backup.PreparationBackup(root, sink.backup_dir)
    assert receipt_path.stat().st_ino in synced
    assert resumed.checkpoint('chunks/train/000000', [path], identities) == receipt
    assert archive.stat().st_ino in synced
    assert len(list((sink.backup_dir / 'objects').iterdir())) == 2
    synced.clear()
    shutil.rmtree(root)
    backup.restore(sink.backup_dir, root)
    assert receipt_path.stat().st_ino in synced and archive.stat().st_ino in synced


@pytest.mark.parametrize('options', [
    {'backup_workers': 0}, {'backup_workers': 3}, {'backup_workers': True},
    {'backup_workers': 1.5}, {'backup_workers': 2},
])
def test_invalid_backup_worker_options_are_read_only(tmp_path, monkeypatch, options):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    destination = tmp_path / 'out'
    plan = release.release_plan(root, source, destination)
    release.execute_release(plan)
    ready = (destination / 'READY.json').read_bytes()
    with pytest.raises(ValueError, match='1 or 2|requires overlap'):
        release.execute_release(plan, **options)
    assert (destination / 'READY.json').read_bytes() == ready
