"""Bounded checkpoint overlap uses generated audio and real archives, never training."""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from realtime_use import prepare_release as release
from realtime_use.preparation_backup import PreparationBackup, restore
from test_prepare_release import backup_receipts, fake_renderer, fixture_release


def test_real_checkpoints_overlap_next_render_and_join_before_final(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, render = fake_renderer(monkeypatch)
    output, backup = tmp_path / 'out', tmp_path / 'backup'
    plan = release.release_plan(root, source, output, chunk_size=1)
    checkpoint = PreparationBackup.checkpoint
    first_started, second_rendered = threading.Event(), threading.Event()
    final_chunk_started, allow_final_chunk = threading.Event(), threading.Event()
    chunk_threads, committed = [], []
    owner = threading.get_ident()

    def delayed_checkpoint(self, unit, paths, identity_paths):
        if unit.startswith('chunks/'):
            chunk_threads.append(threading.get_ident())
            if unit == 'chunks/validation/000000':
                first_started.set()
                assert second_rendered.wait(5), 'Next local render did not overlap backup'
            if unit == 'chunks/train/000002':
                final_chunk_started.set()
                assert allow_final_chunk.wait(5)
        else:
            assert threading.get_ident() == owner
        if unit == 'final':
            assert len(committed) == 6
            assert not (output / 'READY.json').exists()
        result = checkpoint(self, unit, paths, identity_paths)
        if unit.startswith('chunks/'):
            committed.append(unit)
        return result

    def local_render(plan, **kwargs):
        if len(calls) == 1:
            assert first_started.wait(5)
            second_rendered.set()
        return render(plan, **kwargs)

    def release_final_checkpoint():
        assert final_chunk_started.wait(5)
        assert not (output / 'READY.json').exists()
        assert not (output / 'READY.pending.json').exists()
        assert not (output / 'manifests').exists()
        allow_final_chunk.set()

    monkeypatch.setattr(PreparationBackup, 'checkpoint', delayed_checkpoint)
    monkeypatch.setattr(release.prep, 'execute_preparation', local_render)
    monitor = threading.Thread(target=release_final_checkpoint)
    monitor.start()
    result = release.execute_release(plan, backup_dir=backup, overlap_backups=True)
    monitor.join(5)
    assert not monitor.is_alive()
    assert result['training_pairs_ready'] is True and result['training_started'] is False
    assert len(set(chunk_threads)) == 1 and owner not in chunk_threads
    receipts = backup_receipts(backup)
    final = next(item for item in receipts if item['unit'] == 'final')
    assert len([unit for unit in final['dependencies'] if unit.startswith('chunks/')]) == 6
    request = (output / 'request.json').read_bytes()
    assert 'overlap_backups' not in json.loads(request)
    assert json.loads((output / 'execution.json').read_text())['overlap_backups'] is True

    # Switching back to synchronous checkpoints reuses the same verified objects.
    monkeypatch.setattr(PreparationBackup, 'checkpoint', checkpoint)
    monkeypatch.setattr(release.prep, 'execute_preparation', render)
    assert release.execute_release(plan, backup_dir=backup) == result
    assert len(backup_receipts(backup)) == len(receipts)
    assert (output / 'request.json').read_bytes() == request
    shutil.rmtree(output)
    restore(backup, output)
    assert json.loads((output / 'READY.json').read_text()) == result


def test_failed_async_backup_stops_before_third_chunk_and_blocks_final(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, render = fake_renderer(monkeypatch)
    output, backup = tmp_path / 'out', tmp_path / 'backup'
    plan = release.release_plan(root, source, output, chunk_size=1)
    checkpoint = PreparationBackup.checkpoint
    first_started, second_rendered, failed = (threading.Event() for _ in range(3))

    def fail_checkpoint(self, unit, paths, identity_paths):
        if unit == 'chunks/validation/000000':
            first_started.set()
            assert second_rendered.wait(5)
            failed.set()
            raise OSError('persistent checkpoint unavailable')
        return checkpoint(self, unit, paths, identity_paths)

    def local_render(plan, **kwargs):
        if len(calls) == 1:
            assert first_started.wait(5)
            second_rendered.set()
            assert failed.wait(5)
        return render(plan, **kwargs)

    monkeypatch.setattr(PreparationBackup, 'checkpoint', fail_checkpoint)
    monkeypatch.setattr(release.prep, 'execute_preparation', local_render)
    with pytest.raises(OSError, match='persistent checkpoint unavailable'):
        release.execute_release(plan, backup_dir=backup, overlap_backups=True)
    assert len(calls) == 2
    assert not (output / 'READY.json').exists()
    assert not (output / 'manifests').exists()
    assert 'final' not in {item['unit'] for item in backup_receipts(backup)}
    assert not any(t.name.startswith('chunk-backup') for t in threading.enumerate())


def test_render_failure_joins_pending_backup_and_preserves_recovery(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, render = fake_renderer(monkeypatch)
    output, backup = tmp_path / 'out', tmp_path / 'backup'
    plan = release.release_plan(root, source, output, chunk_size=1)
    checkpoint = PreparationBackup.checkpoint
    started, render_failed, committed = (threading.Event() for _ in range(3))

    def pending_checkpoint(self, unit, paths, identity_paths):
        if unit == 'chunks/validation/000000':
            started.set()
            assert render_failed.wait(5)
            result = checkpoint(self, unit, paths, identity_paths)
            committed.set()
            return result
        return checkpoint(self, unit, paths, identity_paths)

    def fail_render(plan, **kwargs):
        if len(calls) == 1:
            assert started.wait(5)
            render_failed.set()
            raise ValueError('local render failed')
        return render(plan, **kwargs)

    monkeypatch.setattr(PreparationBackup, 'checkpoint', pending_checkpoint)
    monkeypatch.setattr(release.prep, 'execute_preparation', fail_render)
    with pytest.raises(ValueError, match='local render failed'):
        release.execute_release(plan, backup_dir=backup, overlap_backups=True)
    assert committed.is_set(), 'Failure returned before the pending backup finished'
    assert not any(t.name.startswith('chunk-backup') for t in threading.enumerate())
    shutil.rmtree(output)
    restore(backup, output)
    assert (output / 'chunks/validation/000000/manifest.jsonl').is_file()
    assert not (output / 'READY.json').exists()


def test_last_async_backup_failure_blocks_final_publication(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    output, backup = tmp_path / 'out', tmp_path / 'backup'
    plan = release.release_plan(root, source, output, chunk_size=1)
    checkpoint = PreparationBackup.checkpoint

    def fail_last(self, unit, paths, identity_paths):
        if unit == 'chunks/train/000002':
            raise OSError('last checkpoint failed')
        return checkpoint(self, unit, paths, identity_paths)

    monkeypatch.setattr(PreparationBackup, 'checkpoint', fail_last)
    with pytest.raises(OSError, match='last checkpoint failed'):
        release.execute_release(plan, backup_dir=backup, overlap_backups=True)
    assert len(calls) == 6
    assert not (output / 'manifests').exists()
    assert not (output / 'READY.json').exists()
    assert 'final' not in {item['unit'] for item in backup_receipts(backup)}
    assert not any(t.name.startswith('chunk-backup') for t in threading.enumerate())


@pytest.mark.skipif(os.name != 'posix', reason='Preparation process control requires POSIX')
@pytest.mark.parametrize('workers', [1, 2])
def test_sigterm_drains_background_checkpoint_before_process_exit(tmp_path, workers):
    # An actual subprocess and signal prove the overlap lifetime retains the
    # existing graceful cancellation behavior; no renderer or model is invoked.
    code = '''
import sys, time
from pathlib import Path
from realtime_use import prepare_release as release
root = Path(sys.argv[1])
workers = int(sys.argv[2])
class Backup:
    def checkpoint(self, unit, *args):
        (root / f'{unit}.started').touch()
        deadline = time.monotonic() + 15
        while not (root / 'finish').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Checkpoint fixture timed out')
            time.sleep(.01)
        (root / f'{unit}.committed').touch()
with release.prep._graceful_sigterm(), release._ChunkBackups(Backup(), True, workers) as checkpoints:
    for index in range(workers):
        checkpoints.checkpoint(str(index), [], [])
    while True:
        time.sleep(1)
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(tmp_path), str(workers)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while len(list(tmp_path.glob('*.started'))) != workers:
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline
            time.sleep(.01)
        process.send_signal(signal.SIGTERM)
        time.sleep(.05)
        assert process.poll() is None, 'Parent exited before joining its active backup'
        (tmp_path / 'finish').touch()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        assert len(list(tmp_path.glob('*.committed'))) == workers
    finally:
        (tmp_path / 'finish').touch()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.parametrize('kwargs', [
    {'overlap_backups': True}, {'overlap_backups': 1},
])
def test_invalid_overlap_options_leave_existing_ready_untouched(tmp_path, monkeypatch, kwargs):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output)
    release.execute_release(plan)
    ready = (output / 'READY.json').read_bytes()
    with pytest.raises(ValueError, match='boolean|backup directory'):
        release.execute_release(plan, **kwargs)
    assert (output / 'READY.json').read_bytes() == ready
