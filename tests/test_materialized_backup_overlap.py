"""Source-stage overlap on generated archives, including real child cancellation."""
import json
import os
from pathlib import Path
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


_materialize = release._materialize_group


def wait_for(path, seconds=20):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f'Timed out waiting for fixture: {path}')
        time.sleep(.01)


def controlled_materialize(root, path, output, role, scope):
    """Picklable control wrapper; extraction still runs the real materializer."""
    root, key = Path(root), Path(output).name
    control = json.loads((root / 'overlap-control.json').read_text())
    (root / f'started-{key}').write_text(str(os.getpid()))
    if key == control['keys'][1]:
        if control['mode'] == 'blocked':
            wait_for(root / 'allow-extraction', seconds=60)
        elif control['mode'] == 'extract-failure':
            wait_for(root / 'backup-started')
            (root / 'extract-failed').touch()
            raise ValueError('fixture extraction failed')
    result = _materialize(str(root), path, output, role, scope)
    (root / f'closed-{key}').touch()
    return result


def setup_run(tmp_path, monkeypatch, mode='normal'):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, renderer = fake_renderer(monkeypatch)
    output, backup = tmp_path / 'out', tmp_path / 'backup'
    plan = release.release_plan(root, source, output, chunk_size=1)
    keys = list(plan['groups'])
    (root / 'overlap-control.json').write_text(json.dumps({'keys': keys, 'mode': mode}))
    monkeypatch.setattr(release, '_materialize_group', controlled_materialize)
    return root, output, backup, plan, keys, calls


def test_overlap_is_bounded_and_all_receipts_precede_audit_with_exact_restore(tmp_path, monkeypatch):
    root, output, backup, plan, keys, calls = setup_run(tmp_path, monkeypatch)
    checkpoint, audit = PreparationBackup.checkpoint, release.audit_materialized_sources
    owner = threading.get_ident()
    active, committed = [], []
    def delayed(self, unit, paths, identities):
        if unit.startswith('materialized/'):
            key = unit.split('/', 1)[1]
            assert threading.get_ident() != owner
            assert (root / f'closed-{key}').exists()
            active.append(key)
            assert len(active) == 1
            if key == keys[0]:
                wait_for(root / f'closed-{keys[1]}')
                # Backup A and closed B are the entire uncommitted backlog.
                assert not (root / f'started-{keys[2]}').exists()
            result = checkpoint(self, unit, paths, identities)
            committed.append(key)
            active.remove(key)
            return result
        return checkpoint(self, unit, paths, identities)
    def audited(request, materialized):
        assert not active and committed == keys
        assert list(materialized) == keys
        assert not calls
        return audit(request, materialized)
    monkeypatch.setattr(PreparationBackup, 'checkpoint', delayed)
    monkeypatch.setattr(release, 'audit_materialized_sources', audited)
    result = release.execute_release(plan, backup_dir=backup, overlap_materialized_backups=True)
    assert result['training_started'] is False
    request = (output / 'request.json').read_bytes()
    assert 'overlap_materialized_backups' not in json.loads(request)
    assert json.loads((output / 'execution.json').read_text())['overlap_materialized_backups'] is True
    audio = {p.relative_to(output): p.read_bytes() for p in output.rglob('*.wav')}
    receipts = backup_receipts(backup)
    final = next(row for row in receipts if row['unit'] == 'final')
    assert {unit for unit in final['dependencies'] if unit.startswith('materialized/')} == {
        f'materialized/{key}' for key in keys}
    monkeypatch.setattr(PreparationBackup, 'checkpoint', checkpoint)
    monkeypatch.setattr(release, 'audit_materialized_sources', audit)
    assert release.execute_release(plan, backup_dir=backup) == result
    assert (output / 'request.json').read_bytes() == request
    assert len(backup_receipts(backup)) == len(receipts)
    shutil.rmtree(output)
    assert restore(backup, output)['ready_restored'] is True
    assert {p: (output / p).read_bytes() for p in audio} == audio


def test_backup_failure_promptly_terminates_active_extractor_and_never_audits(tmp_path, monkeypatch):
    root, output, backup, plan, keys, calls = setup_run(tmp_path, monkeypatch, mode='blocked')
    checkpoint, audit = PreparationBackup.checkpoint, release.audit_materialized_sources
    def failed(self, unit, paths, identities):
        if unit == f'materialized/{keys[0]}':
            wait_for(root / f'started-{keys[1]}')
            raise OSError('fixture persistent backup failed')
        return checkpoint(self, unit, paths, identities)
    monkeypatch.setattr(PreparationBackup, 'checkpoint', failed)
    monkeypatch.setattr(release, 'audit_materialized_sources', lambda *args: pytest.fail('Premature audit'))
    with pytest.raises(OSError, match='persistent backup failed'):
        release.execute_release(plan, backup_dir=backup, overlap_materialized_backups=True)
    pid = int((root / f'started-{keys[1]}').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not (root / f'started-{keys[2]}').exists()
    assert not calls and not (output / 'READY.json').exists()
    assert not any(t.name.startswith('materialized-backup') for t in threading.enumerate())
    # The failed group remains local and can be retried with unchanged identity.
    request = (output / 'request.json').read_bytes()
    (root / 'allow-extraction').touch()
    monkeypatch.setattr(PreparationBackup, 'checkpoint', checkpoint)
    monkeypatch.setattr(release, 'audit_materialized_sources', audit)
    result = release.execute_release(plan, backup_dir=backup, overlap_materialized_backups=True)
    assert result['training_pairs_ready'] is True and result['training_started'] is False
    assert (output / 'request.json').read_bytes() == request


def test_extraction_failure_drains_started_backup_before_return(tmp_path, monkeypatch):
    root, output, backup, plan, keys, calls = setup_run(tmp_path, monkeypatch, mode='extract-failure')
    checkpoint = PreparationBackup.checkpoint
    def delayed(self, unit, paths, identities):
        if unit == f'materialized/{keys[0]}':
            (root / 'backup-started').touch()
            wait_for(root / 'extract-failed')
        return checkpoint(self, unit, paths, identities)
    monkeypatch.setattr(PreparationBackup, 'checkpoint', delayed)
    monkeypatch.setattr(release, 'audit_materialized_sources', lambda *args: pytest.fail('Premature audit'))
    with pytest.raises(ValueError, match='extraction failed'):
        release.execute_release(plan, backup_dir=backup, overlap_materialized_backups=True)
    assert f'materialized/{keys[0]}' in {row['unit'] for row in backup_receipts(backup)}
    assert not (root / f'started-{keys[2]}').exists()
    assert not calls and not (output / 'READY.json').exists()
    assert not any(t.name.startswith('materialized-backup') for t in threading.enumerate())


@pytest.mark.parametrize('kwargs', [
    {'overlap_materialized_backups': True},
    {'overlap_materialized_backups': 1},
    {'overlap_materialized_backups': True, 'materialize_workers': 2, 'backup_dir': 'persistent'},
])
def test_invalid_overlap_options_fail_before_output_mutation(tmp_path, monkeypatch, kwargs):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output)
    release.execute_release(plan)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    if kwargs.get('backup_dir'):
        kwargs['backup_dir'] = tmp_path / kwargs['backup_dir']
    with pytest.raises(ValueError, match='boolean|backup directory'):
        release.execute_release(plan, **kwargs)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert not (tmp_path / 'persistent').exists()


@pytest.mark.skipif(os.name != 'posix', reason='Preparation signals require POSIX')
def test_sigterm_kills_extractor_but_joins_closed_group_backup(tmp_path):
    script = tmp_path / 'signal_fixture.py'
    script.write_text('''import sys
from pathlib import Path
import pytest
sys.path.insert(0, sys.argv[2])
from test_materialized_backup_overlap import setup_run, wait_for
from realtime_use import prepare_release as release
from realtime_use.preparation_backup import PreparationBackup

if __name__ == '__main__':
    base = Path(sys.argv[1])
    patch = pytest.MonkeyPatch()
    root, output, backup, plan, keys, calls = setup_run(base, patch, mode='blocked')
    original = PreparationBackup.checkpoint
    def pending(self, unit, paths, identity):
        if unit == f'materialized/{keys[0]}':
            (base / 'backup-started').touch()
            wait_for(base / 'allow-backup', seconds=60)
        return original(self, unit, paths, identity)
    patch.setattr(PreparationBackup, 'checkpoint', pending)
    release.execute_release(plan, backup_dir=backup, overlap_materialized_backups=True)
''')
    process = subprocess.Popen([sys.executable, str(script), str(tmp_path), str(Path(__file__).parent)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        wait_for(tmp_path / 'backup-started')
        control = json.loads((tmp_path / 'dataset/overlap-control.json').read_text())
        pid_path = tmp_path / 'dataset' / f'started-{control["keys"][1]}'
        wait_for(pid_path)
        extractor_pid = int(pid_path.read_text())
        process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while True:
            try:
                os.kill(extractor_pid, 0)
            except ProcessLookupError:
                break
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert process.poll() is None
        process.send_signal(signal.SIGTERM)
        (tmp_path / 'allow-backup').touch()
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, (stdout, stderr)
        assert not (tmp_path / 'out/READY.json').exists()
        shutil.rmtree(tmp_path / 'out')
        assert restore(tmp_path / 'backup', tmp_path / 'out')['ready_restored'] is False
        assert (tmp_path / 'out/materialized' / control['keys'][0] / 'source.scp').exists()
    finally:
        (tmp_path / 'allow-backup').touch()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
