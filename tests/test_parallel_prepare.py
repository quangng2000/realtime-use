"""CPU process rendering checks; the optional real simulator never trains."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from realtime_use import prepare_data as prep


@pytest.mark.parametrize('workers', [0, -1, True, 1.5, '2'])
def test_worker_count_rejected_before_touching_output(workers):
    with pytest.raises(ValueError, match='positive integer'):
        prep.execute_preparation({}, workers=workers)


def test_spawn_environment_limits_children_and_restores_caller(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.delenv('MKL_NUM_THREADS', raising=False)
    with prep._spawn_cpu_environment():
        assert os.environ['OMP_NUM_THREADS'] == '1'
        assert os.environ['MKL_NUM_THREADS'] == '1'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['OMP_NUM_THREADS'] == '8'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert 'MKL_NUM_THREADS' not in os.environ


def official_plan(tmp_path, output_name, *, with_ffmpeg=False):
    location = os.environ.get('REALTIME_USE_URGENT_SOURCE')
    if not location:
        pytest.skip('Set REALTIME_USE_URGENT_SOURCE for real spawned URGENT rendering')
    source = Path(location).resolve()
    prep.verify_source(source)
    rate, frames = 16000, 8000
    speech = .1 * np.sin(2 * np.pi * 300 * np.arange(frames) / rate)
    noise = np.random.default_rng(13).normal(0, .02, frames)
    rir = np.zeros(256); rir[37] = 1; rir[99] = .4
    for name, wave in [('speech', speech), ('noise', noise), ('rir', rir)]:
        sf.write(tmp_path / f'{name}.wav', wave, rate, subtype='FLOAT')
    inventories = {role: {uid: {'path': str(tmp_path / f'{name}.wav'), 'sample_rate': rate}}
                   for role, uid, name in [('speech', 'speech', 'speech'), ('noise', 'noise', 'noise'),
                                            ('wind', 'wind_noise_test', 'noise'), ('rir', 'rir', 'rir')]}
    output = tmp_path / output_name
    (output / 'log').mkdir(parents=True)
    plan = {'source': str(source), 'output_dir': str(output), 'inventories': inventories,
            'rir_alignment': {'method': 'upstream-estimated-onset'},
            'provenance': {'split': 'train'}, 'usage': 'research', 'commercial_clearance': 'pending',
            'target': 'anechoic_unit_gain_estimated_onset_delayed_same_joint_normalization'}
    augmentations = ['none', 'clipping(min=0.05,max=0.95)',
                     'packet_loss(packet_loss_indices=[1,3],packet_duration_ms=20)',
                     'bandwidth_limitation-polyphase->8000'] * 2
    if with_ffmpeg:
        import shutil
        from torchaudio.utils import ffmpeg_utils
        if not shutil.which('ffmpeg'):
            pytest.skip('FFmpeg executable/shared libraries required for codec/wind checks')
        assert ffmpeg_utils.get_versions()
        augmentations = ['codec(format=mp3,encoder=None,qscale=3)',
            'wind_noise(threshold=0.2,ratio=2,attack=10,release=20,sc_gain=1,clipping=False,clipping_threshold=0.9)'] * 4
    metadata = [dict(id=f'fileid_{i}', fs=rate, length=frames, snr_dB=10,
                     speech_uid='speech', speech_sid='fixture-speaker',
                     noise_uid='wind_noise_test' if augmentation.startswith('wind_noise') else 'noise',
                     rir_uid='none' if augmentation.startswith('wind_noise') else 'rir',
                     augmentation=augmentation) for i, augmentation in enumerate(augmentations)]
    with (output / 'log/meta.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(metadata)
    return plan


@pytest.mark.parametrize('with_ffmpeg', [False, True])
def test_real_spawn_matches_sequential_and_resumes_with_changed_worker_count(tmp_path, monkeypatch, with_ffmpeg):
    sequential = official_plan(tmp_path, 'sequential', with_ffmpeg=with_ffmpeg)
    parallel = official_plan(tmp_path, 'parallel', with_ffmpeg=with_ffmpeg)
    # Sequential wind uses the caller's cwd; spawned workers always have their
    # own workspace, even when two wind rows run at the same time.
    monkeypatch.chdir(tmp_path)
    module = prep.load_simulator(Path(sequential['source']))
    expected = prep.render_metadata(sequential, module,
        progress_path=Path(sequential['output_dir']) / 'render-progress.jsonl')
    stats = {}
    ledger = Path(parallel['output_dir']) / 'render-progress.jsonl'
    rows = prep.render_metadata(parallel, None, progress_path=ledger, workers=2, execution=stats)
    assert stats['rendered'] == 8 and stats['reused'] == 0
    assert stats['worker_pids'] and os.getpid() not in stats['worker_pids']
    assert stats['max_in_flight'] <= 4
    assert [row['id'] for row in rows] == [row['id'] for row in expected]
    assert [entry['index'] for entry in map(json.loads, ledger.read_text().splitlines())] == list(range(8))
    for left, right in zip(expected, rows):
        for field in ['noisy', 'clean']:
            np.testing.assert_array_equal(sf.read(left[field])[0], sf.read(right[field])[0])
        assert left['rir_alignment'] == right['rir_alignment']
    assert not list(Path(parallel['output_dir']).glob('.render-workers-*'))
    saved = ledger.read_bytes()
    files = {row['clean']: Path(row['clean']).stat().st_mtime_ns for row in rows}
    resumed = {}
    again = prep.render_metadata(parallel, module, progress_path=ledger, workers=1, execution=resumed)
    assert again == rows and ledger.read_bytes() == saved
    assert resumed['reused'] == 8 and resumed['rendered'] == 0
    assert all(Path(path).stat().st_mtime_ns == timestamp for path, timestamp in files.items())


def test_spawn_failure_keeps_only_ordered_successes_and_retries_uncommitted_rows(tmp_path, monkeypatch):
    plan = official_plan(tmp_path, 'failure')
    output = Path(plan['output_dir'])
    metadata_path = output / 'log/meta.tsv'
    with metadata_path.open() as handle:
        metadata = list(csv.DictReader(handle, delimiter='\t'))
    metadata[2]['speech_uid'] = 'temporarily_unavailable'
    missing = tmp_path / 'temporarily_unavailable.wav'
    plan['inventories']['speech']['temporarily_unavailable'] = {'path': str(missing), 'sample_rate': 16000}
    with metadata_path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(metadata)
    ledger = output / 'render-progress.jsonl'
    with pytest.raises(FileNotFoundError):
        prep.render_metadata(plan, None, progress_path=ledger, workers=2)
    assert [entry['index'] for entry in map(json.loads, ledger.read_text().splitlines())] == [0, 1]
    missing.write_bytes((tmp_path / 'speech.wav').read_bytes())
    stats = {}
    rows = prep.render_metadata(plan, None, progress_path=ledger, workers=2, execution=stats)
    assert len(rows) == 8 and stats['reused'] == 2 and stats['rendered'] == 6
    assert [entry['index'] for entry in map(json.loads, ledger.read_text().splitlines())] == list(range(8))
    assert not list(output.glob('.render-workers-*'))


def test_sigterm_context_restores_handler_and_rejects_background_thread():
    import signal
    import threading
    original = signal.getsignal(signal.SIGTERM)
    with prep._graceful_sigterm():
        current = signal.getsignal(signal.SIGTERM)
        assert current is not original
        with prep._graceful_sigterm():
            assert signal.getsignal(signal.SIGTERM) is current
    assert signal.getsignal(signal.SIGTERM) is original
    failures = []
    def background():
        try:
            with prep._graceful_sigterm():
                pass
        except RuntimeError as exc:
            failures.append(str(exc))
    worker = threading.Thread(target=background)
    worker.start(); worker.join()
    assert len(failures) == 1 and 'main thread' in failures[0]


@pytest.mark.skipif(os.name != 'posix', reason='Preparation locking and signals target Linux/macOS')
def test_parent_sigterm_joins_spawned_workers_before_releasing_output_lock(tmp_path):
    import fcntl
    import signal
    import subprocess
    import sys
    import time
    script = tmp_path / 'sigterm_prepare_worker.py'
    script.write_text('''from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing
import os
from pathlib import Path
import sys
import time
from realtime_use.prepare_data import _graceful_sigterm

def work(root, number):
    root = Path(root)
    (root / f'worker-{number}.pid').write_text(str(os.getpid()))
    while not (root / 'release.flag').exists():
        time.sleep(.02)
    return number

if __name__ == '__main__':
    root = Path(sys.argv[1])
    with _graceful_sigterm(), (root / 'output.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'))
        try:
            jobs = [pool.submit(work, str(root), number) for number in range(2)]
            for job in jobs:
                job.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
''')
    child = subprocess.Popen([sys.executable, str(script), str(tmp_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    worker_pids = []
    try:
        deadline = time.monotonic() + 30
        while len(list(tmp_path.glob('worker-*.pid'))) < 2:
            if child.poll() is not None or time.monotonic() > deadline:
                output = child.communicate(timeout=1) if child.poll() is not None else ('', 'startup timed out')
                pytest.fail(f'Spawned preparation fixture failed: {output}')
            time.sleep(.02)
        worker_pids = [int(path.read_text()) for path in tmp_path.glob('worker-*.pid')]
        child.send_signal(signal.SIGTERM)
        time.sleep(.1)
        # A repeated SIGTERM must not interrupt joining the active workers.
        child.send_signal(signal.SIGTERM)
        time.sleep(.1)
        assert child.poll() is None
        with (tmp_path / 'output.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for pid in worker_pids:
            os.kill(pid, 0)
        (tmp_path / 'release.flag').touch()
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        for pid in worker_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        with (tmp_path / 'output.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        (tmp_path / 'release.flag').touch()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate(timeout=10)
