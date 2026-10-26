"""Committed-metadata inventory pruning, using audio rendering only; no training."""
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from realtime_use import prepare_data as prep


def fixture_simulator():
    """Deterministic local audio fixture exercising the complete render adapter."""
    module = SimpleNamespace(
        estimate_early_rir=lambda value, **kwargs: value,
        _realtime_use_rir_onset=lambda value, level_ratio: 3,
        wind_noise=lambda speech_sample, clipping: speech_sample + (.03 if clipping else .02),
        save_audio=lambda *args, **kwargs: None)
    def process(info, *, speech_dic, noise_dic, rir_dic, **kwargs):
        speech, rate = sf.read(speech_dic[info['speech_uid']], dtype='float32')
        noise, _ = sf.read(noise_dic[info['noise_uid']], dtype='float32')
        clean, noisy = speech.copy(), speech.copy()
        if info['rir_uid'] != 'none':
            rir, _ = sf.read(rir_dic[info['rir_uid']], dtype='float32')
            target_rir = module.estimate_early_rir(rir[None, :], fs=rate)[0]
            clean = np.convolve(speech, target_rir)[:len(speech)]
            noisy = np.convolve(speech, rir)[:len(speech)]
        if info['augmentation'].startswith('wind_noise'):
            noisy = module.wind_noise(noisy, clipping=True)
        noisy = noisy + .1 * noise
        module.save_audio(noisy, info['noisy_path'], rate)
        module.save_audio(clean, info['clean_path'], rate)
    module.process_one_sample = process
    return module


def fixture_worker_init(plan, scratch):
    """Spawnable fixture initializer; production scheduling/rendering is retained."""
    mappings = {role: {uid: row['path'] for uid, row in inventory.items()}
                for role, inventory in plan['inventories'].items()}
    mappings['noise'].update(mappings['wind'])
    os.chdir(tempfile.mkdtemp(prefix='fixture-', dir=scratch))
    prep._RENDER_WORKER = plan, fixture_simulator(), mappings, {}


def write_metadata(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def add_unused(plan, count=1000):
    for role, inventory in plan['inventories'].items():
        for index in range(count):
            inventory[f'unused-{role}-{index}'] = {
                'sample_rate': 16000, 'path': f'/unopened-fixture/{role}/{index}.wav',
                'source_id': f'fixture-{role}'}


def fixture_plan(tmp_path):
    rate, frames = 16000, 320
    waves = {'speech': np.linspace(-.1, .1, frames, dtype=np.float32),
             'noise': np.random.default_rng(12).normal(0, .02, frames).astype(np.float32),
             'wind_noise_fixture': np.full(frames, .01, dtype=np.float32),
             'rir': np.r_[np.zeros(3), 1., .3, np.zeros(5)].astype(np.float32)}
    inventories = {role: {} for role in ('speech', 'noise', 'wind', 'rir')}
    for uid, wave in waves.items():
        path = tmp_path / f'{uid}.wav'
        sf.write(path, wave, rate, subtype='FLOAT')
        role = 'wind' if uid.startswith('wind_noise') else uid
        inventories[role][uid] = {'path': str(path), 'sample_rate': rate}
    output = tmp_path / 'render'
    (output / 'log').mkdir(parents=True)
    plan = {'source': str(tmp_path / 'fixture-source'), 'output_dir': str(output),
            'inventories': inventories, 'config': {'noise_scps': ['unchanged-full-pool.scp']},
            'rir_alignment': {'method': 'upstream-estimated-onset'},
            'provenance': {'split': 'train'}, 'usage': 'research',
            'commercial_clearance': 'pending',
            'target': 'anechoic_unit_gain_estimated_onset_delayed_same_joint_normalization'}
    rows = []
    for index in range(24):
        wind = index % 3 == 1
        rows.append({'id': f'fileid_{index}', 'speech_uid': 'speech', 'speech_sid': 'speaker',
                     'noise_uid': 'wind_noise_fixture' if wind else 'noise',
                     'rir_uid': 'rir' if index % 3 == 0 else 'none',
                     'snr_dB': '10', 'fs': str(rate), 'length': str(frames),
                     'augmentation': ('wind_noise(clipping=False,threshold=0.2)' if wind else 'none')})
    write_metadata(output / 'log/meta.tsv', rows)
    add_unused(plan)
    return plan


def render_snapshot(output):
    """Compare exact samples; FLOAT WAV PEAK timestamps are execution-dependent."""
    files = {str(path.relative_to(output)): path.read_bytes()
             for path in output.rglob('*') if path.is_file()}
    result = {}
    for name, raw in files.items():
        if name.endswith('.wav'):
            audio, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            result[name] = rate, audio.shape, audio.tobytes()
        elif name == 'render-progress.jsonl':
            records = []
            for line in raw.splitlines():
                record = json.loads(line)
                assert record['record_sha256'] == prep._json_hash(
                    {key: value for key, value in record.items() if key != 'record_sha256'})
                for role, expected in record['output_sha256'].items():
                    relative = str(Path(record['row'][role]).relative_to(output))
                    assert hashlib.sha256(files[relative]).hexdigest() == expected
                # Each run's file/checkpoint hashes were independently validated;
                # only timestamp-sensitive hashes are excluded from comparison.
                records.append({key: value for key, value in record.items()
                                if key not in {'record_sha256', 'output_sha256'}})
            result[name] = records
        else:
            result[name] = raw
    return result


def assert_one_vs_twelve_sample_parity(plan, module):
    original = deepcopy(plan)
    output = Path(plan['output_dir'])
    ledger = output / 'render-progress.jsonl'
    # Full parent artifacts must stay full and byte-identical across execution.
    full_plan = json.dumps(prep._base_plan(plan), sort_keys=True).encode()
    for name in ('plan.json', 'provenance.json'):
        (output / name).write_bytes(full_plan)
    expected = prep.render_metadata(plan, module, workers=1, progress_path=ledger)
    snapshots = render_snapshot(output)
    for name in ('clean', 'noisy'):
        shutil.rmtree(output / name)
    ledger.unlink()
    stats = {}
    parallel = deepcopy(original)
    observed = prep.render_metadata(parallel, None, workers=12, progress_path=ledger, execution=stats)
    assert observed == expected
    assert snapshots == render_snapshot(output)
    assert stats['workers'] == 12 and os.getpid() not in stats['worker_pids']
    assert stats['rendered'] == len(expected)
    assert parallel['inventories'] == original['inventories']
    assert prep._base_plan(parallel) == prep._base_plan(original)


def test_pruned_twelve_spawned_workers_match_full_sequential_samples_and_metadata(tmp_path, monkeypatch):
    plan = fixture_plan(tmp_path)
    monkeypatch.setattr(prep, '_render_worker_init', fixture_worker_init)
    assert_one_vs_twelve_sample_parity(plan, fixture_simulator())


def test_worker_payload_contains_only_selected_ids_without_mutating_full_plan(tmp_path):
    plan = fixture_plan(tmp_path)
    original = deepcopy(plan)
    metadata = prep.read_metadata(Path(plan['output_dir']) / 'log/meta.tsv')
    worker = prep._render_worker_plan(plan, metadata)
    assert {role: set(entries) for role, entries in worker['inventories'].items()} == {
        'speech': {'speech'}, 'noise': {'noise'}, 'wind': {'wind_noise_fixture'}, 'rir': {'rir'}}
    assert worker['config'] == plan['config']
    assert worker['provenance'] == plan['provenance']
    assert len(pickle.dumps(worker)) < len(pickle.dumps(plan)) / 50
    worker['inventories']['speech']['speech']['path'] = 'mutated-worker-entry'
    assert plan == original
    # A chunk without reverberation needs no RIR inventory at all.
    for row in metadata:
        row['rir_uid'] = 'none'
    assert prep._render_worker_plan(plan, metadata)['inventories']['rir'] == {}


@pytest.mark.parametrize('role,uid', [('speech', 'speech'), ('noise', 'noise'),
                                    ('wind', 'wind_noise_fixture'), ('rir', 'rir')])
def test_missing_selected_id_fails_before_spawn_or_audio_output(tmp_path, monkeypatch, role, uid):
    plan = fixture_plan(tmp_path)
    del plan['inventories'][role][uid]
    def forbidden_pool(*args, **kwargs):
        pytest.fail('Missing metadata IDs must fail before spawning')
    monkeypatch.setattr(prep, 'ProcessPoolExecutor', forbidden_pool)
    output = Path(plan['output_dir'])
    with pytest.raises(ValueError, match='missing .* ID'):
        prep.render_metadata(plan, None, workers=12, progress_path=output / 'render-progress.jsonl')
    assert not (output / 'render-progress.jsonl').exists()
    assert not (output / 'clean').exists() and not (output / 'noisy').exists()


def test_official_urgent_one_vs_twelve_exact_samples_with_wind_and_unused_pools(tmp_path, monkeypatch):
    # The preparation host provides pinned URGENT/ESPnet/FFmpeg. This opt-in test
    # uses that unmodified renderer and no fixture initializer or training code.
    from test_parallel_prepare import official_plan
    plan = official_plan(tmp_path, 'official', with_ffmpeg=True)
    output = Path(plan['output_dir'])
    rows = prep.read_metadata(output / 'log/meta.tsv')
    for index, rir in enumerate(('rir', 'none'), len(rows)):
        rows.append({**rows[0], 'id': f'fileid_{index}', 'noise_uid': 'noise',
                     'rir_uid': rir, 'augmentation': 'none'})
    write_metadata(output / 'log/meta.tsv', rows)
    add_unused(plan)
    monkeypatch.chdir(tmp_path)
    module = prep.load_simulator(Path(plan['source']))
    assert_one_vs_twelve_sample_parity(plan, module)
