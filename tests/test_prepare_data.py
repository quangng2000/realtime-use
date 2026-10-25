"""Simulator-adapter tests use generated tiny audio; no training or real data."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from realtime_use import prepare_data as prep


def input_files(tmp_path):
    sources, config = [], {'speech_text': [], 'prob_reverberation': .25, 'prob_wind_noise': .1, 'num_augmentations': {1: 1.0},
                           'augmentations': {key: {'weight': 1} for key in
                                             ('clipping', 'bandwidth_limitation', 'codec', 'packet_loss')}}
    for field, role in prep.ROLES.items():
        uid = 'wind_noise_test' if role == 'wind' else role
        audio = tmp_path / f'{role}.wav'
        sf.write(audio, np.linspace(-.1, .1, 80), 8000, subtype='FLOAT')
        scp = tmp_path / f'{role}.scp'
        scp.write_text(f'{uid} 8000 {audio}\n')
        config[field] = [str(scp)]
        sources.append(dict(id=f'test-{role}', scp=str(scp), role=role,
                            license='CC0-1.0', license_evidence='generated local test fixture', commercial_use=True))
    speakers = tmp_path / 'speakers'
    speakers.write_text('speech local-speaker\n')
    config['speech_utt2spk'] = [str(speakers)]
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    provenance = dict(split='train', sources=sources, rir_direct_path_samples={'rir': {'8000': 3}})
    provenance_path = tmp_path / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance))
    return config_path, provenance_path


def test_plan_checks_native_inputs_permissions_and_documents_upstream_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'verify_source', lambda path: None)
    config, provenance = input_files(tmp_path)
    plan = prep.preparation_plan(tmp_path, config, provenance, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    assert plan['config']['out_format'] == 'wav'
    assert '0.75' in plan['upstream_probability_note']
    assert plan['inventories']['speech']['speech']['sample_rate'] == 8000
    record = json.loads(provenance.read_text())
    record['sources'][0]['commercial_use'] = False
    provenance.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='commercial-use'):
        prep.preparation_plan(tmp_path, config, provenance, tmp_path / 'output')


def test_anechoic_target_is_delayed_unit_impulse_without_early_reflections():
    rir = np.array([[0., .4, 1., .7, .3]])
    target = prep.direct_path_impulse(rir, 1)
    np.testing.assert_array_equal(target, [[0, 1, 0, 0, 0]])
    speech = np.array([1., 2., 3.])
    np.testing.assert_array_equal(np.convolve(speech, target[0])[:3], [0, 1, 2])
    with pytest.raises(ValueError, match='fit'):
        prep.direct_path_impulse(rir, 5)


def test_scoped_render_adapters_protect_target_and_correct_wind_false(tmp_path):
    observed = {}
    def wind_noise(speech_sample, clipping):
        observed['clipping'] = clipping
        speech_sample *= 2
        return speech_sample
    module = SimpleNamespace(estimate_early_rir=lambda value, **kw: value,
                             wind_noise=wind_noise, save_audio=lambda *args: None)
    original = module.estimate_early_rir
    clean = np.ones((1, 5))
    with prep.rendering_adapters(module, delay=2, wind_clipping=False):
        np.testing.assert_array_equal(module.estimate_early_rir(np.ones((1, 5))), [[0, 0, 1, 0, 0]])
        output = module.wind_noise(clean, clipping=True)
        assert observed['clipping'] is False
        np.testing.assert_array_equal(clean, np.ones((1, 5)))
        np.testing.assert_array_equal(output, np.ones((1, 5)) * 2)
        module.save_audio(clean, tmp_path / 'float.wav', 8000)
        assert sf.info(tmp_path / 'float.wav').subtype == 'FLOAT'
    assert module.estimate_early_rir is original
    assert module.wind_noise is wind_noise


def test_rendered_manifest_preserves_speaker_rate_and_source_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'verify_source', lambda path: None)
    config, provenance = input_files(tmp_path)
    output = tmp_path / 'output'
    plan = prep.preparation_plan(tmp_path, config, provenance, output)
    (output / 'log').mkdir(parents=True)
    row = dict(id='fileid_1', noisy_path='/must/not/write.wav', clean_path='/must/not/write2.wav',
               fs=8000, length=80, snr_dB=5, speech_uid='speech', speech_sid='local-speaker',
               noise_uid='noise', rir_uid='rir', augmentation='clipping(min=0,max=0.9)')
    with (output / 'log/meta.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter='\t')
        writer.writeheader(); writer.writerow(row)
    calls = []
    def process(info, **kwargs):
        calls.append((info, kwargs))
        assert str(output) in info['clean_path']
    module = SimpleNamespace(process_one_sample=process, estimate_early_rir=lambda value: value,
                             wind_noise=lambda *args: None, save_audio=lambda *args: None)
    rows = prep.render_metadata(plan, module)
    assert len(calls) == 1
    assert rows[0]['speaker_id'] == 'local-speaker'
    assert rows[0]['sample_rate'] == 8000
    assert rows[0]['direct_path_delay_samples'] == 3
    assert len(rows[0]['source_speech_sha256']) == 64
    assert len(rows[0]['source_noise_sha256']) == 64


def test_split_audit_rejects_speaker_recording_or_noise_leakage(tmp_path):
    paths = []
    for split in ['train', 'validation']:
        path = tmp_path / f'{split}.jsonl'
        row = dict(split=split, speech_uid=split, speaker_id=f'speaker-{split}',
                   source_speech_sha256=f'hash-{split}', noise_uid=f'noise-{split}',
                   source_noise_sha256=f'noisehash-{split}')
        path.write_text(json.dumps(row) + '\n')
        paths.append(path)
    assert prep.audit_splits(paths)['examples'] == 2
    row['speaker_id'] = 'speaker-train'
    paths[1].write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Split leakage'):
        prep.audit_splits(paths)


def test_import_has_no_optional_simulator_or_network_side_effects():
    import subprocess
    import sys
    code = '''import socket, sys
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network forbidden"))
import realtime_use.prepare_data
assert "espnet2.train.preprocessor" not in sys.modules
assert "g2p_en" not in sys.modules
assert "simulate_data_from_param" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


def test_research_mode_preserves_pending_permissions_without_claiming_clearance(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'verify_source', lambda path: None)
    config, provenance = input_files(tmp_path)
    record = json.loads(provenance.read_text())
    record['sources'][0].update(commercial_use=None, license='CC-BY-NC-4.0')
    record.pop('rir_direct_path_samples')
    provenance.write_text(json.dumps(record))
    plan = prep.preparation_plan(tmp_path, config, provenance, tmp_path / 'output',
                                 usage='research', rir_alignment='upstream-estimated-onset')
    assert plan['usage'] == 'research'
    assert plan['commercial_clearance'] == 'pending'
    assert plan['provenance']['sources'][0]['commercial_use'] is None
    assert plan['rir_alignment']['physical_direct_path_audited'] is False
    assert 'estimated_onset' in plan['target']
    assert 'rir_direct_path_samples' not in plan['provenance']
    assert not (tmp_path / 'output').exists()
    with pytest.raises(ValueError, match='commercial-use'):
        prep.preparation_plan(tmp_path, config, provenance, tmp_path / 'other',
                              rir_alignment='upstream-estimated-onset')
    with pytest.raises(ValueError, match='audited direct-path'):
        prep.preparation_plan(tmp_path, config, provenance, tmp_path / 'other', usage='research')
    record['sources'][0]['license_evidence'] = ''
    provenance.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='license evidence'):
        prep.preparation_plan(tmp_path, config, provenance, tmp_path / 'other',
                              usage='research', rir_alignment='upstream-estimated-onset')


def test_estimated_onset_records_actual_rendered_rir_and_is_not_a_physical_audit():
    received = {}
    rir = np.array([[0., .01, .2, 1., .5]], dtype=np.float32)
    def estimator(wave, level_ratio):
        received.update(wave=wave.copy(), level_ratio=level_ratio)
        return 2
    module = SimpleNamespace(estimate_early_rir=lambda value: value,
                             wind_noise=lambda *args: None, save_audio=lambda *args: None,
                             _realtime_use_rir_onset=estimator)
    with prep.rendering_adapters(module, delay=None, wind_clipping=False,
                                 rir_alignment='upstream-estimated-onset') as evidence:
        impulse = module.estimate_early_rir(rir, fs=8000)
    np.testing.assert_array_equal(received['wave'], rir[0])
    np.testing.assert_array_equal(impulse, [[0, 0, 1, 0, 0]])
    assert received['level_ratio'] == .1
    assert evidence['sample_rate'] == 8000
    assert evidence['delay_samples'] == 2
    assert evidence['physical_direct_path_audited'] is False
    assert len(evidence['rendered_rir_sha256']) == 64
    for invalid in (np.zeros((1, 4)), np.full((1, 4), np.nan), np.ones((2, 4))):
        with prep.rendering_adapters(module, delay=None, wind_clipping=False,
                                     rir_alignment='upstream-estimated-onset'):
            with pytest.raises(ValueError, match='nonzero mono'):
                module.estimate_early_rir(invalid, fs=8000)


def test_estimated_manifest_does_not_mislabel_delay_as_audited(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'verify_source', lambda path: None)
    config, provenance = input_files(tmp_path)
    output = tmp_path / 'output'
    plan = prep.preparation_plan(tmp_path, config, provenance, output,
                                 usage='research', rir_alignment='upstream-estimated-onset')
    (output / 'log').mkdir(parents=True)
    row = dict(id='fileid_1', fs=8000, length=80, snr_dB=5, speech_uid='speech',
               speech_sid='local-speaker', noise_uid='noise', rir_uid='rir', augmentation='none')
    with (output / 'log/meta.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter='\t')
        writer.writeheader(); writer.writerow(row)
    module = SimpleNamespace(estimate_early_rir=lambda value: value,
                             wind_noise=lambda *args: None, save_audio=lambda *args: None,
                             _realtime_use_rir_onset=lambda wave, level_ratio: 2)
    def process(info, **kwargs):
        module.estimate_early_rir(np.array([[0., 0., 1., .5]]), fs=int(info['fs']))
    module.process_one_sample = process
    rows = prep.render_metadata(plan, module)
    assert rows[0]['estimated_onset_delay_samples'] == 2
    assert 'direct_path_delay_samples' not in rows[0]
    assert rows[0]['usage'] == 'research'
    assert rows[0]['commercial_clearance'] == 'pending'
    assert rows[0]['rir_alignment']['physical_direct_path_audited'] is False
    evidence = plan['rir_alignment_evidence']['rir']['8000']
    assert evidence['source_rir_sha256'] == prep.sha256(tmp_path / 'rir.wav')
    assert evidence['rendered_rir_frames'] == 4



def resumable_execution(tmp_path, monkeypatch):
    """A two-row simulator fixture that exercises the real persistence path."""
    monkeypatch.setattr(prep, 'verify_source', lambda path: None)
    monkeypatch.setattr(prep.shutil, 'which', lambda name: '/test/ffmpeg')
    monkeypatch.setattr(prep.importlib.metadata, 'version', lambda name: '202509' if name == 'espnet' else 'test-version')
    config, provenance = input_files(tmp_path)
    output = tmp_path / 'output'
    plan = prep.preparation_plan(tmp_path, config, provenance, output)
    counts = {'generate': 0, 'render': []}
    def generate(*args, **kwargs):
        counts['generate'] += 1
        rows = [dict(id=f'fileid_{i}', fs=8000, length=80, snr_dB=5,
                     speech_uid='speech', speech_sid='local-speaker', noise_uid='noise',
                     rir_uid='none', augmentation='none') for i in range(2)]
        command = args[0]
        config_path = Path(command[command.index('--config') + 1])
        log_dir = Path(yaml.safe_load(config_path.read_text())['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / 'meta.tsv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(prep.subprocess, 'run', generate)
    module = SimpleNamespace(estimate_early_rir=lambda value: value,
                             wind_noise=lambda *args: None, save_audio=lambda *args: None)
    def process(info, **kwargs):
        counts['render'].append(info['id'])
        wave = np.linspace(-.1, .1, int(info['length']))
        sf.write(info['noisy_path'], wave * .5, int(info['fs']), subtype='FLOAT')
        sf.write(info['clean_path'], wave, int(info['fs']), subtype='FLOAT')
    module.process_one_sample = process
    monkeypatch.setattr(prep, 'load_simulator', lambda source: module)
    return plan, output, module, counts


def test_resume_recovers_completed_rows_and_rerenders_interrupted_pair(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    normal = module.process_one_sample
    def interrupt_second(info, **kwargs):
        normal(info, **kwargs)
        if info['id'] == 'fileid_1':
            raise RuntimeError('simulated interruption before recording success')
    module.process_one_sample = interrupt_second
    with pytest.raises(RuntimeError, match='simulated interruption'):
        prep.execute_preparation(plan)
    assert not (output / 'manifest.jsonl').exists()
    ledger = output / 'render-progress.jsonl'
    assert len(ledger.read_text().splitlines()) == 1
    # A torn last ledger write and existing waveform files do not count as success.
    with ledger.open('a') as handle:
        handle.write('{"index":')
    module.process_one_sample = normal
    manifest = prep.execute_preparation(plan, resume=True)
    assert counts == {'generate': 1, 'render': ['fileid_0', 'fileid_1', 'fileid_1']}
    assert len(manifest.read_text().splitlines()) == 2
    report = json.loads((output / 'provenance.json').read_text())
    assert report['manifest_sha256'] == prep.sha256(manifest)
    assert report['progress_sha256'] == prep.sha256(ledger)
    assert report['metadata_sha256'] == prep.sha256(output / 'log/meta.tsv')
    with pytest.raises(FileExistsError):
        prep.execute_preparation(plan)


def test_resume_validates_saved_pairs_and_rebuilds_corrupt_audio(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    manifest = prep.execute_preparation(plan)
    original = manifest.read_bytes()
    sf.write(output / 'clean/000000000.wav', np.zeros(80), 8000, subtype='FLOAT')
    prep.execute_preparation(plan, resume=True)
    assert counts['generate'] == 1
    assert counts['render'] == ['fileid_0', 'fileid_1', 'fileid_0']
    assert manifest.read_bytes() == original
    # A complete second resume performs validation but no rendering/generation.
    prep.execute_preparation(plan, resume=True)
    assert len(counts['render']) == 3


def test_resume_refuses_changed_plan_metadata_or_source_audio(tmp_path, monkeypatch):
    from copy import deepcopy
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    manifest = prep.execute_preparation(plan)
    changed = deepcopy(plan)
    changed['config']['seed'] = 99
    with pytest.raises(ValueError, match='same saved preparation plan'):
        prep.execute_preparation(changed, resume=True)
    assert manifest.is_file()
    meta = output / 'log/meta.tsv'
    original = meta.read_bytes()
    meta.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='committed checksum'):
        prep.execute_preparation(plan, resume=True)
    assert counts['generate'] == 1
    meta.write_bytes(original)
    sf.write(tmp_path / 'speech.wav', np.zeros(80), 8000, subtype='FLOAT')
    with pytest.raises(ValueError, match='Source audio changed'):
        prep.execute_preparation(plan, resume=True)
    assert not manifest.exists()
    assert (output / 'manifest.previous.jsonl').exists()


def test_resume_does_not_trust_uncommitted_existing_generation(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    prep.execute_preparation(plan)
    (output / 'metadata.json').unlink()
    (output / 'log/generation-complete.json').unlink()
    with pytest.raises(ValueError, match='will not be regenerated or trusted'):
        prep.execute_preparation(plan, resume=True)
    assert counts['generate'] == 1


def test_resume_restores_only_verified_estimated_alignment_evidence(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    plan['rir_alignment'] = {'method': 'upstream-estimated-onset'}
    original_generate = prep.subprocess.run
    def generate(*args, **kwargs):
        original_generate(*args, **kwargs)
        command = args[0]
        config_path = Path(command[command.index('--config') + 1])
        meta = Path(yaml.safe_load(config_path.read_text())['log_dir']) / 'meta.tsv'
        with meta.open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        with meta.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'rir_uid': 'rir'})
    monkeypatch.setattr(prep.subprocess, 'run', generate)
    original_process = module.process_one_sample
    def process(info, **kwargs):
        module.estimate_early_rir(np.array([[0., 0., 1., .5]]), fs=int(info['fs']))
        original_process(info, **kwargs)
    module.process_one_sample = process
    module._realtime_use_rir_onset = lambda wave, level_ratio: 2
    prep.execute_preparation(plan)
    expected = plan['rir_alignment_evidence']
    # Evidence is execution output; it is rebuilt from verified records, not
    # trusted from a caller's possibly stale in-memory plan.
    plan['rir_alignment_evidence'] = {'unused': {'8000': {'delay_samples': 99}}}
    prep.execute_preparation(plan, resume=True)
    assert len(counts['render']) == 2
    assert plan['rir_alignment_evidence'] == expected
    assert plan['rir_alignment_evidence']['rir']['8000']['delay_samples'] == 2


def test_progress_rejects_corrupt_committed_line(tmp_path):
    path = tmp_path / 'progress.jsonl'
    path.write_text('{"index": 0, "record_sha256": "wrong"}\n')
    with pytest.raises(ValueError, match='Corrupt preparation progress'):
        prep._load_progress(path)



def test_resume_restarts_only_uncommitted_interrupted_generation(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    original_generate = prep.subprocess.run
    expected = {}
    def interrupted_generation(*args, **kwargs):
        original_generate(*args, **kwargs)
        command = args[0]
        config_path = Path(command[command.index('--config') + 1])
        log_dir = Path(yaml.safe_load(config_path.read_text())['log_dir'])
        meta = log_dir / 'meta.tsv'
        expected['complete'] = meta.read_bytes()
        meta.write_bytes(expected['complete'][:50])
        raise RuntimeError('generator interrupted before completion')
    monkeypatch.setattr(prep.subprocess, 'run', interrupted_generation)
    with pytest.raises(RuntimeError, match='generator interrupted'):
        prep.execute_preparation(plan)
    assert not (output / 'log/meta.tsv').exists()
    assert not (output / 'metadata.json').exists()
    assert counts['render'] == []
    monkeypatch.setattr(prep.subprocess, 'run', original_generate)
    manifest = prep.execute_preparation(plan, resume=True)
    assert manifest.exists()
    assert (output / 'log/meta.tsv').read_bytes() == expected['complete']
    assert counts['generate'] == 2
    assert list(output.glob('.metadata-interrupted-*'))


def test_resume_recovers_completed_generation_before_promotion(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    original_json = prep._atomic_json
    def interrupt_after_completion(path, value):
        original_json(path, value)
        if path.name == 'generation-complete.json':
            raise RuntimeError('interrupted after generation commit')
    monkeypatch.setattr(prep, '_atomic_json', interrupt_after_completion)
    with pytest.raises(RuntimeError, match='after generation commit'):
        prep.execute_preparation(plan)
    staged = output / '.metadata-generation/log/meta.tsv'
    expected = staged.read_bytes()
    assert not (output / 'log/meta.tsv').exists()
    monkeypatch.setattr(prep, '_atomic_json', original_json)
    prep.execute_preparation(plan, resume=True)
    assert counts['generate'] == 1
    assert (output / 'log/meta.tsv').read_bytes() == expected


def test_resume_repairs_missing_mirror_from_atomic_generation_marker(tmp_path, monkeypatch):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    prep.execute_preparation(plan)
    (output / 'metadata.json').unlink()
    prep.execute_preparation(plan, resume=True)
    assert counts['generate'] == 1
    assert len(counts['render']) == 2
    assert json.loads((output / 'metadata.json').read_text())['sha256'] == prep.sha256(output / 'log/meta.tsv')


def test_raw_metadata_quotes_and_tabs_do_not_swallow_following_utterances(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'verify_source', lambda path: None)
    config, provenance = input_files(tmp_path)
    output = tmp_path / 'quoted-output'
    plan = prep.preparation_plan(tmp_path, config, provenance, output)
    (output / 'log').mkdir(parents=True)
    header = ['id', 'noisy_path', 'speech_uid', 'speech_sid', 'clean_path', 'noise_uid',
              'snr_dB', 'rir_uid', 'augmentation', 'fs', 'length', 'text']
    transcripts = ['"An opening quote, a tab\tand no closing quote',
                   'The next utterance ends with a quote."', '"A complete quoted utterance."']
    expected = [dict(zip(header, [f'fileid_{index}', '/ignored/noisy.wav', 'speech',
        'local-speaker', '/ignored/clean.wav', 'noise', '5', 'none', 'none', '8000', '80', text]))
        for index, text in enumerate(transcripts)]
    metadata = output / 'log/meta.tsv'
    # Match the actual upstream writer: raw tab joins, no CSV escaping.
    metadata.write_text('\t'.join(header) + '\n' + ''.join(
        '\t'.join(row[key] for key in header) + '\n' for row in expected))
    assert prep.read_metadata(metadata) == expected
    called = []
    module = SimpleNamespace(estimate_early_rir=lambda value: value,
                             wind_noise=lambda *args: None, save_audio=lambda *args: None,
                             process_one_sample=lambda info, **kwargs: called.append(info['id']))
    rows = prep.render_metadata(plan, module)
    assert called == ['fileid_0', 'fileid_1', 'fileid_2']
    assert [row['transcript'] for row in rows] == transcripts
    assert [row['speech_uid'] for row in rows] == ['speech'] * 3


@pytest.mark.parametrize('malformed', ['duplicate_header', 'text_not_last', 'missing_field',
    'extra_structural_tab', 'bad_rate', 'bad_length', 'nonfinite_snr', 'duplicate_id', 'invalid_id'])
def test_raw_metadata_rejects_malformed_structural_rows(tmp_path, malformed):
    header = ['id', 'speech_uid', 'speech_sid', 'noise_uid', 'rir_uid', 'snr_dB', 'augmentation', 'fs', 'length']
    fields = ['fileid_1', 'speech', 'speaker', 'noise', 'none', '5', 'none', '8000', '80']
    if malformed == 'duplicate_header':
        header.append('id'); fields.append('fileid_2')
    elif malformed == 'text_not_last':
        header.insert(0, 'text'); fields.insert(0, 'transcript')
    elif malformed == 'missing_field':
        fields.pop()
    elif malformed == 'extra_structural_tab':
        fields[-1] += '\textra'
    elif malformed == 'bad_rate':
        fields[-2] = '8000.0'
    elif malformed == 'bad_length':
        fields[-1] = '0'
    elif malformed == 'nonfinite_snr':
        fields[5] = 'nan'
    elif malformed == 'invalid_id':
        fields[0] = 'arbitrary_sample'
    row = '\t'.join(fields) + '\n'
    metadata = tmp_path / 'bad.tsv'
    metadata.write_text('\t'.join(header) + '\n' + row + (row if malformed == 'duplicate_id' else ''))
    with pytest.raises(ValueError):
        prep.read_metadata(metadata)


def test_threaded_loader_validation_matches_serial_audio_and_is_bounded(tmp_path, monkeypatch):
    import threading
    import torch

    manifest = tmp_path / 'pairs.jsonl'
    rows = []
    for index in range(6):
        wave = np.linspace(-.1, .1, 80 + index).astype(np.float32)
        paths = {key: tmp_path / f'{index}-{key}.wav' for key in ('noisy', 'clean')}
        for key, path in paths.items():
            sf.write(path, wave * (.5 if key == 'noisy' else 1), 8000, subtype='FLOAT')
        rows.append({'id': str(index), 'sample_rate': 8000, 'transcript': f'utterance {index}',
                     **{key: str(path) for key, path in paths.items()}})
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    original_bytes = manifest.read_bytes()
    original_getitem = prep.PairedAudioDataset.__getitem__
    dataset = prep.PairedAudioDataset(manifest, prep.RATE_SET, training=False, load_teacher=False)
    def summarize(item):
        return (item['id'], item['transcript'], item['length'], item['sample_rate'],
                item['noisy'].numpy().tobytes(), item['clean'].numpy().tobytes())
    expected = {index: summarize(dataset[index]) for index in range(len(dataset))}
    serial = prep._validate_loader(manifest)
    observed, threads = {}, set()
    rendezvous = threading.Barrier(2)
    def inspect(self, index):
        # All inputs are immutable; inference must not build/use augmentation or
        # crops, and each pair is independently decoded by the actual loader.
        assert self.training is False and self.load_teacher is False
        assert self.crop_samples is None and self.crop_seconds is None
        assert all(value is None for value in self.augmentations.values())
        rendezvous.wait(timeout=5)
        threads.add(threading.get_ident())
        item = original_getitem(self, index)
        observed[index] = summarize(item)
        return item
    monkeypatch.setattr(prep.PairedAudioDataset, '__getitem__', inspect)
    rng = torch.get_rng_state().clone()
    parallel = prep._validate_loader(manifest, workers=2)
    assert observed == expected
    assert len(threads) == 2
    assert torch.equal(torch.get_rng_state(), rng)
    assert manifest.read_bytes() == original_bytes
    assert serial['workers'] == 1 and serial['executor'] == 'in_process'
    assert parallel['workers'] == 2 and parallel['executor'] == 'threads'
    assert serial['examples'] == parallel['examples'] == 6
    assert parallel['max_in_flight'] == 4
    assert 0 <= parallel['construction_seconds'] <= parallel['total_seconds']
    assert 0 <= parallel['decode_seconds'] <= parallel['total_seconds']


@pytest.mark.parametrize('workers', [1, 2])
def test_loader_validation_failure_never_publishes_manifest(tmp_path, monkeypatch, workers):
    plan, output, module, counts = resumable_execution(tmp_path, monkeypatch)
    # Keep the upstream renderer fixture in-process while exercising the actual
    # threaded loader and publication path with genuine saved audio files.
    original_render = prep.render_metadata
    def render(*args, workers=1, **kwargs):
        return original_render(*args, workers=1, **kwargs)
    monkeypatch.setattr(prep, 'render_metadata', render)
    final = prep.execute_preparation(plan, workers=workers)
    original_manifest = final.read_bytes()
    report = json.loads((output / 'provenance.json').read_text())
    assert report['loader_validation']['workers'] == workers
    assert report['loader_validation']['examples'] == 2
    assert report['loader_validation']['total_seconds'] >= 0
    original_getitem = prep.PairedAudioDataset.__getitem__
    def corrupt_after_header_checks(self, index):
        if index == 1:
            sf.write(self.records[index]['clean'], np.full(80, np.nan), 8000, subtype='FLOAT')
        return original_getitem(self, index)
    monkeypatch.setattr(prep.PairedAudioDataset, '__getitem__', corrupt_after_header_checks)
    with pytest.raises(ValueError, match='non-finite'):
        prep.execute_preparation(plan, workers=workers, resume=True)
    assert not final.exists()
    assert (output / 'manifest.previous.jsonl').read_bytes() == original_manifest
    assert (output / 'manifest.pending.jsonl').is_file()
    assert counts['render'] == ['fileid_0', 'fileid_1']
    # A subsequent valid resume repairs the corrupted pair and can publish.
    monkeypatch.setattr(prep.PairedAudioDataset, '__getitem__', original_getitem)
    assert prep.execute_preparation(plan, workers=workers, resume=True).read_bytes() == original_manifest
    assert counts['render'] == ['fileid_0', 'fileid_1', 'fileid_1']
