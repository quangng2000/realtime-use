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
