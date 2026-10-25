"""Optional actual-URGENT preparation preflight, using only synthetic signals.

Set REALTIME_USE_URGENT_SOURCE to the pinned checkout. No training is invoked,
no real corpus is processed, and network access is blocked during import.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import pytest
import soundfile as sf

from realtime_use import prepare_data as prep


@pytest.fixture
def official(tmp_path, monkeypatch):
    location = os.environ.get('REALTIME_USE_URGENT_SOURCE')
    if not location:
        pytest.skip('Set REALTIME_USE_URGENT_SOURCE for the optional upstream preparation preflight')
    source = Path(location)
    prep.verify_source(source)
    def forbid(*args, **kwargs):
        raise RuntimeError('Network access is forbidden in simulator preflight')
    monkeypatch.setattr(socket.socket, 'connect', forbid)
    previous = sys.modules.get('espnet2.train.preprocessor')
    module = prep.load_simulator(source)
    assert sys.modules.get('espnet2.train.preprocessor') is previous
    monkeypatch.chdir(tmp_path)
    return module


def test_vendored_function_block_hash_is_unchanged():
    import realtime_use._vendor.espnet_detection as detection
    source = Path(detection.__file__)
    metadata = json.loads(source.with_name('SOURCE.json').read_text())
    functions = source.read_text().split('def framing(', 1)[1]
    assert hashlib.sha256(('def framing(' + functions).encode()).hexdigest() == metadata['function_block_sha256']


def test_vendored_functions_match_installed_pinned_upstream(official):
    import importlib.util
    import realtime_use._vendor.espnet_detection as detection
    spec = importlib.util.find_spec('espnet2')
    upstream = Path(spec.submodule_search_locations[0]) / 'train/preprocessor.py'
    text = upstream.read_text()
    block = text[text.index('def framing('):text.index('\ndef any_allzero')]
    metadata = json.loads(Path(detection.__file__).with_name('SOURCE.json').read_text())
    assert hashlib.sha256(block.encode()).hexdigest() == metadata['function_block_sha256']


@pytest.mark.parametrize('augmentation', ['none', 'clipping(min=0.05,max=0.95)',
    'bandwidth_limitation-polyphase->8000',
    'packet_loss(packet_loss_indices=[1,3],packet_duration_ms=20)'])
def test_actual_official_noise_reverb_and_distortions_with_anechoic_target(official, tmp_path, augmentation):
    rate, count, delay = 16000, 3200, 37
    time = np.arange(count) / rate
    speech = (.1 * np.sin(2 * np.pi * 300 * time)).astype(np.float64)
    noise = np.random.default_rng(9).normal(0, .02, count)
    rir = np.zeros(256); rir[delay] = 1; rir[delay + 80] = .4
    for name, audio in [('speech', speech), ('noise', noise), ('rir', rir)]:
        sf.write(tmp_path / f'{name}.wav', audio, rate, subtype='FLOAT')
    info = dict(id='fileid_1', fs=rate, snr_dB=10, speech_uid='speech', noise_uid='noise',
                rir_uid='rir', length=count, augmentation=augmentation,
                clean_path=str(tmp_path/'clean.wav'), noisy_path=str(tmp_path/'noisy.wav'))
    with prep.rendering_adapters(official, delay=delay, wind_clipping=False):
        official.process_one_sample(info, speech_dic={'speech':str(tmp_path/'speech.wav')},
             noise_dic={'noise':str(tmp_path/'noise.wav')}, rir_dic={'rir':str(tmp_path/'rir.wav')})
    clean, _ = sf.read(tmp_path/'clean.wav')
    noisy, _ = sf.read(tmp_path/'noisy.wav')
    assert len(clean) == len(noisy) == count
    assert np.isfinite(noisy).all()
    np.testing.assert_allclose(clean[:delay], np.zeros(delay), atol=1e-12)
    delayed = np.pad(speech, (delay, 0))[:count]
    gain = np.dot(clean, delayed) / np.dot(delayed, delayed)
    np.testing.assert_allclose(clean, gain * delayed, atol=1e-7)
    assert not np.allclose(noisy, clean)


@pytest.mark.parametrize('kind', ['codec', 'wind_noise'])
def test_actual_official_codec_and_wind_preflight(official, tmp_path, kind):
    import shutil
    from torchaudio.utils import ffmpeg_utils
    assert shutil.which('ffmpeg'), 'Install FFmpeg executable and compatible shared libraries before preparation'
    assert ffmpeg_utils.get_versions(), 'Torchaudio AudioEffector needs its FFmpeg extension'
    rate, count = 16000, 8000
    speech = .1 * np.sin(2 * np.pi * 300 * np.arange(count) / rate)
    noise = np.random.default_rng(5).normal(0, .02, count)
    for name, audio in [('speech', speech), ('noise', noise)]:
        sf.write(tmp_path/f'{name}.wav', audio, rate, subtype='FLOAT')
    noise_uid = 'wind_noise_1' if kind == 'wind_noise' else 'noise'
    augmentation = ('wind_noise(threshold=0.2,ratio=2,attack=10,release=20,sc_gain=1,clipping=False,clipping_threshold=0.9)'
                    if kind == 'wind_noise' else 'codec(format=mp3,encoder=None,qscale=3)')
    info = dict(id='fileid_2', fs=rate, snr_dB=10, speech_uid='speech', noise_uid=noise_uid,
                rir_uid='none', length=count, augmentation=augmentation,
                clean_path=str(tmp_path/'clean.wav'), noisy_path=str(tmp_path/'noisy.wav'))
    with prep.rendering_adapters(official, delay=None, wind_clipping=False):
        official.process_one_sample(info, speech_dic={'speech':str(tmp_path/'speech.wav')},
                                    noise_dic={noise_uid:str(tmp_path/'noise.wav')}, rir_dic={})
    clean, _ = sf.read(tmp_path/'clean.wav')
    noisy, _ = sf.read(tmp_path/'noisy.wav')
    assert len(clean) == len(noisy) == count
    assert np.isfinite(noisy).all()
    gain = np.dot(clean, speech) / np.dot(speech, speech)
    np.testing.assert_allclose(clean, gain * speech, atol=1e-7)
