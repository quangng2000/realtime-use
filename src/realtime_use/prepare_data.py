"""Pinned URGENT simulator integration; never downloads corpora or starts training.

The upstream generator chooses seven degradations. Rendering delegates to its
implementation, with explicit anechoic-target and upstream wind-bug adapters.
Use a separate preparation environment with the upstream dependencies installed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from .data import PairedAudioDataset

URGENT_COMMIT = "daf1730cc11bf450d05c2d9e1d8bb3afdd63c427"
URGENT_URL = "https://github.com/urgent-challenge/urgent2025_challenge"
ROLES = {"speech_scps": "speech", "noise_scps": "noise", "rir_scps": "rir", "wind_noise_scps": "wind"}
RATE_SET = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source: Path) -> None:
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != URGENT_COMMIT:
        raise ValueError(f"Use the audited URGENT source revision {URGENT_COMMIT}, found {revision}")
    # Verify the bytes actually imported/executed, independent of checkout
    # index state or optional metric submodules absent from a partial checkout.
    for relative in ('LICENSE', 'simulation/generate_data_param.py',
                     'simulation/simulate_data_from_param.py', 'simulation/rir_utils.py'):
        expected = subprocess.check_output(['git', '-C', str(source), 'show', f'{URGENT_COMMIT}:{relative}'])
        if (source / relative).read_bytes() != expected:
            raise ValueError(f'The pinned URGENT source differs at {relative}')
    if 'Apache License' not in (source / 'LICENSE').read_text():
        raise ValueError("Missing expected upstream Apache license")


def read_scp(path: Path) -> dict[str, tuple[int, Path]]:
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: expected uid sample_rate absolute_audio_path (no shell pipelines)")
        uid, rate, filename = parts
        audio = Path(filename).expanduser()
        if uid in records or not audio.is_absolute():
            raise ValueError(f"{path}:{line_number}: use unique IDs and absolute audio paths")
        info = sf.info(audio)
        if info.samplerate != int(rate) or info.channels != 1 or info.frames < 1:
            raise ValueError(f"{path}:{line_number}: declared-rate, mono, nonempty input required")
        records[uid] = (int(rate), audio.resolve())
    if not records:
        raise ValueError(f"{path}: empty SCP")
    return records


def preparation_plan(source: Path, config_path: Path, provenance_path: Path, output: Path,
                     *, usage: str = 'commercial', rir_alignment: str = 'audited') -> dict[str, Any]:
    """Validate local input metadata and produce a reproducible, nonexecuting plan."""
    source, config_path, provenance_path, output = (path.expanduser().resolve() for path in
                                                    (source, config_path, provenance_path, output))
    if usage not in {'commercial', 'research'}:
        raise ValueError('usage must be commercial or research')
    if rir_alignment not in {'audited', 'upstream-estimated-onset'}:
        raise ValueError('rir_alignment must be audited or upstream-estimated-onset')
    verify_source(source)
    config = yaml.safe_load(config_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("Simulation config and provenance must be mappings")
    if provenance.get('split') not in {'train', 'validation', 'test'}:
        raise ValueError("Provenance split must be train, validation, or test")
    sources = provenance.get('sources', [])
    if not sources:
        raise ValueError("Provenance must document each input SCP source")
    by_scp = {}
    for entry in sources:
        required = ('id', 'role', 'scp', 'license', 'license_evidence', 'commercial_use')
        if not isinstance(entry, dict) or any(key not in entry for key in required):
            raise ValueError(f"Each source requires {required}")
        if not entry['license'] or not entry['license_evidence']:
            raise ValueError('Every source requires its actual license and license evidence')
        if entry['commercial_use'] is not None and not isinstance(entry['commercial_use'], bool):
            raise ValueError('commercial_use must be true, false, or null (unreviewed)')
        if usage == 'commercial':
            if entry['commercial_use'] is not True:
                raise ValueError("Preparation requires documented commercial-use permission for each source")
            if re.search(r'non.?commercial|cc[- ]by[- ]nc', str(entry['license']), re.I):
                raise ValueError("Noncommercial sources are incompatible with this preparation route")
        scp = Path(entry['scp']).expanduser()
        scp = scp if scp.is_absolute() else provenance_path.parent / scp
        scp = scp.resolve(strict=True)
        if str(scp) in by_scp:
            raise ValueError("Duplicate source SCP in provenance")
        by_scp[str(scp)] = {**entry, 'scp': str(scp), 'sha256': sha256(scp)}
    inventories = {}
    for field, role in ROLES.items():
        paths = config.get(field) or []
        if not paths:
            raise ValueError(f"Full seven-degradation preparation requires {field}")
        resolved, inventory = [], {}
        for filename in paths:
            path = Path(filename).expanduser()
            path = path if path.is_absolute() else config_path.parent / path
            path = path.resolve(strict=True)
            entry = by_scp.get(str(path))
            if entry is None or entry['role'] != role:
                raise ValueError(f"Missing matching {role} provenance for {path}")
            for uid, (rate, audio) in read_scp(path).items():
                if uid in inventory:
                    raise ValueError(f"Duplicate {role} uid: {uid}")
                if role == 'speech' and rate not in RATE_SET:
                    raise ValueError(f"Unsupported speech sample rate {rate}")
                if role == 'wind' and not uid.startswith('wind_noise'):
                    raise ValueError("Official wind IDs must start with wind_noise")
                inventory[uid] = {'sample_rate': rate, 'path': str(audio), 'source_id': entry['id']}
            resolved.append(str(path))
        config[field] = resolved
        inventories[role] = inventory
    for rate in {entry['sample_rate'] for entry in inventories['speech'].values()}:
        for role in ('noise', 'wind', 'rir'):
            if not any(entry['sample_rate'] >= rate for entry in inventories[role].values()):
                raise ValueError(f'No {role} source at or above speech rate {rate}')
    if set(inventories['noise']) & set(inventories['wind']):
        raise ValueError('Noise and wind IDs must be disjoint')
    for field in ('speech_utt2spk', 'speech_text'):
        resolved = []
        for filename in config.get(field) or []:
            path = Path(filename).expanduser()
            resolved.append(str((path if path.is_absolute() else config_path.parent / path).resolve(strict=True)))
        config[field] = resolved
    if len(config['speech_utt2spk']) != len(config['speech_scps']):
        raise ValueError('Provide one speech_utt2spk per speech SCP')
    speakers = {}
    for filename in config['speech_utt2spk']:
        for line in Path(filename).read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2 or fields[0] in speakers:
                raise ValueError('Speaker metadata must have unique utterance IDs and one speaker ID')
            speakers[fields[0]] = fields[1]
    if set(speakers) != set(inventories['speech']):
        raise ValueError('Speaker metadata must cover exactly the speech inventory')
    augmentations = config.get('augmentations', {})
    if not {'clipping', 'bandwidth_limitation', 'codec', 'packet_loss'} <= augmentations.keys():
        raise ValueError('Use official config entries for clipping, bandwidth_limitation, codec, and packet_loss')
    if any(float(augmentations[key].get('weight', 0)) <= 0 for key in ('clipping', 'bandwidth_limitation', 'codec', 'packet_loss')):
        raise ValueError('All four additional degradation weights must be positive')
    count_probabilities = config.get('num_augmentations', {})
    if not any(int(count) > 0 and float(chance) > 0 for count, chance in count_probabilities.items()):
        raise ValueError('num_augmentations must allow a nonzero number of additional distortions')
    probability = float(config.get('prob_reverberation', .5))
    if not 0 < probability < 1 or not 0 < float(config.get('prob_wind_noise', 0)) < 1:
        raise ValueError('Use strictly positive probabilities below one for reverberation and wind')
    delays = provenance.get('rir_direct_path_samples', {})
    speech_rates = sorted({entry['sample_rate'] for entry in inventories['speech'].values()})
    for uid, entry in inventories['rir'].items():
        # The official generator may select any RIR at an equal/higher rate.
        for rate in speech_rates:
            if rir_alignment == 'audited' and entry['sample_rate'] >= rate:
                delay = delays.get(uid, {}).get(str(rate))
                if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
                    raise ValueError(f'Provide audited direct-path delay for RIR {uid} at rendering rate {rate}')
    config.update(output_dir=str(output), log_dir=str(output / 'log'), out_format='wav', store_noise=False)
    return {'format_version': 2, 'source': str(source), 'source_url': URGENT_URL,
            'source_commit': URGENT_COMMIT, 'config': config, 'inventories': inventories,
            'provenance': {**provenance, 'sources': list(by_scp.values())},
            'input_config_sha256': sha256(config_path), 'output_dir': str(output),
            'usage': usage,
            'commercial_clearance': 'documented' if usage == 'commercial' else 'pending',
            'usage_note': ('Source permission records are caller attestations, not legal verification.' if usage == 'commercial'
                           else 'Research preparation only; source licenses are retained. This mode does not establish research or commercial rights.'),
            'rir_alignment': {'method': rir_alignment,
                              'physical_direct_path_audited': rir_alignment == 'audited',
                              **({'estimator': 'URGENT simulation/rir_utils.py:get_rir_start_sample',
                                  'level_ratio': 0.1, 'source_commit': URGENT_COMMIT,
                                  'note': 'First 10%-of-peak threshold crossing is an onset estimate, not a verified physical direct-path arrival.'}
                                 if rir_alignment == 'upstream-estimated-onset' else {})},
            'target': ('anechoic_unit_gain_direct_path_delayed_same_joint_normalization' if rir_alignment == 'audited'
                       else 'anechoic_unit_gain_estimated_onset_delayed_same_joint_normalization'),
            'upstream_probability_note': f'Pinned generator omits RIR when U <= prob_reverberation; effective reverb probability is {1-probability:g}. No silent rewrite.',
            'degradation_coverage': 'Seven families enabled; actual sampled coverage is reported after rendering, not guaranteed by a finite random draw.'}


def direct_path_impulse(rir: np.ndarray, delay: int) -> np.ndarray:
    if rir.ndim != 2 or isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay < rir.shape[-1]:
        raise ValueError('Audited direct-path delay must fit the rendered RIR')
    impulse = np.zeros_like(rir)
    impulse[:, delay] = 1
    return impulse


@contextmanager
def rendering_adapters(module, *, delay: int | None, wind_clipping: bool,
                       rir_alignment: str = 'audited'):
    """Small scoped changes to upstream target semantics and two wind issues.

    No degradation formula is copied. Upstream bool('False') is corrected using
    the metadata value, and its wind routine receives a copy to protect targets.
    Float WAV output avoids introducing PCM16 quantization into clean targets.
    """
    if rir_alignment not in {'audited', 'upstream-estimated-onset'}:
        raise ValueError('Unknown RIR alignment mode')
    evidence = {'method': rir_alignment, 'physical_direct_path_audited': rir_alignment == 'audited'}
    previous = {name: getattr(module, name) for name in ('estimate_early_rir', 'wind_noise', 'save_audio')}
    def target_rir(rir, **kwargs):
        actual_delay = delay
        if rir_alignment == 'upstream-estimated-onset':
            # This callback receives the exact mono/resampled RIR used by the
            # official convolution. Do not estimate before resampling and scale
            # an index, which need not match the actual rendered waveform.
            if rir.ndim != 2 or rir.shape[0] != 1 or not np.isfinite(rir).all() or not np.any(rir):
                raise ValueError('Onset estimation requires a finite nonzero mono RIR')
            estimator = getattr(module, '_realtime_use_rir_onset', None)
            if estimator is None:
                raise ValueError('Load the pinned official URGENT onset estimator')
            actual_delay = int(estimator(rir[0], level_ratio=0.1))
            evidence.update(estimator='URGENT simulation/rir_utils.py:get_rir_start_sample',
                            level_ratio=0.1, source_commit=URGENT_COMMIT)
        if actual_delay is None:
            raise ValueError('Missing audited direct-path delay')
        impulse = direct_path_impulse(rir, actual_delay)
        rendered = np.ascontiguousarray(rir, dtype='<f8')
        evidence.update(delay_samples=actual_delay, sample_rate=int(kwargs.get('fs', 48000)),
                        rendered_rir_sha256=hashlib.sha256(rendered.tobytes()).hexdigest(),
                        rendered_rir_hash_format='mono float64 little-endian C-order sample bytes',
                        rendered_rir_frames=rir.shape[-1])
        return impulse
    def wind(*args, **kwargs):
        import inspect
        bound = inspect.signature(previous['wind_noise']).bind(*args, **kwargs)
        bound.arguments['speech_sample'] = bound.arguments['speech_sample'].copy()
        bound.arguments['clipping'] = wind_clipping
        return previous['wind_noise'](*bound.args, **bound.kwargs)
    def save(audio, filename, fs):
        audio = np.asarray(audio)
        if not np.isfinite(audio).all():
            raise ValueError('Official simulator produced nonfinite audio; inspect silent/invalid sources')
        sf.write(filename, audio[0] if audio.ndim == 2 else audio, fs, subtype='FLOAT')
    module.estimate_early_rir, module.wind_noise, module.save_audio = target_rir, wind, save
    try:
        yield evidence
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_progress(path: Path) -> dict[int, dict[str, Any]]:
    """Recover successful rows; discard only a final unterminated write."""
    completed = {}
    if not path.exists():
        return completed
    with path.open('r+b') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(offset)
                break
            try:
                record = json.loads(line)
                checksum = record.pop('record_sha256')
                if checksum != _json_hash(record):
                    raise ValueError('checksum differs')
                index = record['index']
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError('invalid row index')
                completed[index] = record
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Corrupt preparation progress at byte {offset}; no records reused') from exc
    return completed


def _validate_pair(row: dict[str, Any], frames: int) -> dict[str, str]:
    """Check decoded content as well as bytes before recording/reusing a pair."""
    hashes = {}
    for key in ('noisy', 'clean'):
        path = Path(row[key])
        wave, rate = sf.read(path, dtype='float32', always_2d=True)
        if rate != row['sample_rate'] or wave.shape != (frames, 1) or frames < 1 or not np.isfinite(wave).all():
            raise ValueError(f'Invalid rendered {key}: mono finite audio at the declared rate/length required')
        hashes[key] = sha256(path)
    return hashes


def read_metadata(path: Path) -> list[dict[str, str]]:
    """Read the pinned generator's raw TSV, which does not use CSV quoting.

    Upstream joins structural columns with tabs and appends unescaped transcript
    text last. A quote in that text must never consume a following physical row;
    additional tabs in the final text field belong to the transcript itself.
    """
    required = {'id', 'speech_uid', 'speech_sid', 'noise_uid', 'rir_uid',
                'snr_dB', 'augmentation', 'fs', 'length'}
    allowed = required | {'noisy_path', 'clean_path', 'noise_path', 'text'}
    rows, seen = [], set()
    def without_ending(line):
        if line.endswith('\n'):
            line = line[:-1]
            if line.endswith('\r'):
                line = line[:-1]
        return line
    with Path(path).open(encoding='utf-8', newline='\n') as handle:
        header = without_ending(handle.readline()).split('\t')
        if (len(header) != len(set(header)) or not required <= set(header)
                or set(header) - allowed or ('text' in header and header[-1] != 'text')):
            raise ValueError(f'{path}: invalid raw URGENT metadata header')
        for line_number, line in enumerate(handle, 2):
            fields = without_ending(line).split('\t', maxsplit=len(header) - 1)
            if len(fields) != len(header):
                raise ValueError(f'{path}:{line_number}: expected {len(header)} raw TSV fields')
            row = dict(zip(header, fields))
            for key, value in row.items():
                if key == 'text':
                    if any(ord(character) < 32 and character != '\t' for character in value):
                        raise ValueError(f'{path}:{line_number}: transcript contains unsupported line/control characters')
                elif not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
                    raise ValueError(f'{path}:{line_number}: malformed structural field {key}')
            for key in ('id', 'speech_uid', 'speech_sid', 'noise_uid', 'rir_uid'):
                if any(character.isspace() for character in row[key]):
                    raise ValueError(f'{path}:{line_number}: {key} must be one token')
            if not re.fullmatch(r'fileid_[0-9]+', row['id']) or row['id'] in seen:
                raise ValueError(f'{path}:{line_number}: invalid or duplicate official sample ID')
            if any(not re.fullmatch(r'[0-9]+', row[key]) or int(row[key]) <= 0 for key in ('fs', 'length')):
                raise ValueError(f'{path}:{line_number}: fs and length must be positive integers')
            try:
                finite_snr = math.isfinite(float(row['snr_dB']))
            except ValueError:
                finite_snr = False
            if not finite_snr:
                raise ValueError(f'{path}:{line_number}: snr_dB must be finite')
            seen.add(row['id'])
            rows.append(row)
    return rows


def _render_row(plan, module, index, original, mappings, hashes, saved=None, *, validate=True):
    """Render/validate one disjoint pair; never mutate the shared progress ledger."""
    output = Path(plan['output_dir'])
    def source_hash(filename):
        if filename not in hashes:
            hashes[filename] = sha256(Path(filename))
        return hashes[filename]
    info = dict(original)
    rate = int(info['fs'])
    # Do not trust output paths from TSV; confine writes to this new run.
    for key, directory in (('noisy_path', 'noisy'), ('clean_path', 'clean')):
        destination = output / directory / f'{index:09d}.wav'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(output.resolve()) or destination.is_symlink():
            raise ValueError('Rendered output paths must remain inside the preparation directory')
        info[key] = str(destination)
    rir_id = info['rir_uid']
    alignment = plan.get('rir_alignment', {}).get('method', 'audited')
    delay = (plan['provenance']['rir_direct_path_samples'][rir_id][str(rate)]
             if rir_id != 'none' and alignment == 'audited' else None)
    source_hashes = {
        'source_speech_sha256': source_hash(mappings['speech'][info['speech_uid']]),
        'source_noise_sha256': source_hash(mappings['noise'][info['noise_uid']]),
    }
    rir_hash = source_hash(mappings['rir'][rir_id]) if rir_id != 'none' else None
    if saved is not None:
        if saved['metadata_row_sha256'] != _json_hash(original):
            raise ValueError(f'Generated metadata changed for completed row {index}')
        row = saved['row']
        if (row['noisy'] != info['noisy_path'] or row['clean'] != info['clean_path']
                or row['sample_rate'] != rate):
            raise ValueError(f'Completed row {index} has inconsistent output paths/rate')
        if any(row.get(key) != value for key, value in source_hashes.items()):
            raise ValueError(f'Source audio changed since completed row {index}')
        if rir_hash is not None and row['rir_alignment'].get('source_rir_sha256') != rir_hash:
            raise ValueError(f'Source RIR changed since completed row {index}')
        try:
            reusable = _validate_pair(row, int(info['length'])) == saved['output_sha256']
        except (OSError, RuntimeError, ValueError):
            reusable = False
        if reusable:
            return row, None
    wind_match = re.search(r'clipping=(True|False)(?:,|\))', info['augmentation'])
    with rendering_adapters(module, delay=delay, wind_clipping=bool(wind_match and wind_match[1] == 'True'),
                            rir_alignment=alignment) as alignment_evidence:
        module.process_one_sample(info, force_1ch=True, store_noise=False,
                                  speech_dic=mappings['speech'], noise_dic=mappings['noise'], rir_dic=mappings['rir'])
    row = {'id': f"{plan['provenance']['split']}:{info['id']}",
           'noisy': info['noisy_path'], 'clean': info['clean_path'], 'sample_rate': rate,
           'split': plan['provenance']['split'], 'speech_uid': info['speech_uid'],
           'speaker_id': info['speech_sid'],
           'noise_uid': info['noise_uid'], 'rir_uid': rir_id,
           'alignment_delay_samples': alignment_evidence.get('delay_samples', delay or 0),
           **({'direct_path_delay_samples': delay or 0} if alignment == 'audited' else
              {'estimated_onset_delay_samples': alignment_evidence.get('delay_samples', 0)}),
           'rir_alignment': alignment_evidence if rir_id != 'none' else {'method': 'none', 'delay_samples': 0},
           'usage': plan.get('usage', 'commercial'),
           'commercial_clearance': plan.get('commercial_clearance', 'documented'),
           'augmentation': info['augmentation'], 'target': plan['target'],
           **({'transcript': info['text']} if info.get('text') not in (None, '<not-available>') else {}),
           'upstream_commit': URGENT_COMMIT,
           **source_hashes,
           'provenance': str(output / 'provenance.json')}
    if rir_id != 'none':
        alignment_evidence['source_rir_sha256'] = rir_hash
    record = None
    if validate:
        record = {'index': index, 'metadata_row_sha256': _json_hash(original),
                  'row': row, 'output_sha256': _validate_pair(row, int(info['length']))}
        record['record_sha256'] = _json_hash(record)
    return row, record


_RENDER_WORKER = None


def _render_worker_init(plan, scratch):
    """Spawn initializer: one official module and isolated wind workspace per CPU worker."""
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_source(Path(plan['source']))
    module = load_simulator(Path(plan['source']))
    mappings = {role: {uid: entry['path'] for uid, entry in entries.items()}
                for role, entries in plan['inventories'].items()}
    mappings['noise'].update(mappings['wind'])
    directory = tempfile.mkdtemp(prefix=f'worker-{os.getpid()}-', dir=scratch)
    os.chdir(directory)
    global _RENDER_WORKER
    _RENDER_WORKER = plan, module, mappings, {}


def _render_worker_task(task):
    plan, module, mappings, hashes = _RENDER_WORKER
    index, original, saved, validate = task
    row, record = _render_row(plan, module, index, original, mappings, hashes, saved, validate=validate)
    return row, record, os.getpid()


@contextmanager
def _spawn_cpu_environment():
    # Spawned imports see these limits before NumPy/Torch initialize native
    # pools. Restore the caller's environment after all children have exited.
    values = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                    'NUMEXPR_NUM_THREADS')}
    values['CUDA_VISIBLE_DEVICES'] = ''
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _graceful_sigterm():
    """Unwind a multiprocessing owner before releasing its output lock.

    SIGTERM normally skips Python finally blocks. Convert the first signal to
    SystemExit, then ignore repeats while executor shutdown joins the children.
    This does not make SIGKILL safe. Only the main thread may own signal handlers.
    Nested preparation contexts keep the outer handler installed through cleanup.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Multiprocess preparation must run on the main thread for graceful SIGTERM cleanup')
    previous = signal.getsignal(signal.SIGTERM)
    if getattr(previous, '_realtime_use_graceful_sigterm', False):
        yield
        return
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)
    terminate._realtime_use_graceful_sigterm = True
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _worker_count(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    return workers


def render_metadata(plan: dict[str, Any], module, *, progress_path: Path | None = None,
                    workers: int = 1, execution: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Render official metadata with bounded CPU work and an ordered parent ledger.

    Worker count is execution policy, not a recipe setting: it may change on
    resume without changing any sampled augmentation parameters or output IDs.
    Audio stays in files; only metadata, hashes and alignment evidence cross IPC.
    """
    workers = _worker_count(workers)
    output = Path(plan['output_dir'])
    mappings = {role: {uid: entry['path'] for uid, entry in entries.items()}
                for role, entries in plan['inventories'].items()}
    mappings['noise'].update(mappings['wind'])
    completed = _load_progress(progress_path) if progress_path is not None else {}
    metadata = read_metadata(output / 'log/meta.tsv')
    if any(index >= len(metadata) for index in completed):
        raise ValueError('Preparation progress refers to metadata rows that do not exist')
    rows = []
    stats = execution if execution is not None else {}
    stats.update(workers=workers, worker_pids=[], rendered=0, reused=0, max_in_flight=0)
    def commit(result):
        row, record = result[:2]
        pid = result[2] if len(result) == 3 else os.getpid()
        if pid not in stats['worker_pids']:
            stats['worker_pids'].append(pid)
        stats['reused' if record is None and progress_path is not None else 'rendered'] += 1
        if row['rir_uid'] != 'none':
            plan.setdefault('rir_alignment_evidence', {}).setdefault(row['rir_uid'], {})[
                str(row['sample_rate'])] = dict(row['rir_alignment'])
        if progress_path is not None and record is not None:
            with progress_path.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        rows.append(row)

    if workers == 1:
        hashes = {}
        for index, original in enumerate(metadata):
            commit(_render_row(plan, module, index, original, mappings, hashes,
                               completed.get(index), validate=progress_path is not None))
        return rows
    if not metadata:
        return rows
    # Transient FFmpeg/wind files stay on local temporary storage, even when
    # final pairs live on a remote filesystem. Each child has a separate cwd.
    # Bounded submission also bounds completed results waiting behind a slower
    # earlier row. Metadata-order commit keeps journals/manifests deterministic.
    worker_plan = {key: value for key, value in plan.items() if key != 'rir_alignment_evidence'}
    with _graceful_sigterm(), _spawn_cpu_environment(), tempfile.TemporaryDirectory(prefix='realtime-use-render-') as scratch:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_render_worker_init, initargs=(worker_plan, scratch))
        tasks = iter(enumerate(metadata))
        pending = deque()
        try:
            def submit_one():
                item = next(tasks, None)
                if item is None:
                    return False
                index, original = item
                task = (index, original, completed.get(index), progress_path is not None)
                pending.append(pool.submit(_render_worker_task, task))
                stats['max_in_flight'] = max(stats['max_in_flight'], len(pending))
                return True
            for _ in range(min(len(metadata), 2 * workers)):
                submit_one()
            while pending:
                commit(pending.popleft().result())
                submit_one()
        finally:
            # Only a bounded amount of uncommitted work can complete on failure.
            # Such files are deliberately re-rendered on the next resume.
            pool.shutdown(wait=True, cancel_futures=True)
    return rows


def load_simulator(source: Path):
    """Import pinned official renderer without unrelated ESPnet NLP downloads.

    Only its detect_non_silence import is redirected to byte-identical vendored
    ESPnet functions. All module substitutions are restored immediately.
    """
    from ._vendor.espnet_detection import detect_non_silence

    names = ('espnet2.train.preprocessor', 'generate_data_param', 'rir_utils')
    originals = {name: sys.modules.get(name) for name in names}
    detection = ModuleType('espnet2.train.preprocessor')
    detection.detect_non_silence = detect_non_silence
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, source / 'simulation' / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    try:
        sys.modules['espnet2.train.preprocessor'] = detection
        load('generate_data_param', 'generate_data_param.py')
        rir_module = load('rir_utils', 'rir_utils.py')
        name = '_realtime_use_urgent_renderer'
        previous_renderer = sys.modules.get(name)
        try:
            renderer = load(name, 'simulate_data_from_param.py')
            renderer._realtime_use_rir_onset = rir_module.get_rir_start_sample
            return renderer
        finally:
            if previous_renderer is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_renderer
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _base_plan(plan: dict[str, Any]) -> dict[str, Any]:
    # JSON normalizes integer YAML mapping keys (e.g. augmentation counts).
    return json.loads(json.dumps({key: value for key, value in plan.items() if key != 'rir_alignment_evidence'}))


def _ensure_metadata(plan: dict[str, Any]) -> Path:
    """Commit generated logs as one directory so interrupted draws are resumable."""
    output, source = Path(plan['output_dir']), Path(plan['source'])
    log_dir = output / 'log'
    metadata_path = log_dir / 'meta.tsv'
    mirror = output / 'metadata.json'
    staging = output / '.metadata-generation'
    identity = _json_hash(_base_plan(plan))

    def completion(directory):
        marker = directory / 'generation-complete.json'
        if not marker.exists():
            return None
        record = json.loads(marker.read_text())
        if record.get('plan_sha256') != identity or record.get('sha256') != sha256(directory / 'meta.tsv'):
            raise ValueError('Generated metadata differs from its committed checksum or plan')
        return record

    def quarantine(directory):
        # Keep interrupted bytes for inspection; never mistake their existence
        # for a successful generator exit or replace them silently.
        archive = Path(tempfile.mkdtemp(prefix='.metadata-interrupted-', dir=output))
        directory.replace(archive / directory.name)

    if metadata_path.exists():
        committed = completion(log_dir)
        if committed is not None:
            if mirror.exists() and json.loads(mirror.read_text()).get('sha256') != committed['sha256']:
                raise ValueError('Metadata mirror differs from its committed checksum')
            _atomic_json(mirror, {'sha256': committed['sha256']})
            return metadata_path
        # Compatibility with previously committed outputs from the original
        # journal implementation, which used only the external checksum file.
        if mirror.exists():
            if json.loads(mirror.read_text()).get('sha256') != sha256(metadata_path):
                raise ValueError('Existing metadata differs from its committed checksum')
            return metadata_path
        if _load_progress(output / 'render-progress.jsonl') or (output / 'manifest.jsonl').exists():
            raise ValueError('Metadata lacks a completion record after rendering; it will not be regenerated or trusted')
        quarantine(log_dir)
    elif mirror.exists() or _load_progress(output / 'render-progress.jsonl'):
        raise ValueError('Committed generation metadata is missing; cannot resume this output')
    elif log_dir.exists():
        quarantine(log_dir)

    # A crash after the completion marker but before directory promotion reuses
    # the already completed draw, without invoking the generator again.
    if staging.exists():
        committed = completion(staging / 'log')
        if committed is not None:
            (staging / 'log').replace(log_dir)
            _atomic_json(mirror, {'sha256': committed['sha256']})
            return metadata_path
        quarantine(staging)
    staging.mkdir()
    (staging / 'log').mkdir()
    generation_config = {**plan['config'], 'log_dir': str(staging / 'log')}
    config_path = staging / 'simulation.yaml'
    config_path.write_text(yaml.safe_dump(generation_config, sort_keys=False))
    subprocess.run([sys.executable, str(source / 'simulation/generate_data_param.py'),
                    '--config', str(config_path)], cwd=output, check=True)
    completed_path = staging / 'log/meta.tsv'
    with completed_path.open('rb') as handle:
        os.fsync(handle.fileno())
    record = {'sha256': sha256(completed_path), 'plan_sha256': identity}
    _atomic_json(staging / 'log/generation-complete.json', record)
    (staging / 'log').replace(log_dir)
    _atomic_json(mirror, {'sha256': record['sha256']})
    return metadata_path



def _validate_loader(manifest: Path, *, workers: int = 1) -> dict[str, Any]:
    """Validate through the inference loader, retaining no decoded audio results.

    With no crops, augmentation, or teacher, __getitem__ only reads immutable
    records and independently opens each audio file. Threads overlap filesystem
    reads without transferring tensors or changing training-loader behavior.
    """
    workers = _worker_count(workers)
    started = time.monotonic()
    dataset = PairedAudioDataset(manifest, RATE_SET, training=False, load_teacher=False)
    construction_seconds = time.monotonic() - started
    decode_started = time.monotonic()
    max_in_flight = 0
    def validate(index):
        dataset[index]  # Tensors are discarded inside the reading thread.
    if workers == 1:
        for index in range(len(dataset)):
            validate(index)
        max_in_flight = min(len(dataset), 1)
    else:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='pair-validation')
        pending = deque()
        indices = iter(range(len(dataset)))
        try:
            def submit_one():
                index = next(indices, None)
                if index is None:
                    return False
                pending.append(pool.submit(validate, index))
                return True
            for _ in range(min(len(dataset), 2 * workers)):
                submit_one()
            max_in_flight = len(pending)
            while pending:
                pending.popleft().result()
                submit_one()
        finally:
            # On a bad pair, drain active reads before propagating the failure;
            # the caller must never publish the pending manifest on an error.
            pool.shutdown(wait=True, cancel_futures=True)
    return {'workers': workers, 'executor': 'threads' if workers > 1 else 'in_process',
            'examples': len(dataset), 'max_in_flight': max_in_flight,
            'construction_seconds': construction_seconds,
            'decode_seconds': time.monotonic() - decode_started,
            'total_seconds': time.monotonic() - started}


def _execute_locked(plan: dict[str, Any], module, *, resume: bool, workers: int = 1) -> Path:
    output = Path(plan['output_dir'])
    plan_path = output / 'plan.json'
    config_path = output / 'simulation.yaml'
    if plan_path.exists():
        if not resume or _base_plan(json.loads(plan_path.read_text())) != _base_plan(plan):
            raise ValueError('Resume requires the same saved preparation plan; use a new output for changed inputs/settings')
    else:
        if resume and any(path.name != '.preparation.lock' for path in output.iterdir()):
            raise ValueError('Cannot resume an output without its saved plan.json')
        config_path.write_text(yaml.safe_dump(plan['config'], sort_keys=False))
        _atomic_json(plan_path, _base_plan(plan))
    if yaml.safe_load(config_path.read_text()) != plan['config']:
        raise ValueError('Saved simulation.yaml differs from the preparation plan')
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'librosa', 'torch', 'torchaudio', 'espnet', 'resampy')}
    runtime_path = output / 'render-runtime.json'
    if runtime_path.exists():
        if json.loads(runtime_path.read_text()) != versions:
            raise ValueError('Preparation runtime versions changed; restore the saved environment or use a new output')
    else:
        _atomic_json(runtime_path, versions)
    metadata_path = _ensure_metadata(plan)
    # A readiness marker must not survive while a resumed dataset is being
    # revalidated/repaired. Keep the previous manifest for inspection.
    final_manifest = output / 'manifest.jsonl'
    if final_manifest.exists():
        final_manifest.replace(output / 'manifest.previous.jsonl')
    # Rebuild accumulated evidence from verified ledger rows/new render calls.
    plan.pop('rir_alignment_evidence', None)
    previous_directory = Path.cwd()
    try:
        os.chdir(output)  # upstream wind temporary WAVs stay inside this run
        render_started = time.monotonic()
        execution = {}
        rows = render_metadata(plan, module, progress_path=output / 'render-progress.jsonl', workers=workers, execution=execution)
        render_seconds = time.monotonic() - render_started
    finally:
        os.chdir(previous_directory)
    if not rows:
        raise ValueError('Official generator produced no examples')
    manifest = output / 'manifest.pending.jsonl'
    with manifest.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    loader_validation = _validate_loader(manifest, workers=workers)
    coverage = {name: sum(name in row['augmentation'] for row in rows)
                for name in ('clipping', 'bandwidth_limitation', 'codec', 'packet_loss', 'wind_noise')}
    coverage.update(reverberation=sum(row['rir_uid'] != 'none' for row in rows),
                    additive_noise=sum(not row['noise_uid'].startswith('wind_noise') for row in rows))
    _atomic_json(output / 'provenance.json', {**plan, 'runtime_versions': versions,
        'metadata_sha256': sha256(metadata_path), 'progress_sha256': sha256(output / 'render-progress.jsonl'),
        'manifest_sha256': sha256(manifest), 'realized_degradation_counts': coverage, 'examples': len(rows),
        'loader_validation': loader_validation,
        'render_execution': {**execution, 'start_method': 'spawn' if workers > 1 else 'in_process',
                             'numeric_threads_per_worker': 1 if workers > 1 else None,
                             'render_and_resume_validation_seconds': render_seconds}})
    return manifest.replace(final_manifest)


def execute_preparation(plan: dict[str, Any], *, resume: bool = False, workers: int = 1) -> Path:
    """Prepare only; explicit resume reuses verified rows under the identical plan."""
    workers = _worker_count(workers)
    source, output = Path(plan['source']), Path(plan['output_dir'])
    verify_source(source)
    if output.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite preparation directory: {output}')
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError('Preparation output must be a real directory')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('Official wind/codec preparation requires FFmpeg on PATH')
    try:
        if importlib.metadata.version('espnet') != '202509':
            raise RuntimeError('Use espnet==202509 in the preparation environment')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('Install the optional preparation dependencies in a separate environment') from exc
    try:
        module = load_simulator(source)
    except (ImportError, OSError) as exc:
        raise RuntimeError('Install the pinned URGENT simulator dependencies in a separate preparation environment') from exc
    output.mkdir(parents=True, exist_ok=resume)
    # Preparation hosts are Linux/macOS. An advisory lock is released by the OS
    # even after abrupt process termination; no stale PID files need bypassing.
    import fcntl
    signal_guard = _graceful_sigterm() if workers > 1 else nullcontext()
    with signal_guard, (output / '.preparation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is preparing this output directory') from exc
        try:
            return _execute_locked(plan, module, resume=resume, workers=workers)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def audit_splits(manifests: list[Path]) -> dict[str, Any]:
    """Reject recording/speaker leakage using explicit provenance identifiers."""
    seen = {key: {} for key in ('speech_uid', 'speaker_id', 'source_speech_sha256', 'noise_uid', 'source_noise_sha256')}
    splits, count = set(), 0
    for manifest in manifests:
        local_split = set()
        with manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                split = row.get('split')
                if split not in {'train', 'validation', 'test'}:
                    raise ValueError(f'{manifest}: missing/invalid split')
                local_split.add(split)
                for key, identities in seen.items():
                    identity = row.get(key)
                    if not identity:
                        raise ValueError(f'{manifest}: missing {key} provenance')
                    if identity in identities and identities[identity] != split:
                        raise ValueError(f'Split leakage: {key}={identity!r} occurs in {identities[identity]} and {split}')
                    identities[identity] = split
                count += 1
        if len(local_split) != 1:
            raise ValueError(f'{manifest}: expected exactly one nonempty split')
        splits.update(local_split)
    return {'manifests': len(manifests), 'splits': sorted(splits), 'examples': count,
            'checks': list(seen), 'note': 'Namespaced IDs and source-byte hashes do not detect all transcoded duplicates; audit source split design too.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('urgent')
    for flag in ('source', 'config', 'provenance', 'output-dir'):
        prepare.add_argument(f'--{flag}', type=Path, required=True)
    prepare.add_argument('--execute', action='store_true', help='Render local inputs; default prints plan only')
    prepare.add_argument('--resume', action='store_true', help='With --execute, reuse only verified rows under the identical saved plan')
    prepare.add_argument('--workers', type=int, default=1, help='CPU rendering processes; may change on resume without changing the recipe')
    prepare.add_argument('--usage', choices=('commercial', 'research'), default='commercial',
                         help='Research preserves pending rights review; it does not establish permission')
    prepare.add_argument('--rir-alignment', choices=('audited', 'upstream-estimated-onset'), default='audited',
                         help='Explicitly opt into the official onset heuristic when physical delay audits are unavailable')
    audit = commands.add_parser('audit-splits')
    audit.add_argument('manifests', nargs='+', type=Path)
    args = parser.parse_args()
    if args.command == 'audit-splits':
        print(json.dumps(audit_splits(args.manifests), indent=2))
    else:
        plan = preparation_plan(args.source, args.config, args.provenance, args.output_dir,
                                usage=args.usage, rir_alignment=args.rir_alignment)
        if args.resume and not args.execute:
            parser.error('--resume requires --execute')
        if args.workers < 1:
            parser.error('--workers must be a positive integer')
        print(execute_preparation(plan, resume=args.resume, workers=args.workers) if args.execute else json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
