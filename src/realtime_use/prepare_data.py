"""Pinned URGENT simulator integration; never downloads corpora or starts training.

The upstream generator chooses seven degradations. Rendering delegates to its
implementation, with explicit anechoic-target and upstream wind-bug adapters.
Use a separate preparation environment with the upstream dependencies installed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
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


def preparation_plan(source: Path, config_path: Path, provenance_path: Path, output: Path) -> dict[str, Any]:
    """Validate local input metadata and produce a reproducible, nonexecuting plan."""
    source, config_path, provenance_path, output = (path.expanduser().resolve() for path in
                                                    (source, config_path, provenance_path, output))
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
        if entry['commercial_use'] is not True or not entry['license'] or not entry['license_evidence']:
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
            if entry['sample_rate'] >= rate:
                delay = delays.get(uid, {}).get(str(rate))
                if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
                    raise ValueError(f'Provide audited direct-path delay for RIR {uid} at rendering rate {rate}')
    config.update(output_dir=str(output), log_dir=str(output / 'log'), out_format='wav', store_noise=False)
    return {'format_version': 1, 'source': str(source), 'source_url': URGENT_URL,
            'source_commit': URGENT_COMMIT, 'config': config, 'inventories': inventories,
            'provenance': {**provenance, 'sources': list(by_scp.values())},
            'input_config_sha256': sha256(config_path), 'output_dir': str(output),
            'target': 'anechoic_unit_gain_direct_path_delayed_same_joint_normalization',
            'upstream_probability_note': f'Pinned generator omits RIR when U <= prob_reverberation; effective reverb probability is {1-probability:g}. No silent rewrite.',
            'degradation_coverage': 'Seven families enabled; actual sampled coverage is reported after rendering, not guaranteed by a finite random draw.'}


def direct_path_impulse(rir: np.ndarray, delay: int) -> np.ndarray:
    if rir.ndim != 2 or isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay < rir.shape[-1]:
        raise ValueError('Audited direct-path delay must fit the rendered RIR')
    impulse = np.zeros_like(rir)
    impulse[:, delay] = 1
    return impulse


@contextmanager
def rendering_adapters(module, *, delay: int | None, wind_clipping: bool):
    """Small scoped changes to upstream target semantics and two wind issues.

    No degradation formula is copied. Upstream bool('False') is corrected using
    the metadata value, and its wind routine receives a copy to protect targets.
    Float WAV output avoids introducing PCM16 quantization into clean targets.
    """
    previous = {name: getattr(module, name) for name in ('estimate_early_rir', 'wind_noise', 'save_audio')}
    def target_rir(rir, **kwargs):
        if delay is None:
            raise ValueError('Missing audited direct-path delay')
        return direct_path_impulse(rir, delay)
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
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def render_metadata(plan: dict[str, Any], module) -> list[dict[str, Any]]:
    """Render official metadata using upstream code; useful as a testable adapter."""
    output = Path(plan['output_dir'])
    inventory = plan['inventories']
    mappings = {role: {uid: entry['path'] for uid, entry in entries.items()} for role, entries in inventory.items()}
    mappings['noise'].update(mappings['wind'])
    rows = []
    hashes = {}
    def source_hash(filename):
        if filename not in hashes:
            hashes[filename] = sha256(Path(filename))
        return hashes[filename]
    with (output / 'log/meta.tsv').open() as handle:
        metadata = list(csv.DictReader(handle, delimiter='\t'))
    for index, original in enumerate(metadata):
        info = dict(original)
        rate = int(info['fs'])
        # Do not trust output paths from TSV; confine writes to this new run.
        for key, directory in (('noisy_path', 'noisy'), ('clean_path', 'clean')):
            destination = output / directory / f'{index:09d}.wav'
            destination.parent.mkdir(parents=True, exist_ok=True)
            info[key] = str(destination)
        rir_id = info['rir_uid']
        delay = None if rir_id == 'none' else plan['provenance']['rir_direct_path_samples'][rir_id][str(rate)]
        wind_match = re.search(r'clipping=(True|False)(?:,|\))', info['augmentation'])
        with rendering_adapters(module, delay=delay, wind_clipping=bool(wind_match and wind_match[1] == 'True')):
            module.process_one_sample(info, force_1ch=True, store_noise=False,
                                      speech_dic=mappings['speech'], noise_dic=mappings['noise'], rir_dic=mappings['rir'])
        row = {'id': f"{plan['provenance']['split']}:{info['id']}",
               'noisy': info['noisy_path'], 'clean': info['clean_path'], 'sample_rate': rate,
               'split': plan['provenance']['split'], 'speech_uid': info['speech_uid'],
               'speaker_id': info['speech_sid'],
               'noise_uid': info['noise_uid'], 'rir_uid': rir_id, 'direct_path_delay_samples': delay or 0,
               'augmentation': info['augmentation'], 'target': plan['target'],
               **({'transcript': info['text']} if info.get('text') not in (None, '<not-available>') else {}),
               'upstream_commit': URGENT_COMMIT,
               'source_speech_sha256': source_hash(mappings['speech'][info['speech_uid']]),
               'source_noise_sha256': source_hash(mappings['noise'][info['noise_uid']]),
               'provenance': str(output / 'provenance.json')}
        rows.append(row)
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
        load('rir_utils', 'rir_utils.py')
        name = '_realtime_use_urgent_renderer'
        previous_renderer = sys.modules.get(name)
        try:
            return load(name, 'simulate_data_from_param.py')
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


def execute_preparation(plan: dict[str, Any]) -> Path:
    """Explicit preparation only: no dataset downloads or model/training imports."""
    source, output = Path(plan['source']), Path(plan['output_dir'])
    verify_source(source)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite preparation directory: {output}')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('Official wind/codec preparation requires FFmpeg on PATH')
    try:
        if importlib.metadata.version('espnet') != '202509':
            raise RuntimeError('Use espnet==202509 in the preparation environment')
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('Install the optional preparation dependencies in a separate environment') from exc
    # Heavy upstream imports happen only on explicit execution, never plan/audit.
    try:
        module = load_simulator(source)
    except (ImportError, OSError) as exc:
        raise RuntimeError('Install the pinned URGENT simulator dependencies in a separate preparation environment') from exc
    output.mkdir(parents=True)
    (output / 'log').mkdir()
    config_path = output / 'simulation.yaml'
    config_path.write_text(yaml.safe_dump(plan['config'], sort_keys=False))
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    subprocess.run([sys.executable, str(source / 'simulation/generate_data_param.py'),
                    '--config', str(config_path)], cwd=output, check=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(output)  # upstream wind temporary WAVs stay inside this run
        rows = render_metadata(plan, module)
    finally:
        os.chdir(previous_directory)
    if not rows:
        raise ValueError('Official generator produced no examples')
    manifest = output / 'manifest.pending.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    dataset = PairedAudioDataset(manifest, RATE_SET, training=False, load_teacher=False)
    for index in range(len(dataset)):
        dataset[index]  # decode finite/length validation; metadata is not enough
    manifest = manifest.rename(output / 'manifest.jsonl')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'librosa', 'torch', 'torchaudio', 'espnet', 'resampy')}
    coverage = {name: sum(name in row['augmentation'] for row in rows)
                for name in ('clipping', 'bandwidth_limitation', 'codec', 'packet_loss', 'wind_noise')}
    coverage.update(reverberation=sum(row['rir_uid'] != 'none' for row in rows),
                    additive_noise=sum(not row['noise_uid'].startswith('wind_noise') for row in rows))
    (output / 'provenance.json').write_text(json.dumps({**plan, 'runtime_versions': versions,
        'manifest_sha256': sha256(manifest), 'realized_degradation_counts': coverage,
        'examples': len(rows)}, indent=2) + '\n')
    return manifest


def audit_splits(manifests: list[Path]) -> dict[str, Any]:
    """Reject recording/speaker leakage using explicit provenance identifiers."""
    seen = {key: {} for key in ('speech_uid', 'speaker_id', 'source_speech_sha256', 'noise_uid', 'source_noise_sha256')}
    splits, count = set(), 0
    for manifest in manifests:
        local_split = set()
        for line in manifest.read_text().splitlines():
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
    prepare.add_argument('--execute', action='store_true', help='Render permitted local inputs; default prints plan only')
    audit = commands.add_parser('audit-splits')
    audit.add_argument('manifests', nargs='+', type=Path)
    args = parser.parse_args()
    if args.command == 'audit-splits':
        print(json.dumps(audit_splits(args.manifests), indent=2))
    else:
        plan = preparation_plan(args.source, args.config, args.provenance, args.output_dir)
        print(execute_preparation(plan) if args.execute else json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
