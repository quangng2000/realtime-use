"""Verify and materialize one downloaded source pool for the URGENT adapter.

This module only handles existing local data. It never downloads, trains, or
asserts dataset permissions. Extraction uses standard-library streaming tar I/O.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Any

import numpy as np
import soundfile as sf


ROLES = ('speech', 'noise', 'rir', 'wind')
SPLITS = ('train', 'validation', 'test')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Archive paths must be nonempty relative POSIX paths')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path == PurePosixPath('.'):
        raise ValueError(f'Unsafe archive path: {value!r}')
    return path


def _token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError(f'{field} must be a nonempty token without whitespace/control characters')
    return value


def _atomic_text(path: Path, value: str | Iterable[str]) -> None:
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as handle:
        temporary = Path(handle.name)
        try:
            if isinstance(value, str):
                handle.write(value)
            else:
                handle.writelines(value)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _verify_audio(path: Path, row: dict[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular audio file: {path}')
    if file_sha256(path) != row['sha256']:
        raise ValueError(f'Audio checksum mismatch: {row["id"]}')
    with sf.SoundFile(path) as audio:
        if (audio.samplerate, audio.channels, len(audio)) != (row['sample_rate'], row['channels'], row['frames']):
            raise ValueError(f'Audio rate/channel/frame metadata mismatch: {row["id"]}')
        for block in audio.blocks(blocksize=65536, dtype='float32'):
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite audio: {row["id"]}')


def _verification(row: dict[str, Any], output_sha256: str) -> dict[str, Any]:
    """Keep original source identity separate from the normalized file identity."""
    return {'uid': row['uid'], 'source_sha256': row['sha256'],
            'source_channels': row['channels'], 'materialized_sha256': output_sha256,
            'materialized_channels': 1, 'materialized_sample_rate': row['sample_rate'],
            'materialized_frames': row['frames'],
            'channel_transform': ('identity' if row['channels'] == 1 else 'first_channel_float32_wav')}


def _reuse_audio(target: Path, row: dict[str, Any], verification_dir: Path) -> None:
    if row['channels'] == 1:
        _verify_audio(target, row)
        row.update(_verification(row, row['sha256']))
        return
    metadata_path = verification_dir / (target.name + '.json')
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f'Missing regular first-channel verification record: {row["id"]}')
    metadata = json.loads(metadata_path.read_text())
    output_hash = metadata.get('materialized_sha256', '')
    if not re.fullmatch(r'[0-9a-f]{64}', output_hash) or metadata != _verification(row, output_hash):
        raise ValueError(f'First-channel verification identity mismatch: {row["id"]}')
    _verify_audio(target, {**row, 'channels': 1, 'sha256': output_hash})
    row.update(metadata)


def _publish_audio(original: Path, target: Path, row: dict[str, Any], verification_dir: Path) -> None:
    _verify_audio(original, row)
    if row['channels'] == 1:
        os.replace(original, target)
        row.update(_verification(row, row['sha256']))
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.wav', delete=False) as handle:
        normalized = Path(handle.name)
    try:
        # This follows URGENT's force_1ch selection, never a channel average.
        with sf.SoundFile(original) as source, sf.SoundFile(
            normalized, mode='w', samplerate=row['sample_rate'], channels=1,
            format='WAV', subtype='FLOAT'
        ) as destination:
            for block in source.blocks(blocksize=65536, dtype='float32', always_2d=True):
                destination.write(block[:, :1])
        output_hash = file_sha256(normalized)
        _verify_audio(normalized, {**row, 'channels': 1, 'sha256': output_hash})
        metadata = _verification(row, output_hash)
        # Publish verification first: a crash before the audio rename is safely
        # recoverable by re-extracting the still-missing output on the next run.
        _atomic_text(verification_dir / (target.name + '.json'), json.dumps(metadata, indent=2) + '\n')
        os.replace(normalized, target)
        row.update(metadata)
    finally:
        normalized.unlink(missing_ok=True)


def make_plan(root: Path, manifest: Path, output: Path, role: str,
              max_items: int | None = None, channel_policy: str = 'reject',
              wind_id_scope: str = 'global') -> dict[str, Any]:
    """Validate selected source records without creating or extracting anything."""
    if channel_policy not in ('reject', 'first'):
        raise ValueError('channel_policy must be reject or first')
    if role not in ROLES:
        raise ValueError(f'Unknown source role: {role}')
    if wind_id_scope not in ('global', 'split') or (wind_id_scope == 'split' and role != 'wind'):
        raise ValueError('wind_id_scope=split is only valid for independently generated wind pools')
    if max_items is not None and max_items < 1:
        raise ValueError('max_items must be positive')
    root, manifest, output = (p.expanduser().resolve() for p in (root, manifest, output))
    if any(c.isspace() for c in str(output)):
        raise ValueError('The official SCP format cannot represent output paths containing whitespace')
    rows: list[dict[str, Any]] = []
    seen_ids, seen_members = set(), set()
    source_split = None
    with manifest.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ('id', 'source_id', 'source', 'split', 'kind', 'sample_rate', 'frames',
                        'channels', 'sha256', 'shard', 'path')
            if not isinstance(row, dict) or any(key not in row for key in required):
                raise ValueError(f'{manifest}:{line_number}: missing source-record fields')
            source = _token(row['source'], 'source')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', source):
                raise ValueError('source must be a simple corpus identifier')
            source_id = _token(row['source_id'], 'source_id')
            _token(row['id'], 'id')
            if row['split'] not in SPLITS and not (role == 'rir' and row['split'] == 'pool'):
                raise ValueError('split must be train, validation, or test; only RIRs may use pool')
            current = (source, row['split'])
            if source_split is not None and current != source_split:
                raise ValueError('Each invocation must contain exactly one source and split')
            source_split = current
            if role == 'speech' and row['kind'] != 'speech':
                raise ValueError('Speech inventories require speech source records')
            if role == 'wind' and not source_id.startswith('wind_noise'):
                raise ValueError('Wind source IDs must begin with wind_noise')
            if role == 'rir' and row['kind'] != 'rir':
                raise ValueError('RIR inventories require rir source records')
            if role in ('noise', 'wind') and row['kind'] != 'noise':
                raise ValueError('Noise/wind inventories require noise source records')
            for key in ('sample_rate', 'frames', 'channels'):
                if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 1:
                    raise ValueError(f'{key} must be a positive integer')
            if row['channels'] != 1 and channel_policy == 'reject':
                raise ValueError('Multichannel sources require explicit channel_policy=first')
            if not isinstance(row['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
                raise ValueError('Each record requires its original lowercase SHA-256')
            member = str(_relative(row['path']))
            shard_relative = _relative(row['shard'])
            shard = (root / shard_relative).resolve(strict=True)
            if not shard.is_relative_to(root) or not shard.is_file():
                raise ValueError('Source shard escapes the dataset root or is not a file')
            # Recorded-source identities exclude split to expose leakage. Some
            # synthetic wind runs restart their counters independently per split.
            # Opting into split-local IDs preserves raw IDs and waveform hashes.
            uid = f'{source}:{source_id}'
            if role == 'wind':
                if wind_id_scope == 'split':
                    uid = f'{source}:{row["split"]}:{source_id}'
                uid = f'wind_noise:{uid}'
            if uid in seen_ids or (str(shard), member) in seen_members:
                raise ValueError('Duplicate source identity or archive member in selected records')
            seen_ids.add(uid)
            seen_members.add((str(shard), member))
            materialized_speaker_id = None
            if role == 'speech':
                materialized_speaker_id = _token(row.get('speaker_id'), 'speaker_id')
                if source == 'mls':
                    parts = source_id.split('_', 2)
                    if len(parts) < 3 or parts[0] != 'mls' or not parts[1]:
                        raise ValueError('MLS source IDs must include mls_<language>_<utterance>')
                    materialized_speaker_id = f'mls:{parts[1]}:{materialized_speaker_id}'
                text = row.get('text')
                if text is not None and (not isinstance(text, str) or '\n' in text or '\r' in text):
                    raise ValueError('Transcripts must be single-line text')
            suffix = PurePosixPath(member).suffix.lower()
            if suffix not in {'.wav', '.flac', '.ogg', '.opus', '.aiff', '.aif'}:
                raise ValueError(f'Unsupported source audio suffix: {suffix}')
            if row['channels'] > 1:
                suffix = '.wav'
            filename = hashlib.sha256(uid.encode()).hexdigest() + suffix
            rows.append({**row, 'uid': uid, 'resolved_shard': str(shard), 'member': member,
                         'materialized_path': str(output / 'audio' / filename),
                         'materialized_speaker_id': materialized_speaker_id})
            if max_items is not None and len(rows) >= max_items:
                break
    if not rows:
        raise ValueError('No source records selected')
    return {'version': 3 if wind_id_scope == 'split' else 2,
            **({'wind_id_scope': 'split'} if wind_id_scope == 'split' else {}),
            'root': str(root), 'source_manifest': str(manifest),
            'source_manifest_sha256': file_sha256(manifest), 'output_dir': str(output),
            'role': role, 'source': source_split[0], 'split': source_split[1],
            'max_items': max_items, 'channel_policy': channel_policy, 'selected_count': len(rows),
            'sample_rates': sorted({row['sample_rate'] for row in rows}),
            'records': rows}


def materialize(plan: dict[str, Any]) -> dict[str, Any]:
    """Extract verified audio and atomically publish SCPs after every item passes."""
    output = Path(plan['output_dir'])
    request = {key: value for key, value in plan.items() if key != 'records'}
    if file_sha256(Path(plan['source_manifest'])) != plan['source_manifest_sha256']:
        raise ValueError('Source manifest changed after planning')
    if output.exists():
        request_path = output / 'request.json'
        if not request_path.is_file() or json.loads(request_path.read_text()) != request:
            raise ValueError('Output exists with a different or missing materialization request')
    else:
        output.mkdir(parents=True)
        _atomic_text(output / 'request.json', json.dumps(request, indent=2) + '\n')
    audio_dir = output / 'audio'
    if audio_dir.is_symlink():
        raise ValueError('Audio output directory cannot be a symlink')
    audio_dir.mkdir(exist_ok=True)
    verification_dir = output / 'verification'
    if verification_dir.is_symlink():
        raise ValueError('Verification output directory cannot be a symlink')
    verification_dir.mkdir(exist_ok=True)
    pending = defaultdict(dict)
    reused = 0
    for row in plan['records']:
        target = Path(row['materialized_path'])
        if target.exists() or target.is_symlink():
            _reuse_audio(target, row, verification_dir)
            reused += 1
        else:
            pending[row['resolved_shard']][row['member']] = row
    extracted = 0
    for shard, wanted in pending.items():
        found = set()
        # Never call extract/extractall. Tar headers never control an output path.
        with tarfile.open(shard, mode='r|*') as archive:
            for member in archive:
                name = str(_relative(member.name))
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f'Archive links/special entries are not supported: {name}')
                if name in wanted:
                    if name in found or not member.isfile():
                        raise ValueError(f'Duplicate or nonregular selected archive member: {name}')
                    found.add(name)
                    row = wanted[name]
                    target = Path(row['materialized_path'])
                    with tempfile.NamedTemporaryFile(dir=audio_dir, suffix=target.suffix, delete=False) as handle:
                        temporary = Path(handle.name)
                        try:
                            with archive.extractfile(member) as source:
                                while block := source.read(1024 * 1024):
                                    handle.write(block)
                            handle.close()
                            _publish_audio(temporary, target, row, verification_dir)
                        finally:
                            temporary.unlink(missing_ok=True)
                    extracted += 1
                # Python <=3.12 retains streamed TarInfo objects unless cleared.
                archive.members.clear()
        missing = set(wanted) - found
        if missing:
            raise ValueError(f'Missing {len(missing)} selected archive members in {shard}: {sorted(missing)[:3]}')
    rows = plan['records']
    _atomic_text(output / 'source.scp', (f'{r["uid"]} {r["sample_rate"]} {r["materialized_path"]}\n' for r in rows))
    if plan['role'] == 'speech':
        _atomic_text(output / 'utt2spk', (f'{r["uid"]} {r["materialized_speaker_id"]}\n' for r in rows))
        # Missing transcripts stay absent, rather than becoming invented reference text.
        _atomic_text(output / 'text', (f'{r["uid"]} {r["text"]}\n' for r in rows if r.get('text')))
    _atomic_text(output / 'records.jsonl', (json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    result = {**request, 'status': 'verified', 'extracted': extracted, 'reused': reused,
              'scp': str(output / 'source.scp'), 'licenses': sorted({str(r.get('license', 'unknown')) for r in rows}),
              'commercial_clearance': 'not_attested', 'training_pairs_ready': False,
              'note': 'Source inventory only. Retains input metadata; does not grant permission, '
                      'audit RIR direct-path timing, render degradations, or start training.'}
    _atomic_text(output / 'materialization.json', json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Dataset root containing source shards')
    parser.add_argument('--manifest', type=Path, required=True, help='One source/split JSONL inventory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--role', choices=ROLES, required=True)
    parser.add_argument('--max-items', type=int, help='Deterministic first-N subset for readiness checks')
    parser.add_argument('--channel-policy', choices=('reject', 'first'), default='reject',
                        help='Reject multichannel audio, or retain its first channel as mono float WAV')
    parser.add_argument('--wind-id-scope', choices=('global', 'split'), default='global',
                        help='Use split only for independently generated wind pools with split-local counters; waveform leakage checks still apply')
    parser.add_argument('--execute', action='store_true', help='Materialize existing source audio; never train')
    args = parser.parse_args()
    plan = make_plan(args.root, args.manifest, args.output_dir, args.role, args.max_items,
                     args.channel_policy, args.wind_id_scope)
    result = materialize(plan) if args.execute else {k: v for k, v in plan.items() if k != 'records'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
