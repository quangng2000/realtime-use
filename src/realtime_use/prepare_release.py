"""Prepare a verified downloaded source release for paired training; never train.

The default command only inspects the release. Explicit execution materializes
audio, runs the pinned URGENT adapter in resumable chunks, and audits the result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable

import yaml

from . import materialize_sources as bridge
from . import prepare_data as prep


INDEX_ROLES = {
    'speech_train': ('speech', 'train'), 'speech_validation': ('speech', 'validation'),
    'noise_train': ('noise', 'train'), 'noise_validation': ('noise', 'validation'),
    'wind_train': ('wind', 'train'), 'wind_validation': ('wind', 'validation'),
    'rir_pool': ('rir', 'pool'),
}


def _rows(path: Path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _confined(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a confined relative artifact path: {relative}')
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f'Artifact escapes release root or is not a file: {relative}')
    return result


def _verified_artifact(root: Path, entry: dict[str, Any], *, checksum: bool = True) -> Path:
    path = _confined(root, entry['path'])
    if path.stat().st_size != entry['bytes'] or (checksum and prep.sha256(path) != entry['sha256']):
        raise ValueError(f'Artifact integrity mismatch: {entry["path"]}')
    return path


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2) + '\n'


def _write(path: Path, lines: str | Iterable[str], *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Refusing symlink output: {path}')
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(lines) if isinstance(lines, str) else handle.writelines(lines)
            handle.close()
            if immutable and path.exists():
                if prep.sha256(path) != prep.sha256(temporary):
                    raise ValueError(f'Refusing changed existing output: {path}')
                return
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _template(source: Path, split: str) -> tuple[dict[str, Any], str]:
    relative = f'conf/simulation_{split}.yaml'
    path = source / relative
    expected = subprocess.check_output(['git', '-C', str(source), 'show', f'{prep.URGENT_COMMIT}:{relative}'])
    if path.read_bytes() != expected:
        raise ValueError(f'Pinned official simulation config differs: {relative}')
    return yaml.safe_load(expected), hashlib.sha256(expected).hexdigest()


def release_plan(root: Path, source: Path, output: Path, *, chunk_size: int = 1024,
                 max_speech_per_split: int | None = None,
                 validation_reuse_rir: bool = False) -> dict[str, Any]:
    """Read-only release inspection. Full audio verification occurs on extraction."""
    root, source, output = (p.expanduser().resolve() for p in (root, source, output))
    if chunk_size < 1 or (max_speech_per_split is not None and max_speech_per_split < 1):
        raise ValueError('Chunk size and any explicit speech limit must be positive')
    if any(character.isspace() for character in str(output)):
        raise ValueError('Official SCP paths cannot contain whitespace')
    if output == root or root.is_relative_to(output):
        raise ValueError('Output must not be the source root or an ancestor of it')
    prep.verify_source(source)
    templates = {split: _template(source, split)[1] for split in ('train', 'validation')}
    release_path = root / 'release/RELEASE.json'
    release = json.loads(release_path.read_text())
    if release.get('status') not in {'source_pools_ready', 'partial_source_pools'}:
        raise ValueError('A finalized source release is required')
    if release.get('upstream_commit') != prep.URGENT_COMMIT:
        raise ValueError('Source release uses a different upstream revision')
    artifacts = {item['path']: item for item in release['artifacts']}
    groups, indices, selected_counts, markers = {}, {}, {}, {}
    source_artifacts = {}
    pair_bytes = {'train': 0, 'validation': 0}
    for name, (role, split) in INDEX_ROLES.items():
        relative = f'release/{name}.jsonl'
        if relative not in artifacts:
            raise ValueError(f'Release is missing required {role}/{split} inventory')
        path = _verified_artifact(root, artifacts[relative])
        indices[name] = artifacts[relative]
        count = 0
        for row in _rows(path):
            if role == 'speech' and max_speech_per_split is not None and count >= max_speech_per_split:
                break
            corpus = row.get('source', '')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', corpus) or row.get('split') != split:
                raise ValueError(f'Invalid corpus/split in {relative}')
            expected_kind = 'noise' if role == 'wind' else role
            if row.get('kind') != expected_kind:
                raise ValueError(f'Unexpected source role in {relative}')
            if corpus not in markers:
                marker_path = _confined(root, f'sources/{corpus}/SOURCE_READY.json')
                marker = json.loads(marker_path.read_text())
                if marker.get('status') != 'verified':
                    raise ValueError(f'Source completion marker is not verified: {corpus}')
                # Hash manifests; archive members are hashed and decoded during
                # materialization, avoiding an extra full-corpus archive pass.
                for artifact in marker['artifacts']:
                    _verified_artifact(root, artifact, checksum=artifact['path'].endswith('.jsonl'))
                source_artifacts[corpus] = {artifact['path'] for artifact in marker['artifacts']}
                markers[corpus] = {'path': str(marker_path.relative_to(root)),
                                   'sha256': prep.sha256(marker_path)}
            if row.get('shard') not in source_artifacts[corpus]:
                raise ValueError(f'Selected archive is absent from its verified source marker: {corpus}')
            key = f'{role}__{split}__{corpus}'
            group = groups.setdefault(key, {'key': key, 'role': role, 'split': split, 'source': corpus,
                                            'index': name, 'count': 0, 'sample_rates': [], 'licenses': []})
            group['count'] += 1
            if role == 'speech':
                frames = row['frames']
                if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
                    raise ValueError('Speech frame counts must be positive integers')
                pair_bytes[split] += 8 * frames  # two mono float32 waveforms
            if row['sample_rate'] not in group['sample_rates']:
                group['sample_rates'].append(row['sample_rate'])
            license_name = str(row.get('license', 'unknown'))
            if license_name not in group['licenses']:
                group['licenses'].append(license_name)
            count += 1
        if not count:
            raise ValueError(f'Empty required inventory: {name}')
        selected_counts[name] = count
    for group in groups.values():
        group['sample_rates'].sort()
        group['licenses'].sort()
        # The official synthetic generator restarts its index separately for
        # train/validation. Scope only those documented generated wind IDs;
        # physical recordings retain global identity and all hashes are audited.
        if group['role'] == 'wind' and group['licenses'] == ['Generated using MIT-licensed SC-Wind-Noise-Generator']:
            group['wind_id_scope'] = 'split'
        # Absence means the historical global identity policy. Do not add a
        # default key that would invalidate otherwise identical old requests.
    return {'format_version': 1, 'root': str(root), 'source': str(source), 'output_dir': str(output),
            'source_commit': prep.URGENT_COMMIT, 'release_sha256': prep.sha256(release_path),
            'source_release_status': release['status'], 'incomplete_sources': release.get('failed_or_incomplete_sources', {}),
            'omitted_sources': release.get('omitted', {}), 'indices': indices, 'markers': markers,
            'template_sha256': templates, 'groups': groups, 'selected_counts': selected_counts,
            'estimated_pair_audio_payload_bytes': pair_bytes,
            'storage_estimate_note': 'Two float32 mono waveforms per speech item; excludes WAV headers, source materialization, metadata and filesystem overhead.',
            'chunk_size': chunk_size, 'max_speech_per_split': max_speech_per_split,
            'selection_scope': 'all_released_speech' if max_speech_per_split is None else 'explicit_readiness_subset',
            'validation_reuse_rir': validation_reuse_rir, 'channel_policy': 'first',
            'usage': 'research', 'commercial_clearance': 'pending',
            'rir_alignment': 'upstream-estimated-onset', 'training_started': False,
            'integrity': 'Release/index/source-manifest hashes and archive sizes checked; selected audio member hashes and full decode checked during materialization.'}


def _partition(plan: dict[str, Any]) -> dict[str, Path]:
    root, output = Path(plan['root']), Path(plan['output_dir'])
    destinations = {key: output / 'indices' / f'{key}.jsonl' for key in plan['groups']}
    # Open a bounded number of corpus files, never hold the complete corpus as strings.
    temporary, handles, counts = {}, {}, defaultdict(int)
    try:
        for key, path in destinations.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False)
            handles[key], temporary[key] = handle, Path(handle.name)
        for name, (role, split) in INDEX_ROLES.items():
            count = 0
            for row in _rows(root / plan['indices'][name]['path']):
                if role == 'speech' and plan['max_speech_per_split'] is not None and count >= plan['max_speech_per_split']:
                    break
                key = f'{role}__{split}__{row["source"]}'
                handles[key].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[key] += 1
                count += 1
        for key, handle in handles.items():
            handle.close()
            if counts[key] != plan['groups'][key]['count']:
                raise ValueError('Source index changed after planning')
            with temporary[key].open(encoding='utf-8') as lines:
                _write(destinations[key], lines)
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return destinations


def _progress(output: Path, **values) -> None:
    _write(output / 'progress.json', _json(values), immutable=False)
    print(json.dumps(values), flush=True)


def _worker_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@contextmanager
def _single_threaded_children():
    """Set limits before spawn imports NumPy/Torch, then restore the parent."""
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: '1' for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _materialization_worker_init() -> None:
    # The preparation module imports Torch for independent loader validation.
    # These workers do audio I/O only; prevent nested CPU thread pools.
    import torch
    torch.set_num_threads(1)


def _materialize_group(root: str, path: str, output: str, role: str,
                       wind_id_scope: str) -> dict[str, Any]:
    return bridge.materialize(bridge.make_plan(Path(root), Path(path), Path(output), role,
                                              channel_policy='first', wind_id_scope=wind_id_scope))


def _abort_materializers(executor: ProcessPoolExecutor, pending) -> None:
    for future in pending:
        future.cancel()
    # Python 3.14 exposes terminate_workers; 3.11-3.13 do not. Capture the
    # executor-owned process handles before shutdown, so a failed group does
    # not leave unrelated corpus-sized extraction jobs running in the background.
    if hasattr(executor, 'terminate_workers'):
        executor.terminate_workers()
        return
    processes = list((getattr(executor, '_processes', None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    executor.shutdown(wait=True, cancel_futures=True)


def _materialize_groups(plan, partitions, workers: int, *, backup=None):
    """Bound in-flight jobs to workers and retain deterministic corpus order."""
    output = Path(plan['output_dir'])
    completed = {}

    def task(key, path):
        return (plan['root'], str(path), str(output / 'materialized' / key),
                plan['groups'][key]['role'], plan['groups'][key].get('wind_id_scope', 'global'))

    def completed_group(key, result):
        if backup is not None:
            directory = output / 'materialized' / key
            backup.checkpoint(f'materialized/{key}', [directory],
                              [directory / 'request.json', directory / 'records.jsonl'])
        completed[key] = result

    if workers == 1:
        for index, (key, path) in enumerate(partitions.items()):
            _progress(output, stage='materializing', group=key, completed_groups=index,
                      total_groups=len(partitions), materialize_workers=1)
            completed_group(key, _materialize_group(*task(key, path)))
        return completed

    iterator = iter(partitions.items())
    with _single_threaded_children(), prep._graceful_sigterm():
        executor = ProcessPoolExecutor(max_workers=min(workers, len(partitions)),
            mp_context=multiprocessing.get_context('spawn'), initializer=_materialization_worker_init)
        pending = {}
        try:
            for _ in range(min(workers, len(partitions))):
                key, path = next(iterator)
                pending[executor.submit(_materialize_group, *task(key, path))] = key
            _progress(output, stage='materializing', active_groups=list(pending.values()),
                      completed_groups=0, total_groups=len(partitions), materialize_workers=workers)
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Observe every result before starting more work: if one failed,
                # no new group is launched merely because a peer completed too.
                for future in finished:
                    key = pending.pop(future)
                    completed_group(key, future.result())
                for _ in finished:
                    following = next(iterator, None)
                    if following is not None:
                        key, path = following
                        pending[executor.submit(_materialize_group, *task(key, path))] = key
                _progress(output, stage='materializing', active_groups=list(pending.values()),
                          completed_groups=len(completed), total_groups=len(partitions),
                          materialize_workers=workers)
        except BaseException:
            _abort_materializers(executor, pending)
            raise
        else:
            executor.shutdown(wait=True)
    # Completion order must not change SCP order, seeds, or resume plan hashes.
    return {key: completed[key] for key in partitions}


def _invalidate_ready(output: Path) -> None:
    """A valid resumed request must not advertise readiness while repairing data."""
    ready = output / 'READY.json'
    previous = output / 'last-successful-ready.json'
    if ready.is_symlink() or previous.is_symlink():
        raise ValueError('Readiness markers cannot be symlinks')
    if ready.exists():
        if not ready.is_file():
            raise ValueError('Expected a regular readiness marker')
        os.replace(ready, previous)


def audit_materialized_sources(plan, materialized):
    """Reject train/validation source leakage before expensive pair rendering."""
    seen = {key: {} for key in ('speech_uid', 'speaker_id', 'speech_materialized_sha256',
                                'noise_uid', 'noise_materialized_sha256')}
    counts = defaultdict(int)
    for key, result in materialized.items():
        group = plan['groups'][key]
        role, split = group['role'], group['split']
        if role == 'rir':  # The official recipe shares the same RIR pool.
            continue
        for row in _rows(Path(result['output_dir']) / 'records.jsonl'):
            checksum = row.get('materialized_sha256')
            if not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
                raise ValueError('Materialized source records require verified output hashes')
            identities = ({'speech_uid': row.get('uid'),
                           'speaker_id': row.get('materialized_speaker_id'),
                           'speech_materialized_sha256': checksum} if role == 'speech' else
                          {'noise_uid': row.get('uid'), 'noise_materialized_sha256': checksum})
            # Noise and wind share these dictionaries: identical noise must not
            # cross splits merely by appearing in a different source role.
            for field, identity in identities.items():
                if not identity:
                    raise ValueError(f'Missing materialized source identity: {field}')
                previous_split = seen[field].get(identity)
                if previous_split is not None and previous_split != split:
                    raise ValueError(f'Early split leakage: {field}={identity!r} occurs in {previous_split} and {split}')
                seen[field][identity] = split
            counts[f'{role}_{split}'] += 1
    return {'status': 'passed', 'records': dict(counts), 'checks': list(seen),
            'hash_scope': 'Verified materialized output-file bytes after any channel normalization; different encodings of equivalent audio may evade this check.',
            'rir_policy': 'Shared pool permitted; RIR identity is excluded from leakage checks.'}


def _split_plan(plan, materialized, split):
    source, output = Path(plan['source']), Path(plan['output_dir'])
    config, _ = _template(source, split)
    evidence = []
    config['speech_utt2spk'], config['speech_text'] = [], []
    for field, role in prep.ROLES.items():
        config[field] = []
        for key, result in materialized.items():
            group = plan['groups'][key]
            if group['role'] != role or group['split'] != ('pool' if role == 'rir' else split):
                continue
            config[field].append(result['scp'])
            evidence.append({'id': group['source'], 'role': role, 'scp': result['scp'],
                             'license': '; '.join(group['licenses']), 'commercial_use': None,
                             'license_evidence': {'source_completion': str(Path(plan['root']) / plan['markers'][group['source']]['path']),
                                                  'source_completion_sha256': plan['markers'][group['source']]['sha256'],
                                                  'record_metadata': str(Path(result['output_dir']) / 'records.jsonl')}})
            if role == 'speech':
                config['speech_utt2spk'].append(str(Path(result['output_dir']) / 'utt2spk'))
                config['speech_text'].append(str(Path(result['output_dir']) / 'text'))
    config['repeat_per_utt'] = 1
    if split == 'validation' and plan['validation_reuse_rir']:
        config['reuse_rir'] = True
    settings = output / 'settings' / split
    config_path, provenance_path = settings / 'simulation.yaml', settings / 'provenance.json'
    _write(config_path, yaml.safe_dump(config, sort_keys=False))
    _write(provenance_path, _json({'split': split, 'sources': evidence,
                                  'source_release': str(Path(plan['root']) / 'release/RELEASE.json'),
                                  'selection_scope': plan['selection_scope'], 'commercial_clearance': 'pending'}))
    result = prep.preparation_plan(source, config_path, provenance_path, output / 'chunks' / split,
                                   usage='research', rir_alignment='upstream-estimated-onset')
    # Read text/speaker mappings once per split, not once per rendering chunk.
    result['_speech_lines'] = {}
    for field in ('speech_utt2spk', 'speech_text'):
        mapping = {}
        for filename in config[field]:
            with Path(filename).open() as handle:
                for line in handle:
                    if line.strip():
                        mapping[line.split(maxsplit=1)[0]] = line.rstrip('\n') + '\n'
        result['_speech_lines'][field] = mapping
    return result


def _chunk_plan(base, selected, index, output, used_rirs):
    split = base['provenance']['split']
    directory = output / 'chunk-inputs' / split / f'{index:06d}'
    speech_scp, speakers_path, text_path = directory / 'speech.scp', directory / 'utt2spk', directory / 'text'
    _write(speech_scp, (f'{uid} {base["inventories"]["speech"][uid]["sample_rate"]} {base["inventories"]["speech"][uid]["path"]}\n' for uid in selected))
    _write(speakers_path, (base['_speech_lines']['speech_utt2spk'][uid] for uid in selected))
    _write(text_path, (base['_speech_lines']['speech_text'][uid] for uid in selected
                       if uid in base['_speech_lines']['speech_text']))
    config = deepcopy(base['config'])
    config.update(speech_scps=[str(speech_scp)], speech_utt2spk=[str(speakers_path)], speech_text=[str(text_path)],
                  seed=int(base['config'].get('seed', 0)) + index)
    inventories = {**base['inventories'], 'speech': {uid: base['inventories']['speech'][uid] for uid in selected}}
    if not config.get('reuse_rir', False):
        available = {uid: value for uid, value in base['inventories']['rir'].items() if uid not in used_rirs}
        if not available:
            raise ValueError('Validation RIR pool exhausted; use a distinct output with --validation-reuse-rir if repetition is acceptable')
        for rate in {inventories['speech'][uid]['sample_rate'] for uid in selected}:
            if not any(entry['sample_rate'] >= rate for entry in available.values()):
                raise ValueError(f'Validation RIR pool has no remaining eligible source at or above {rate} Hz for this chunk')
        rir_scp = directory / 'rir.scp'
        _write(rir_scp, (f'{uid} {entry["sample_rate"]} {entry["path"]}\n' for uid, entry in available.items()))
        config['rir_scps'] = [str(rir_scp)]
        inventories['rir'] = available
    destination = output / 'chunks' / split / f'{index:06d}'
    config.update(output_dir=str(destination), log_dir=str(destination / 'log'))
    return {**{key: value for key, value in base.items() if not key.startswith('_')},
            'config': config, 'inventories': inventories, 'output_dir': str(destination),
            'chunk': {'index': index, 'utterances': len(selected), 'speech_scp_sha256': prep.sha256(speech_scp),
                      'seed_policy': 'official_base_seed_plus_chunk_index', 'validation_rir_reuse_across_chunks': bool(config.get('reuse_rir'))}}


def _combine(manifests: list[Path], destination: Path, split: str) -> int:
    counts = [0]
    def lines():
        for index, manifest in enumerate(manifests):
            for row in _rows(manifest):
                if row.get('split') != split:
                    raise ValueError('Rendered manifest has incorrect split')
                row['id'] = f'{split}:chunk{index:06d}:{row["id"]}'
                row['chunk_manifest'] = str(manifest)
                counts[0] += 1
                yield json.dumps(row, ensure_ascii=False) + '\n'
    _write(destination, lines())
    return counts[0]


class _ChunkBackups:
    """Bounded closed chunks in backup while the next chunk renders locally.

    Source materialization stays synchronous: its remote archive reads must not
    compete with checkpoint I/O. All backup workers join before final publication,
    so the final receipt includes every preceding checkpoint dependency.
    """
    def __init__(self, backup, overlap: bool, workers: int = 1):
        self.backup, self.overlap = backup, overlap
        self.workers = workers
        self.executor, self.pending = None, deque()

    def __enter__(self):
        if self.overlap:
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='chunk-backup')
        return self

    def drain(self):
        failure = None
        while self.pending:
            try:
                self.pending[0].result()
            except BaseException as error:
                if failure is None:
                    failure = error
            finally:
                self.pending.popleft()
        if failure is not None:
            raise failure

    def check(self):
        # A later failure must not hide behind an earlier slow checkpoint.
        for future in self.pending:
            if future.done():
                future.result()
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()

    def checkpoint(self, unit, paths, identity_paths):
        if self.executor is None:
            self.backup.checkpoint(unit, paths, identity_paths)
        else:
            self.check()
            if len(self.pending) >= self.workers:
                self.pending[0].result()
                self.pending.popleft()
                self.check()
            # No executor queue beyond the configured active checkpoint count.
            self.pending.append(self.executor.submit(self.backup.checkpoint, unit, paths, identity_paths))

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.drain()
        except BaseException as backup_error:
            if exc_type is None:
                raise
            # Preserve a render/cancellation failure while still observing the
            # background result. The closed chunk remains resumable on disk.
            if hasattr(exc, 'add_note'):
                exc.add_note(f'Chunk backup also failed during cleanup: {backup_error!r}')
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        return False


def execute_release(plan: dict[str, Any], *, workers: int = 1,
                    materialize_workers: int = 1,
                    backup_dir: Path | None = None,
                    overlap_backups: bool = False,
                    backup_compression: str = 'none',
                    backup_workers: int = 1) -> dict[str, Any]:
    workers = _worker_count(workers, 'workers')
    materialize_workers = _worker_count(materialize_workers, 'materialize_workers')
    if not isinstance(overlap_backups, bool):
        raise ValueError('overlap_backups must be a boolean')
    if overlap_backups and backup_dir is None:
        raise ValueError('Overlapping backups requires a backup directory')
    if isinstance(backup_workers, bool) or not isinstance(backup_workers, int) or backup_workers not in (1, 2):
        raise ValueError('backup_workers must be 1 or 2')
    if backup_workers != 1 and not overlap_backups:
        raise ValueError('Two backup workers requires overlap_backups')
    if backup_compression not in ('none', 'gzip'):
        raise ValueError('Backup compression must be none or gzip')
    if backup_compression != 'none' and backup_dir is None:
        raise ValueError('Backup compression requires a backup directory')
    output = Path(plan['output_dir'])
    if backup_dir is not None:
        backup_dir = Path(backup_dir).expanduser().resolve()
        if backup_dir == output or backup_dir.is_relative_to(output) or output.is_relative_to(backup_dir):
            raise ValueError('Backup and preparation directories must not overlap')
    # Revalidate the exact frozen request, including all finalized input hashes.
    current = release_plan(Path(plan['root']), Path(plan['source']), output,
                           chunk_size=plan['chunk_size'], max_speech_per_split=plan['max_speech_per_split'],
                           validation_reuse_rir=plan['validation_reuse_rir'])
    if current != plan:
        raise ValueError('Source release changed after planning')
    if output.exists() and not (output / 'request.json').is_file():
        raise ValueError('Existing output lacks a matching preparation request')
    _write(output / 'request.json', _json(plan))
    backup = None
    if backup_dir is not None:
        from .preparation_backup import PreparationBackup
        backup = PreparationBackup(output, backup_dir, compression=backup_compression)
    _invalidate_ready(output)
    # Concurrency is operational state, not part of the frozen data recipe.
    # Recording it separately lets a resumed job change its CPU allocation.
    _write(output / 'execution.json', _json({'render_workers': workers,
        'materialize_workers': materialize_workers, 'process_start_method': 'spawn',
        **({'backup_dir': str(backup_dir), 'overlap_backups': overlap_backups,
            'backup_compression': backup_compression, 'backup_workers': backup_workers}
           if backup_dir is not None else {}),
        'training_started': False}), immutable=False)
    _progress(output, stage='partitioning', selection=plan['selection_scope'], selected_counts=plan['selected_counts'])
    partitions = _partition(plan)
    if backup is not None:
        backup.checkpoint('root', [output / 'request.json', output / 'indices'],
                          [output / 'request.json', *partitions.values()])
    materialized = _materialize_groups(plan, partitions, materialize_workers, backup=backup)
    _progress(output, stage='auditing_materialized_sources')
    source_audit = audit_materialized_sources(plan, materialized)
    _write(output / 'source-split-audit.json', _json(source_audit))
    combined, counts, rendered = {}, {}, {}
    # Validate/render the constrained validation RIR recipe before committing
    # potentially days of work to the much larger training split.
    signal_guard = prep._graceful_sigterm() if overlap_backups else nullcontext()
    with signal_guard, _ChunkBackups(backup, overlap_backups, backup_workers) as checkpoints:
        for split in ('validation', 'train'):
            checkpoints.check()
            base = _split_plan(plan, materialized, split)
            speech = list(base['inventories']['speech'])
            size, used_rirs, manifests = plan['chunk_size'], set(), []
            for index, start in enumerate(range(0, len(speech), size)):
                checkpoints.check()  # Stop before more work if backup already failed.
                selected = speech[start:start + size]
                _progress(output, stage='rendering', split=split, chunk=index,
                          completed_speech=start, total_speech=len(speech))
                chunk = _chunk_plan(base, selected, index, output, used_rirs)
                manifest = prep.execute_preparation(chunk, resume=True, workers=workers)
                observed = set()
                for row in _rows(manifest):
                    observed.add(row['speech_uid'])
                    if row.get('rir_uid') != 'none':
                        used_rirs.add(row['rir_uid'])
                if observed != set(selected):
                    raise ValueError('Rendered chunk did not cover exactly the selected speech')
                if backup is not None:
                    directory = Path(chunk['output_dir'])
                    inputs = output / 'chunk-inputs' / split / f'{index:06d}'
                    settings = output / 'settings' / split
                    # These files define the samples and their resume identity.
                    # provenance.json includes changing timings/reuse counts and is
                    # archived for diagnosis, but must not force a new checkpoint.
                    identity = [directory / name for name in (
                        'plan.json', 'simulation.yaml', 'render-runtime.json',
                        'render-progress.jsonl', 'manifest.jsonl', 'log/meta.tsv')]
                    identity.extend(path for parent in (inputs, settings)
                                    for path in sorted(parent.rglob('*')) if path.is_file())
                    checkpoints.checkpoint(f'chunks/{split}/{index:06d}',
                                           [directory, inputs, settings], identity)
                manifests.append(manifest)
            rendered[split] = manifests
    # Neither final split manifest is published until every chunk of both
    # splits completed. READY.json remains the final release commit marker.
    pending = {}
    for split, manifests in rendered.items():
        pending[split] = output / 'manifests' / f'{split}.pending.jsonl'
        counts[split] = _combine(manifests, pending[split], split)
        if counts[split] != plan['selected_counts'][f'speech_{split}']:
            raise ValueError('Expected exactly one rendered pair per source utterance')
    _progress(output, stage='auditing', examples=counts)
    audit = prep.audit_splits(list(pending.values()))
    _write(output / 'split-audit.json', _json(audit))
    for split, path in pending.items():
        combined[split] = output / 'manifests' / f'{split}.jsonl'
        with path.open() as handle:
            _write(combined[split], handle)
    hydra = {'data': {'train_manifest': str(combined['train']), 'validation_manifest': str(combined['validation']),
                      'validation_full_utterance': True}}
    _write(output / 'data-paths.yaml', yaml.safe_dump(hydra, sort_keys=False))
    result = {'status': 'paired_manifests_ready', 'training_pairs_ready': True, 'training_started': False,
              'usage': 'research', 'commercial_clearance': 'pending', 'selection_scope': plan['selection_scope'],
              'source_release_status': plan['source_release_status'], 'incomplete_sources': plan['incomplete_sources'],
              'omitted_sources': plan['omitted_sources'], 'examples': counts, 'audit': audit,
              'source_audit': source_audit,
              'request_sha256': prep.sha256(output / 'request.json'), 'rir_alignment': plan['rir_alignment'],
              'manifests': {split: {'path': str(path), 'sha256': prep.sha256(path)} for split, path in combined.items()},
              'hydra_data_paths': str(output / 'data-paths.yaml')}
    if backup is None:
        _write(output / 'READY.json', _json(result))
    else:
        # The backup stores this staged marker as READY.json. Publish the local
        # marker only after every required unit and the final receipt commit.
        ready = output / 'READY.pending.json'
        _write(ready, _json(result))
        final_paths = [output / 'manifests', output / 'source-split-audit.json',
                       output / 'split-audit.json', output / 'data-paths.yaml', ready]
        final_identity = [*combined.values(), *final_paths[1:]]
        backup.checkpoint('final', final_paths, final_identity)
        os.replace(ready, output / 'READY.json')
    _progress(output, stage='ready', examples=counts, training_started=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Source root containing release/RELEASE.json')
    parser.add_argument('--source', type=Path, required=True, help='Pristine pinned official URGENT checkout')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel official renderer processes per chunk; runtime setting, not a recipe change')
    parser.add_argument('--materialize-workers', type=int, default=1,
                        help='Parallel independent source-group extraction processes; runtime setting, not a recipe change')
    parser.add_argument('--backup-dir', type=Path,
                        help='Persistent checkpoints outside the local output; runtime setting, not a recipe change')
    parser.add_argument('--overlap-backups', action='store_true',
                        help='Back up closed chunks while the next renders locally; requires --backup-dir')
    parser.add_argument('--backup-workers', type=int, choices=(1, 2), default=1,
                        help='Concurrent checkpoints; 2 requires --overlap-backups')
    parser.add_argument('--backup-compression', choices=('none', 'gzip'), default='none',
                        help='Lossless checkpoint compression; gzip uses level1 and requires --backup-dir')
    parser.add_argument('--usage', choices=('research',), default='research',
                        help='This release route retains pending permission evidence; commercial preparation uses the audited adapter separately')
    parser.add_argument('--rir-alignment', choices=('upstream-estimated-onset',), default='upstream-estimated-onset')
    parser.add_argument('--max-speech-per-split', type=int, help='Explicit first-N readiness subset; use a distinct output')
    parser.add_argument('--validation-reuse-rir', action='store_true', help='Explicitly permit validation RIR reuse; changes the official validation recipe')
    parser.add_argument('--execute', action='store_true', help='Prepare local audio and paired manifests; never train')
    args = parser.parse_args()
    _worker_count(args.workers, 'workers')
    _worker_count(args.materialize_workers, 'materialize_workers')
    plan = release_plan(args.root, args.source, args.output_dir, chunk_size=args.chunk_size,
                        max_speech_per_split=args.max_speech_per_split, validation_reuse_rir=args.validation_reuse_rir)
    print(_json(execute_release(plan, workers=args.workers, materialize_workers=args.materialize_workers,
                               backup_dir=args.backup_dir, overlap_backups=args.overlap_backups,
                               backup_compression=args.backup_compression, backup_workers=args.backup_workers)
                if args.execute else plan))


if __name__ == '__main__':
    main()
