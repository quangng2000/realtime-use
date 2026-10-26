"""Immutable preparation checkpoints; backup storage needs no rename or locks.

Only closed preparation units may be checkpointed. Writers stream a new tar,
verify it by reading it back, then publish a unique receipt. Restore verifies all
selected archives before extracting into local staging and publishes READY last.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import time
from typing import Any
import uuid


_FORMAT = 1
_BLOCK = 1024 * 1024


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(_BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or not path.parts:
        raise ValueError(f'Unconfined archive path: {value!r}')
    return Path(*path.parts)


def _confined(root: Path, path: Path, *, exists: bool = True) -> Path:
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Checkpoint path is outside output root: {path}') from exc
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f'Checkpoint paths cannot contain symlinks: {path}')
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return relative


def _receipts(backup: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted((backup / 'commits').glob('*.json')):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Backup receipt must be a regular file: {path}')
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # A partial write is not a commit.
        required = {'format_version', 'status', 'unit', 'output_root', 'request_sha256',
                    'identity_sha256', 'identity_files', 'created_ns', 'archive', 'receipt_sha256', 'dependencies'}
        if not isinstance(value, dict) or value.get('status') != 'committed' or not required <= value.keys():
            continue
        checksum = value['receipt_sha256']
        if checksum != _hash_json({key: item for key, item in value.items() if key != 'receipt_sha256'}):
            raise ValueError(f'Committed backup receipt checksum mismatch: {path}')
        if value['format_version'] != _FORMAT or not isinstance(value['created_ns'], int):
            raise ValueError(f'Unsupported backup receipt: {path}')
        archive = value['archive']
        if (not isinstance(archive, dict) or set(archive) != {'path', 'bytes', 'sha256'}
                or not isinstance(archive['bytes'], int) or archive['bytes'] < 0
                or not isinstance(value['unit'], str) or not value['unit']
                or not isinstance(value['identity_files'], dict) or not isinstance(value['dependencies'], dict)):
            raise ValueError(f'Malformed committed backup receipt: {path}')
        identity = _hash_json({'unit': value['unit'], 'identity_files': value['identity_files'],
                               'dependencies': value['dependencies']})
        if value['identity_sha256'] != identity:
            raise ValueError(f'Committed backup semantic identity mismatch: {path}')
        relative = _relative(archive['path'])
        if len(relative.parts) != 2 or relative.parts[0] != 'objects' or relative.suffix != '.tar':
            raise ValueError(f'Invalid backup archive location: {path}')
        result.append(value)
    return result


def _latest(receipts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for receipt in sorted(receipts, key=lambda item: (item['created_ns'], item['receipt_sha256'])):
        result[receipt['unit']] = receipt
    return result


def _verify_archive(backup: Path, receipt: dict[str, Any]) -> Path:
    archive = receipt['archive']
    path = backup / _relative(archive['path'])
    _confined(backup, path)
    if not path.is_file() or path.stat().st_size != archive['bytes'] or _digest(path) != archive['sha256']:
        raise ValueError(f'Committed backup archive integrity mismatch: {path}')
    return path


class _HashingWriter:
    def __init__(self, handle):
        self.handle, self.digest, self.bytes = handle, hashlib.sha256(), 0

    def write(self, data):
        count = self.handle.write(data)
        if count != len(data):
            raise OSError('Incomplete backup archive write')
        self.digest.update(data)
        self.bytes += count
        return count


class PreparationBackup:
    """Back up completed units; concurrency settings never change recipe identity."""
    def __init__(self, output_root: Path, backup_dir: Path):
        self.output_root = output_root.expanduser().resolve(strict=True)
        self.backup_dir = backup_dir.expanduser().resolve()
        if not self.output_root.is_dir():
            raise ValueError('Preparation output root must be a directory')
        if (self.backup_dir.is_relative_to(self.output_root)
                or self.output_root.is_relative_to(self.backup_dir)):
            raise ValueError('Backup and active output directories must not contain one another')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        for name in ('objects', 'commits'):
            path = self.backup_dir / name
            if path.is_symlink():
                raise ValueError('Backup directories cannot be symlinks')
            path.mkdir(exist_ok=True)
        self._receipts = _receipts(self.backup_dir)

    def _identity(self, paths: list[Path], *, final: bool = False) -> tuple[str, dict[str, str]]:
        request = self.output_root / 'request.json'
        _confined(self.output_root, request)
        request_hash = _digest(request)
        files = {}
        for path in [request, *paths]:
            relative = _confined(self.output_root, path)
            if not path.is_file():
                raise ValueError('Checkpoint semantic identity paths must be regular files')
            name = relative.as_posix()
            if final and name == 'READY.pending.json':
                name = 'READY.json'
            if name in files and files[name] != _digest(path):
                raise ValueError('Conflicting checkpoint identity paths')
            files[name] = _digest(path)
        return request_hash, files

    def checkpoint(self, unit: str, paths: list[Path], identity_paths: list[Path]) -> dict[str, Any]:
        if not isinstance(unit, str) or not unit or '\x00' in unit:
            raise ValueError('Checkpoint unit must be a nonempty name')
        request_hash, identities = self._identity(identity_paths, final=unit == 'final')
        for receipt in self._receipts:
            if receipt['output_root'] != str(self.output_root) or receipt['request_sha256'] != request_hash:
                raise ValueError('Backup contains checkpoints from a different output root or preparation request')
        latest = _latest(self._receipts)
        dependencies = ({name: item['receipt_sha256'] for name, item in latest.items() if name != 'final'}
                        if unit == 'final' else {})
        identity = _hash_json({'unit': unit, 'identity_files': identities, 'dependencies': dependencies})
        previous = latest.get(unit)
        if previous is not None and previous['identity_sha256'] == identity:
            _verify_archive(self.backup_dir, previous)
            return previous
        members = {}
        for path in [self.output_root / 'request.json', *paths, *identity_paths]:
            _confined(self.output_root, path)
            candidates = [path, *path.rglob('*')] if path.is_dir() else [path]
            for member in candidates:
                relative = _confined(self.output_root, member)
                mode = member.stat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f'Checkpoint member must be a regular file or directory: {member}')
                if relative.parts:
                    name = relative.as_posix()
                    if unit == 'final' and name == 'READY.pending.json':
                        name = 'READY.json'
                    if name in members and members[name] != member:
                        raise ValueError('Conflicting readiness checkpoint paths')
                    members[name] = member
        if ('READY.json' in members or 'READY.pending.json' in members) and unit != 'final':
            raise ValueError('READY.json may only appear in the final checkpoint')
        if unit == 'final' and ('READY.json' not in members or not dependencies):
            raise ValueError('Final checkpoint requires READY.json and preceding unit receipts')
        token = uuid.uuid4().hex
        archive_relative = f'objects/{token}.tar'
        archive_path = self.backup_dir / archive_relative
        # No rename, lock, or remote fsync is needed: this unique object remains
        # uncommitted until a complete descriptor has been written and read back.
        with archive_path.open('xb') as destination:
            writer = _HashingWriter(destination)
            with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT, bufsize=_BLOCK) as archive:
                for name, path in sorted(members.items()):
                    before = path.stat()
                    info = tarfile.TarInfo(name)
                    info.mtime = int(before.st_mtime)
                    info.mode = stat.S_IMODE(before.st_mode) & 0o777
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        archive.addfile(info)
                    else:
                        info.size = before.st_size
                        with path.open('rb') as source:
                            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                                raise ValueError(f'Checkpoint member changed type: {path}')
                            archive.addfile(info, source)
                        after = path.stat()
                        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                            raise ValueError(f'Checkpoint member changed during backup: {path}')
            destination.flush()
        descriptor = {'path': archive_relative, 'bytes': writer.bytes, 'sha256': writer.digest.hexdigest()}
        if archive_path.stat().st_size != writer.bytes or _digest(archive_path) != descriptor['sha256']:
            raise ValueError('Backup archive read-back integrity check failed')
        if self._identity(identity_paths, final=unit == 'final') != (request_hash, identities):
            raise ValueError('Checkpoint semantic identity changed during backup')
        receipt = {'format_version': _FORMAT, 'status': 'committed', 'unit': unit,
                   'output_root': str(self.output_root), 'request_sha256': request_hash,
                   'identity_files': identities, 'identity_sha256': identity, 'dependencies': dependencies,
                   'archive': descriptor, 'created_ns': time.time_ns()}
        receipt['receipt_sha256'] = _hash_json(receipt)
        encoded = _json(receipt)
        commit = self.backup_dir / 'commits' / f'{token}.json'
        with commit.open('xb') as handle:
            handle.write(encoded)
            handle.flush()
        if commit.read_bytes() != encoded:
            raise ValueError('Backup receipt read-back check failed')
        self._receipts.append(receipt)
        return receipt


def _extract(archive_path: Path, staging: Path) -> set[str]:
    """Confine members explicitly; never use tarfile.extract/extractall."""
    seen = set()
    with tarfile.open(archive_path, mode='r|', bufsize=_BLOCK) as archive:
        for member in archive:
            relative = _relative(member.name)
            name = relative.as_posix()
            if name in seen:
                raise ValueError(f'Duplicate archive member: {member.name}')
            seen.add(name)
            if not (member.isdir() or member.isfile()):
                raise ValueError(f'Archive links/special files are forbidden: {member.name}')
            destination = staging / relative
            _confined(staging, destination, exists=False)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    try:
                        with archive.extractfile(member) as source:
                            shutil.copyfileobj(source, handle, _BLOCK)
                        handle.flush()
                        os.fsync(handle.fileno())
                        if handle.tell() != member.size:
                            raise ValueError(f'Truncated archive member: {member.name}')
                        handle.close()
                        if destination.exists():
                            if not destination.is_file() or _digest(destination) != _digest(temporary):
                                raise ValueError(f'Conflicting checkpoint snapshots: {member.name}')
                        else:
                            temporary.replace(destination)
                            destination.chmod(member.mode & 0o777)
                    finally:
                        temporary.unlink(missing_ok=True)
            archive.members.clear()
    return seen


def restore(backup_dir: Path, output_root: Path) -> dict[str, Any]:
    """Restore one compatible run at its original absolute path; publish READY last."""
    backup, output = backup_dir.expanduser().resolve(strict=True), output_root.expanduser().resolve()
    if backup.is_relative_to(output) or output.is_relative_to(backup):
        raise ValueError('Backup and restore output must not contain one another')
    receipts = _receipts(backup)
    if not receipts:
        raise ValueError('No committed preparation checkpoints found')
    if {item['output_root'] for item in receipts} != {str(output)}:
        raise ValueError('Restore requires the original exact absolute output root')
    requests = {item['request_sha256'] for item in receipts}
    if len(requests) != 1:
        raise ValueError('Backup contains incompatible preparation requests')
    request_hash = requests.pop()
    selected = _latest(receipts)
    final = selected.get('final')
    expected_dependencies = {unit: item['receipt_sha256'] for unit, item in selected.items() if unit != 'final'}
    if final is not None and final['dependencies'] != expected_dependencies:
        del selected['final']  # A resumed/incomplete run must not restore stale READY.
    ordered = sorted(selected, key=lambda unit: (unit == 'final', unit))
    archives = {unit: _verify_archive(backup, selected[unit]) for unit in ordered}
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError('Restore output must be a local directory')
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=output) as scratch:
        staging = Path(scratch)
        for unit in ordered:
            members = _extract(archives[unit], staging)
            for name, expected in selected[unit]['identity_files'].items():
                path = staging / _relative(name)
                if name not in members or not path.is_file() or _digest(path) != expected:
                    raise ValueError(f'Restored semantic identity mismatch for {unit}: {name}')
        request = staging / 'request.json'
        if not request.is_file() or _digest(request) != request_hash:
            raise ValueError('Restored request differs from committed recipe identity')
        if ('final' in selected) != (staging / 'READY.json').is_file():
            raise ValueError('Readiness marker must belong to the final committed checkpoint')
        if (output / 'READY.json').exists() and 'final' not in selected:
            raise ValueError('Existing READY conflicts with an incomplete checkpoint restore')
        files, directories = [], []
        for path in staging.rglob('*'):
            relative = path.relative_to(staging)
            destination = output / relative
            _confined(output, destination, exists=False)
            if path.is_dir():
                if destination.exists() and not destination.is_dir():
                    raise FileExistsError(f'Restore directory conflicts with existing file: {destination}')
                directories.append(destination)
            else:
                if destination.exists() and (not destination.is_file() or _digest(destination) != _digest(path)):
                    raise FileExistsError(f'Restore refuses conflicting existing file: {destination}')
                files.append((relative.as_posix(), path, destination))
        for directory in sorted(directories, key=lambda path: len(path.parts)):
            directory.mkdir(parents=True, exist_ok=True)
        restored, reused = 0, 0
        for name, source, destination in sorted(files, key=lambda entry: (entry[0] == 'READY.json', entry[0])):
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Exclusive publication on local POSIX disk; never silently
                # replace a conflicting file created after the preflight.
                os.link(source, destination, follow_symlinks=False)
                restored += 1
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file() or _digest(destination) != _digest(source):
                    raise FileExistsError(f'Restore refuses conflicting existing file: {destination}')
                reused += 1
    return {'output_root': str(output), 'request_sha256': request_hash, 'units': ordered,
            'restored_files': restored, 'reused_files': reused,
            'ready_restored': 'final' in selected, 'training_started': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('restore', help='Verify and restore preparation checkpoints; never train')
    command.add_argument('--backup-dir', type=Path, required=True)
    command.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.backup_dir, args.output_root), indent=2))


if __name__ == '__main__':
    main()
