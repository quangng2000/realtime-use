"""Release-to-pair integration fixtures contain generated audio; never train."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest
import soundfile as sf
import yaml

from realtime_use import prepare_release as release


def fixture_release(tmp_path, monkeypatch):
    root, source = tmp_path / 'dataset', tmp_path / 'upstream'
    (root / 'release').mkdir(parents=True)
    source.mkdir()
    markers, index_artifacts = defaultdict(list), []
    serial = 0
    for name, (role, split) in release.INDEX_ROLES.items():
        corpus = f'fixture_{role}'
        directory = root / 'sources' / corpus
        directory.mkdir(parents=True, exist_ok=True)
        shard = directory / f'{split}.tar'
        rows = []
        count = 3 if role in ('speech', 'rir') else 1
        with tarfile.open(shard, 'w') as archive:
            for index in range(count):
                serial += 1
                data = io.BytesIO()
                sf.write(data, np.linspace(-serial / 100, serial / 100, 80 + serial),
                         8000, format='WAV', subtype='FLOAT')
                raw = data.getvalue()
                uid = f'wind_noise_{split}_{index}' if role == 'wind' else f'{split}_{index}'
                member = f'audio/{uid}.wav'
                info = tarfile.TarInfo(member)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
                rows.append({'id': f'{corpus}/{split}/{uid}', 'source_id': uid, 'source': corpus,
                             'split': split, 'kind': 'noise' if role == 'wind' else role,
                             'sample_rate': 8000, 'frames': 80 + serial, 'channels': 1,
                             'sha256': hashlib.sha256(raw).hexdigest(), 'shard': str(shard.relative_to(root)),
                             'path': member, 'speaker_id': f'{corpus}:{split}:{index}' if role == 'speech' else None,
                             'text': 'Generated transcript' if role == 'speech' else None, 'license': 'fixture-only'})
        source_manifest = directory / f'{split}.jsonl'
        source_manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        index_manifest = root / 'release' / f'{name}.jsonl'
        index_manifest.write_bytes(source_manifest.read_bytes())
        def artifact(path):
            return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                    'sha256': release.prep.sha256(path)}
        markers[corpus].extend([artifact(shard), artifact(source_manifest)])
        index_artifacts.append(artifact(index_manifest))
    for corpus, artifacts in markers.items():
        (root / 'sources' / corpus / 'SOURCE_READY.json').write_text(json.dumps({'status': 'verified', 'artifacts': artifacts}))
    (root / 'release/RELEASE.json').write_text(json.dumps({'status': 'partial_source_pools',
        'upstream_commit': release.prep.URGENT_COMMIT, 'artifacts': index_artifacts,
        'failed_or_incomplete_sources': {'unavailable_source': 'fixture deliberately partial'},
        'omitted': {'licensed_source': 'not supplied'}}))
    config = {'prob_reverberation': .5, 'prob_wind_noise': .1, 'num_augmentations': {1: 1.0},
              'augmentations': {key: {'weight': 1} for key in ('clipping', 'bandwidth_limitation', 'codec', 'packet_loss')},
              'seed': 0, 'reuse_noise': True, 'repeat_per_utt': 1}
    monkeypatch.setattr(release.prep, 'verify_source', lambda path: None)
    monkeypatch.setattr(release, '_template', lambda path, split: ({**deepcopy(config), 'reuse_rir': split == 'train'}, 'fixture-template-hash'))
    return root, source


def fake_renderer(monkeypatch):
    calls = []
    def execute(plan, *, resume=False, workers=1):
        assert resume is True
        output = Path(plan['output_dir'])
        calls.append(plan)
        execute.worker_counts.append(workers)
        if (output / 'manifest.jsonl').exists():
            assert json.loads((output / 'test-plan.json').read_text()) == json.loads(json.dumps(plan))
            return output / 'manifest.jsonl'
        output.mkdir(parents=True)
        (output / 'test-plan.json').write_text(json.dumps(plan))
        (output / 'plan.json').write_text(json.dumps(plan))
        (output / 'simulation.yaml').write_text(yaml.safe_dump(plan['config']))
        (output / 'render-runtime.json').write_text('{"fixture": "1"}\n')
        (output / 'log').mkdir()
        (output / 'log/meta.tsv').write_text('fixture deterministic metadata\n')
        split = plan['provenance']['split']
        noise_uid, noise = next(iter(plan['inventories']['noise'].items()))
        rir_uid = next(iter(plan['inventories']['rir']))
        speakers = {}
        for filename in plan['config']['speech_utt2spk']:
            speakers.update(line.split() for line in Path(filename).read_text().splitlines())
        rows = []
        for index, (uid, entry) in enumerate(plan['inventories']['speech'].items()):
            rows.append({'id': f'{split}:fileid_{index}', 'split': split, 'speech_uid': uid,
                         'speaker_id': speakers[uid], 'source_speech_sha256': release.prep.sha256(Path(entry['path'])),
                         'noise_uid': noise_uid, 'source_noise_sha256': release.prep.sha256(Path(noise['path'])),
                         'rir_uid': rir_uid, 'sample_rate': entry['sample_rate'],
                         'noisy': entry['path'], 'clean': entry['path']})
        manifest = output / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (output / 'render-progress.jsonl').write_text(manifest.read_text())
        return manifest
    execute.worker_counts = []
    monkeypatch.setattr(release.prep, 'execute_preparation', execute)
    return calls, execute


def test_plan_is_read_only_defaults_to_all_and_reports_partial_release(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    output = tmp_path / 'pairs'
    plan = release.release_plan(root, source, output)
    assert not output.exists()
    assert plan['selection_scope'] == 'all_released_speech'
    assert plan['selected_counts']['speech_train'] == 3
    assert plan['incomplete_sources'] == {'unavailable_source': 'fixture deliberately partial'}
    assert plan['usage'] == 'research' and plan['commercial_clearance'] == 'pending'
    subset = release.release_plan(root, source, tmp_path / 'subset', max_speech_per_split=1)
    assert subset['selected_counts']['speech_train'] == 1
    assert subset['selection_scope'] == 'explicit_readiness_subset'


def test_release_integrity_and_completion_marker_checked(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    manifest = root / 'release/speech_train.jsonl'
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='integrity'):
        release.release_plan(root, source, tmp_path / 'out')


def test_pipeline_materializes_renders_audits_and_resumes_without_rewriting_audio(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    plan = release.release_plan(root, source, tmp_path / 'out', chunk_size=1)
    result = release.execute_release(plan)
    assert result['examples'] == {'train': 3, 'validation': 3}
    assert result['training_started'] is False and result['training_pairs_ready'] is True
    assert result['commercial_clearance'] == 'pending'
    assert len(calls) == 6
    assert [call['provenance']['split'] for call in calls] == ['validation'] * 3 + ['train'] * 3
    validation = [call for call in calls if call['provenance']['split'] == 'validation']
    assert [len(call['inventories']['rir']) for call in validation] == [3, 2, 1]
    assert all(entry['commercial_use'] is None for call in calls for entry in call['provenance']['sources'])
    train_manifest = Path(result['manifests']['train']['path'])
    assert len({row['id'] for row in release._rows(train_manifest)}) == 3
    data_config = yaml.safe_load(Path(result['hydra_data_paths']).read_text())
    assert data_config['data']['train_manifest'] == str(train_manifest)
    audio = list((tmp_path / 'out/materialized').glob('*/audio/*'))
    mtimes = {path: path.stat().st_mtime_ns for path in audio}
    assert release.execute_release(plan) == result
    assert {path: path.stat().st_mtime_ns for path in audio} == mtimes


def test_interrupted_chunk_does_not_publish_final_manifests_and_can_resume(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    _, renderer = fake_renderer(monkeypatch)
    def interrupt(plan, *, resume=False, workers=1):
        if plan['provenance']['split'] == 'validation' and plan['chunk']['index'] == 1:
            raise RuntimeError('simulated interruption')
        return renderer(plan, resume=resume, workers=workers)
    monkeypatch.setattr(release.prep, 'execute_preparation', interrupt)
    plan = release.release_plan(root, source, tmp_path / 'out', chunk_size=1)
    with pytest.raises(RuntimeError, match='interruption'):
        release.execute_release(plan)
    assert not (tmp_path / 'out/manifests/train.jsonl').exists()
    assert not (tmp_path / 'out/READY.json').exists()
    monkeypatch.setattr(release.prep, 'execute_preparation', renderer)
    assert release.execute_release(plan)['training_pairs_ready'] is True


def test_changed_subset_request_rejected_in_existing_output(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    plan = release.release_plan(root, source, tmp_path / 'out', chunk_size=1, max_speech_per_split=1)
    release.execute_release(plan)
    full = release.release_plan(root, source, tmp_path / 'out', chunk_size=1)
    with pytest.raises(ValueError, match='changed existing output'):
        release.execute_release(full)
    assert (tmp_path / 'out/READY.json').is_file()


def test_split_leakage_prevents_ready_publication(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    _, renderer = fake_renderer(monkeypatch)
    def leak(plan, *, resume=False, workers=1):
        path = renderer(plan, resume=resume, workers=workers)
        records = list(release._rows(path))
        for row in records:
            row['speaker_id'] = 'shared-across-splits'
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        return path
    monkeypatch.setattr(release.prep, 'execute_preparation', leak)
    with pytest.raises(ValueError, match='Split leakage'):
        release.execute_release(release.release_plan(root, source, tmp_path / 'out', chunk_size=1))
    assert not (tmp_path / 'out/READY.json').exists()
    assert not (tmp_path / 'out/manifests/train.jsonl').exists()


def test_ready_is_invalidated_before_resume_repairs_and_republished_after_success(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    _, renderer = fake_renderer(monkeypatch)
    plan = release.release_plan(root, source, tmp_path / 'out', chunk_size=1)
    first = release.execute_release(plan)
    def interrupted_repair(chunk, *, resume=False, workers=1):
        assert not (tmp_path / 'out/READY.json').exists()
        raise RuntimeError('repair interrupted')
    monkeypatch.setattr(release.prep, 'execute_preparation', interrupted_repair)
    with pytest.raises(RuntimeError, match='repair interrupted'):
        release.execute_release(plan)
    assert not (tmp_path / 'out/READY.json').exists()
    assert json.loads((tmp_path / 'out/last-successful-ready.json').read_text()) == first
    monkeypatch.setattr(release.prep, 'execute_preparation', renderer)
    assert release.execute_release(plan) == first
    assert (tmp_path / 'out/READY.json').is_file()


@pytest.mark.parametrize('role,field,reference_role', [
    ('speech', 'uid', 'speech'),
    ('speech', 'materialized_speaker_id', 'speech'),
    ('speech', 'materialized_sha256', 'speech'),
    ('noise', 'uid', 'noise'),
    ('wind', 'materialized_sha256', 'noise'),
])
def test_materialized_leakage_rejected_before_any_render(tmp_path, monkeypatch, role, field, reference_role):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    materialize = release.bridge.materialize
    train_reference = {}
    def alias_metadata(plan):
        result = materialize(plan)
        path = Path(result['output_dir']) / 'records.jsonl'
        rows = list(release._rows(path))
        if plan['split'] == 'train':
            train_reference[plan['role']] = rows[0]
        if plan['split'] == 'validation' and plan['role'] == role:
            rows[0][field] = train_reference[reference_role][field]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return result
    monkeypatch.setattr(release.bridge, 'materialize', alias_metadata)
    with pytest.raises(ValueError, match='Early split leakage'):
        release.execute_release(release.release_plan(root, source, tmp_path / 'out', chunk_size=1))
    assert calls == []
    assert not (tmp_path / 'out/READY.json').exists()


def test_validation_rir_rate_exhaustion_fails_before_training_render(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    split_plan = release._split_plan
    def exhausted_rate(plan, materialized, split):
        result = split_plan(plan, materialized, split)
        if split == 'validation':
            # Some RIRs remain, but none can serve the next speech rate.
            for entry in result['inventories']['speech'].values():
                entry['sample_rate'] = 16000
        return result
    monkeypatch.setattr(release, '_split_plan', exhausted_rate)
    with pytest.raises(ValueError, match='no remaining eligible source at or above 16000'):
        release.execute_release(release.release_plan(root, source, tmp_path / 'out', chunk_size=1))
    assert calls == []


def refresh_artifacts(root):
    """Re-publish the generated fixture after intentionally changing its inputs."""
    paths = list((root / 'sources').glob('*/SOURCE_READY.json')) + [root / 'release/RELEASE.json']
    for path in paths:
        document = json.loads(path.read_text())
        for artifact in document['artifacts']:
            target = root / artifact['path']
            artifact.update(bytes=target.stat().st_size, sha256=release.prep.sha256(target))
        path.write_text(json.dumps(document))


def rewrite_index(root, name, change):
    path = root / 'release' / f'{name}.jsonl'
    rows = list(release._rows(path))
    for row in rows:
        change(row)
    value = ''.join(json.dumps(row) + '\n' for row in rows)
    path.write_text(value)
    (root / 'sources' / rows[0]['source'] / f'{rows[0]["split"]}.jsonl').write_text(value)
    refresh_artifacts(root)


def test_parallel_materialization_and_worker_count_resume_keep_recipe_and_audio(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    _, renderer = fake_renderer(monkeypatch)
    plan = release.release_plan(root, source, tmp_path / 'out', chunk_size=1)
    first = release.execute_release(plan, workers=3, materialize_workers=2)
    request = (tmp_path / 'out/request.json').read_bytes()
    manifests = {split: Path(item['path']).read_bytes() for split, item in first['manifests'].items()}
    audio = list((tmp_path / 'out/materialized').glob('*/audio/*'))
    mtimes = {path: path.stat().st_mtime_ns for path in audio}
    assert renderer.worker_counts == [3] * 6
    assert release.execute_release(plan, workers=1, materialize_workers=3) == first
    assert (tmp_path / 'out/request.json').read_bytes() == request
    assert {path: path.stat().st_mtime_ns for path in audio} == mtimes
    assert {split: Path(item['path']).read_bytes() for split, item in first['manifests'].items()} == manifests
    settings = json.loads((tmp_path / 'out/execution.json').read_text())
    assert settings['render_workers'] == 1 and settings['materialize_workers'] == 3
    assert 'workers' not in json.loads(request)


def test_pre_parallel_default_global_request_resumes_without_identity_migration(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    _, renderer = fake_renderer(monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output, chunk_size=1)
    # A request from before parallelism has no execution counts or explicit
    # default wind scope. Reconstruct those historical fields independently.
    legacy = deepcopy(plan)
    for group in legacy['groups'].values():
        group.pop('wind_id_scope', None)
    assert plan == legacy
    output.mkdir()
    original = release._json(legacy).encode()
    (output / 'request.json').write_bytes(original)
    result = release.execute_release(plan, workers=2, materialize_workers=2)
    assert result['training_pairs_ready'] is True
    assert (output / 'request.json').read_bytes() == original
    assert renderer.worker_counts == [2] * 6
    wind_groups = [g for g in plan['groups'].values() if g['role'] == 'wind']
    assert all('wind_id_scope' not in group for group in wind_groups)


def test_parallel_materialization_failure_stops_workers_without_rendering(tmp_path, monkeypatch):
    import multiprocessing
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    rewrite_index(root, 'speech_train', lambda row: row.update(sha256='0' * 64))
    plan = release.release_plan(root, source, tmp_path / 'out', chunk_size=1)
    existing = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(ValueError, match='checksum mismatch'):
        release.execute_release(plan, materialize_workers=2)
    assert calls == []
    assert not (tmp_path / 'out/READY.json').exists()
    assert not (tmp_path / 'out/manifests/train.jsonl').exists()
    assert {process.pid for process in multiprocessing.active_children()} <= existing


@pytest.mark.parametrize('field,value', [('workers', 0), ('workers', True), ('materialize_workers', -1),
                                         ('materialize_workers', 1.5)])
def test_invalid_worker_count_does_not_change_ready_release(tmp_path, monkeypatch, field, value):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    plan = release.release_plan(root, source, tmp_path / 'out')
    release.execute_release(plan)
    ready = (tmp_path / 'out/READY.json').read_bytes()
    with pytest.raises(ValueError, match='positive integer'):
        release.execute_release(plan, **{field: value})
    assert (tmp_path / 'out/READY.json').read_bytes() == ready


@pytest.mark.parametrize('duplicate_audio', [False, True])
def test_documented_generated_wind_ids_are_split_scoped_but_hashes_are_audited(tmp_path, monkeypatch, duplicate_audio):
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    for split in ('train', 'validation'):
        rewrite_index(root, f'wind_{split}', lambda row: row.update(
            source_id='wind_noise_8000hz_0',
            license='Generated using MIT-licensed SC-Wind-Noise-Generator'))
    if duplicate_audio:
        train = next(release._rows(root / 'release/wind_train.jsonl'))
        validation = next(release._rows(root / 'release/wind_validation.jsonl'))
        with tarfile.open(root / train['shard']) as archive:
            raw = archive.extractfile(train['path']).read()
        with tarfile.open(root / validation['shard'], 'w') as archive:
            member = tarfile.TarInfo(validation['path'])
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        rewrite_index(root, 'wind_validation', lambda row: row.update(
            sha256=train['sha256'], frames=train['frames']))
    plan = release.release_plan(root, source, tmp_path / 'out')
    assert {g['wind_id_scope'] for g in plan['groups'].values() if g['role'] == 'wind'} == {'split'}
    if duplicate_audio:
        with pytest.raises(ValueError, match='Early split leakage: noise_materialized_sha256'):
            release.execute_release(plan)
        assert calls == []
    else:
        assert release.execute_release(plan)['training_pairs_ready'] is True


def backup_receipts(directory):
    return [json.loads(path.read_text()) for path in (directory / 'commits').glob('*.json')]


def test_persistent_backup_restores_fixed_path_and_resumes_across_worker_changes(tmp_path, monkeypatch):
    from realtime_use.preparation_backup import PreparationBackup, restore
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    output, backup_dir = tmp_path / 'out', tmp_path / 'persistent'
    plan = release.release_plan(root, source, output, chunk_size=1)
    checkpoint = PreparationBackup.checkpoint
    def check_ready_publication(self, unit, paths, identity_paths):
        assert not (output / 'READY.json').exists()
        if unit == 'final':
            assert (output / 'READY.pending.json').is_file()
        return checkpoint(self, unit, paths, identity_paths)
    monkeypatch.setattr(PreparationBackup, 'checkpoint', check_ready_publication)
    result = release.execute_release(plan, backup_dir=backup_dir, materialize_workers=2)
    request = (output / 'request.json').read_bytes()
    expected = {path.relative_to(output): release.prep.sha256(path)
                for path in output.rglob('*') if path.is_file()
                and (path.suffix == '.wav' or path.name in {'manifest.jsonl', 'READY.json', 'request.json'})}
    receipts = backup_receipts(backup_dir)
    assert {record['unit'] for record in receipts} == {
        'root', 'final', *(f'materialized/{key}' for key in plan['groups']),
        *(f'chunks/{split}/{index:06d}' for split in ('validation', 'train') for index in range(3))}
    archives = {path.name: path.stat().st_mtime_ns for path in (backup_dir / 'objects').iterdir()}
    # Materialization reports naturally change extracted/reused counts on resume.
    # Renderer timing/provenance changes likewise must not duplicate the backup.
    for path in (output / 'chunks').glob('*/*'):
        (path / 'provenance.json').write_text('{"elapsed_seconds": 999, "workers": 3}\n')
    assert release.execute_release(plan, backup_dir=backup_dir, workers=3) == result
    assert (output / 'request.json').read_bytes() == request
    assert {path.name: path.stat().st_mtime_ns for path in (backup_dir / 'objects').iterdir()} == archives
    assert len(backup_receipts(backup_dir)) == len(receipts)
    assert json.loads((output / 'execution.json').read_text())['backup_dir'] == str(backup_dir)
    assert 'backup_dir' not in json.loads(request)

    shutil.rmtree(output)
    restore(backup_dir, output)
    assert {relative: release.prep.sha256(output / relative) for relative in expected} == expected
    # Source archives stay outside the preparation backup and are still needed.
    assert (root / 'release/RELEASE.json').is_file()
    assert release.execute_release(plan, backup_dir=backup_dir, workers=2) == result
    assert len(backup_receipts(backup_dir)) == len(receipts)
    # Selecting another persistent destination does not change the recipe.
    alternate = tmp_path / 'other-persistent'
    assert release.execute_release(plan, backup_dir=alternate) == result
    assert (output / 'request.json').read_bytes() == request
    assert len(backup_receipts(alternate)) == len(receipts)


@pytest.mark.parametrize('failure_unit', ['root', 'materialized/', 'chunks/validation/000001', 'final'])
def test_backup_failure_stops_preparation_without_ready(tmp_path, monkeypatch, failure_unit):
    import multiprocessing
    from realtime_use.preparation_backup import PreparationBackup, restore
    root, source = fixture_release(tmp_path, monkeypatch)
    calls, _ = fake_renderer(monkeypatch)
    output, backup_dir = tmp_path / 'out', tmp_path / 'persistent'
    plan = release.release_plan(root, source, output, chunk_size=1)
    checkpoint = PreparationBackup.checkpoint
    existing = {process.pid for process in multiprocessing.active_children()}
    def fail_checkpoint(self, unit, paths, identity_paths):
        if unit == failure_unit or (failure_unit == 'materialized/' and unit.startswith(failure_unit)):
            assert not (output / 'READY.json').exists()
            raise OSError('persistent backup unavailable')
        return checkpoint(self, unit, paths, identity_paths)
    monkeypatch.setattr(PreparationBackup, 'checkpoint', fail_checkpoint)
    with pytest.raises(OSError, match='persistent backup unavailable'):
        release.execute_release(plan, backup_dir=backup_dir, materialize_workers=2)
    assert not (output / 'READY.json').exists()
    assert 'final' not in {record['unit'] for record in backup_receipts(backup_dir)}
    assert {process.pid for process in multiprocessing.active_children()} <= existing
    if failure_unit in ('root', 'materialized/'):
        assert calls == []
    if failure_unit == 'materialized/':
        assert len(list((output / 'materialized').iterdir())) <= 2
    if failure_unit == 'chunks/validation/000001':
        shutil.rmtree(output)
        restore(backup_dir, output)
        assert not (output / 'READY.json').exists()
        assert (output / 'chunks/validation/000000/manifest.jsonl').is_file()
        assert not (output / 'chunks/validation/000001').exists()
        monkeypatch.setattr(PreparationBackup, 'checkpoint', checkpoint)
        assert release.execute_release(plan, backup_dir=backup_dir)['training_pairs_ready'] is True


def test_overlapping_backup_directory_rejected_without_invalidating_ready(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    fake_renderer(monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output)
    release.execute_release(plan)
    ready = (output / 'READY.json').read_bytes()
    for backup_dir in (output, output / 'backup', tmp_path):
        with pytest.raises(ValueError, match='must not overlap'):
            release.execute_release(plan, backup_dir=backup_dir)
        assert (output / 'READY.json').read_bytes() == ready
