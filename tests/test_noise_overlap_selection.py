"""Exact holdout-noise selection uses generated data only; never train."""
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from realtime_use import prepare_release as release
from realtime_use.preparation_backup import PreparationBackup, restore
from test_materialized_backup_mode import paired_fixture_renderer
from test_prepare_release import backup_receipts, fake_renderer, fixture_release, refresh_artifacts


def duplicate_validation_noise(root, *, replace=False):
    train = root / 'release/noise_train.jsonl'
    rows = list(release._rows(train))
    reference = next(release._rows(root / 'release/noise_validation.jsonl'))
    with tarfile.open(root / reference['shard']) as archive:
        duplicate = archive.extractfile(reference['path']).read()
    shard = root / rows[0]['shard']
    with tarfile.open(shard) as archive:
        contents = [(item.name, archive.extractfile(item).read()) for item in archive]
    row = {**reference, 'id': 'fixture_noise/train/duplicate', 'source_id': 'duplicate',
           'split': 'train', 'shard': str(shard.relative_to(root)), 'path': 'audio/duplicate.wav'}
    rows = [row] if replace else [*rows, row]
    contents = [(row['path'], duplicate)] if replace else [*contents, (row['path'], duplicate)]
    with tarfile.open(shard, 'w') as archive:
        for name, data in contents:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    payload = ''.join(json.dumps(row) + '\n' for row in rows)
    train.write_text(payload)
    (shard.parent / 'train.jsonl').write_text(payload)
    refresh_artifacts(root)


def source_bytes(output):
    return {path: path.read_bytes() for path in (output / 'materialized').rglob('*')
            if path.is_file() and path.name != 'materialization.json'}


def failed_strict_release(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    duplicate_validation_noise(root)
    calls, renderer = fake_renderer(monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output, chunk_size=1)
    with pytest.raises(ValueError, match='Early split leakage: noise_materialized_sha256'):
        release.execute_release(plan)
    assert calls == [] and not (output / 'source-selection').exists()
    return root, output, plan, calls, renderer


def test_holdout_preserves_sources_speech_counts_and_freezes_resume_identity(tmp_path, monkeypatch):
    root, output, plan, calls, renderer = failed_strict_release(tmp_path, monkeypatch)
    raw_release = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    original, request = source_bytes(output), (output / 'request.json').read_bytes()
    result = release.execute_release(plan, noise_overlap_policy='validation-holdout')
    assert result['examples'] == {'train': 3, 'validation': 3}
    assert result['training_started'] is False
    assert source_bytes(output) == original
    assert (output / 'request.json').read_bytes() == request
    assert all(path.read_bytes() == data for path, data in raw_release.items())
    report_path = output / 'source-selection/manifest.json'
    report = json.loads(report_path.read_text())
    assert report['policy'] == 'validation-holdout'
    assert report['original_counts']['noise_train'] == 2
    assert report['effective_counts']['noise_train'] == 1
    assert report['original_counts']['noise_validation'] == report['effective_counts']['noise_validation'] == 1
    assert report['speech_pair_counts'] == result['examples']
    assert report['excluded_count'] == 1
    exclusion = report['exclusions'][0]
    assert exclusion['uid'] == 'fixture_noise:duplicate'
    assert exclusion['validation_matches'][0]['uid'] == 'fixture_noise:validation_0'
    selected = next(iter(report['selected_train_noise'].values()))
    selected_rows = list(release._rows(report_path.parent / selected['records']))
    assert [row['uid'] for row in selected_rows] == ['fixture_noise:train_0']
    binding = result['source_selection']
    assert binding['sha256'] == release.prep.sha256(report_path)
    assert result['source_audit']['source_selection'] == binding
    assert all(call['provenance']['source_selection'] == binding for call in calls)
    assert all('fixture_noise:duplicate' not in call['inventories']['noise'] for call in calls)
    frozen = {path: path.read_bytes() for path in report_path.parent.rglob('*') if path.is_file()}
    assert release.execute_release(plan, workers=2, noise_overlap_policy='validation-holdout') == result
    assert all(path.read_bytes() == data for path, data in frozen.items())
    ready = (output / 'READY.json').read_bytes()
    with pytest.raises(ValueError, match='frozen noise overlap policy'):
        release.execute_release(plan)  # Policy omission cannot silently change the recipe.
    assert (output / 'READY.json').read_bytes() == ready


def test_selection_checkpoint_final_dependencies_and_exact_restore(tmp_path, monkeypatch):
    _, output, plan, _, _ = failed_strict_release(tmp_path, monkeypatch)
    paired_fixture_renderer(monkeypatch)
    backup = tmp_path / 'backup'
    result = release.execute_release(plan, noise_overlap_policy='validation-holdout', backup_dir=backup,
                                     materialized_backup_mode='metadata')
    files = {path.relative_to(output): path.read_bytes() for parent in ('source-selection', 'chunks')
             for path in (output / parent).rglob('*') if path.is_file()}
    receipts = backup_receipts(backup)
    selection = next(item for item in receipts if item['unit'] == 'source-selection')
    final = next(item for item in receipts if item['unit'] == 'final')
    assert final['dependencies']['source-selection'] == selection['receipt_sha256']
    assert set(selection['identity_files']) == {
        'request.json', 'source-selection/manifest.json',
        'source-selection/noise__train__fixture_noise/records.jsonl',
        'source-selection/noise__train__fixture_noise/source.scp'}
    objects = set((backup / 'objects').iterdir())
    shutil.rmtree(output)
    assert restore(backup, output)['ready_restored'] is True
    assert all((output / path).read_bytes() == data for path, data in files.items())
    assert json.loads((output / 'READY.json').read_text()) == result
    assert release.execute_release(plan, noise_overlap_policy='validation-holdout', backup_dir=backup,
                                   materialized_backup_mode='metadata') == result
    assert set((backup / 'objects').iterdir()) == objects


def test_selection_backup_failure_prevents_settings_and_render(tmp_path, monkeypatch):
    _, output, plan, calls, _ = failed_strict_release(tmp_path, monkeypatch)
    checkpoint = PreparationBackup.checkpoint
    def fail(self, unit, paths, identity_paths):
        if unit == 'source-selection':
            raise OSError('selection checkpoint failed')
        return checkpoint(self, unit, paths, identity_paths)
    monkeypatch.setattr(PreparationBackup, 'checkpoint', fail)
    with pytest.raises(OSError, match='selection checkpoint failed'):
        release.execute_release(plan, noise_overlap_policy='validation-holdout', backup_dir=tmp_path / 'backup')
    assert calls == []
    assert not (output / 'settings').exists()
    assert not (output / 'READY.json').exists()


@pytest.mark.parametrize('role,field,reference_role', [
    ('speech', 'uid', 'speech'), ('speech', 'materialized_speaker_id', 'speech'),
    ('speech', 'materialized_sha256', 'speech'), ('noise', 'uid', 'noise'),
    ('wind', 'materialized_sha256', 'noise'),
])
def test_holdout_still_rejects_other_leakage_including_excluded_rows(tmp_path, monkeypatch, role, field, reference_role):
    root, source = fixture_release(tmp_path, monkeypatch)
    duplicate_validation_noise(root)
    calls, _ = fake_renderer(monkeypatch)
    materialize, train = release.bridge.materialize, {}
    def alias(plan):
        result = materialize(plan)
        path = Path(result['output_dir']) / 'records.jsonl'
        rows = list(release._rows(path))
        if plan['split'] == 'train':
            train[plan['role']] = rows
        if plan['split'] == 'validation' and plan['role'] == role:
            # Wind collision uses the otherwise-excluded train noise row: it
            # must be rejected before that exclusion can conceal the evidence.
            reference = train[reference_role][-1 if role == 'wind' else 0]
            rows[0][field] = reference[field]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return result
    monkeypatch.setattr(release.bridge, 'materialize', alias)
    with pytest.raises(ValueError, match='Early split leakage'):
        release.execute_release(release.release_plan(root, source, tmp_path / 'out'),
                                noise_overlap_policy='validation-holdout')
    assert calls == []
    assert not (tmp_path / 'out/READY.json').exists()


def test_holdout_rejects_exhausted_training_corpus(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    duplicate_validation_noise(root, replace=True)
    calls, _ = fake_renderer(monkeypatch)
    with pytest.raises(ValueError, match='exhausts training corpus'):
        release.execute_release(release.release_plan(root, source, tmp_path / 'out'),
                                noise_overlap_policy='validation-holdout')
    assert calls == []


def test_holdout_rejects_insufficient_noise_rate_after_filtering(tmp_path, monkeypatch):
    _, output, plan, _, _ = failed_strict_release(tmp_path, monkeypatch)
    materialized = {key: json.loads((output / 'materialized' / key / 'materialization.json').read_text())
                    for key in plan['groups']}
    group = next(key for key, item in plan['groups'].items() if (item['role'], item['split']) == ('speech', 'train'))
    plan['groups'][group]['sample_rates'] = [16000]
    with pytest.raises(ValueError, match='sample-rate coverage'):
        release._select_noise_holdout(plan, materialized)
    assert not (output / 'source-selection/manifest.json').exists()


@pytest.mark.parametrize('artifact', ['settings', 'chunks', 'chunk-inputs', 'manifests'])
def test_cannot_retrofit_selection_after_render_artifacts(tmp_path, monkeypatch, artifact):
    root, source = fixture_release(tmp_path, monkeypatch)
    output = tmp_path / 'out'
    plan = release.release_plan(root, source, output)
    (output / artifact).mkdir(parents=True)
    before = sorted(output.rglob('*'))
    with pytest.raises(ValueError, match='after rendering artifacts'):
        release.execute_release(plan, noise_overlap_policy='validation-holdout')
    assert sorted(output.rglob('*')) == before


def test_isolated_preflight_and_changed_selection_rejected_before_render(tmp_path, monkeypatch):
    _, output, plan, calls, _ = failed_strict_release(tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    materialized = {key: json.loads((output / 'materialized' / key / 'materialization.json').read_text())
                    for key in plan['groups']}
    effective, binding = release._select_noise_holdout(plan, materialized, selection_dir=tmp_path / 'probe')
    assert release.audit_materialized_sources(plan, effective)['records']['noise_train'] == 1
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not (output / 'source-selection').exists()
    assert Path(binding['manifest']).parent == tmp_path / 'probe'
    release.execute_release(plan, noise_overlap_policy='validation-holdout')
    calls.clear()
    manifest = output / 'source-selection/manifest.json'
    assert manifest.read_bytes() == Path(binding['manifest']).read_bytes()
    changed = json.loads(manifest.read_text())
    changed['excluded_count'] = 0
    manifest.write_text(release._json(changed))
    with pytest.raises(ValueError, match='changed existing output'):
        release.execute_release(plan, noise_overlap_policy='validation-holdout')
    assert calls == []


def test_unknown_policy_fails_before_output_creation(tmp_path, monkeypatch):
    root, source = fixture_release(tmp_path, monkeypatch)
    output = tmp_path / 'out'
    with pytest.raises(ValueError, match='noise_overlap_policy'):
        release.execute_release(release.release_plan(root, source, output), noise_overlap_policy='ignore')
    assert not output.exists()
