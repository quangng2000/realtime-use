"""Release-to-pair integration fixtures contain generated audio; never train."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
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
    def execute(plan, *, resume=False):
        assert resume is True
        output = Path(plan['output_dir'])
        calls.append(plan)
        if (output / 'manifest.jsonl').exists():
            assert json.loads((output / 'test-plan.json').read_text()) == json.loads(json.dumps(plan))
            return output / 'manifest.jsonl'
        output.mkdir(parents=True)
        (output / 'test-plan.json').write_text(json.dumps(plan))
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
        return manifest
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
    def interrupt(plan, *, resume=False):
        if plan['provenance']['split'] == 'validation' and plan['chunk']['index'] == 1:
            raise RuntimeError('simulated interruption')
        return renderer(plan, resume=resume)
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
    def leak(plan, *, resume=False):
        path = renderer(plan, resume=resume)
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
    def interrupted_repair(chunk, *, resume=False):
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
