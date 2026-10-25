"""Source-pool bridge tests operate on local generated audio, never training."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import soundfile as sf

from realtime_use.materialize_sources import make_plan, materialize
from realtime_use.prepare_data import read_scp, audit_splits


def pool(tmp_path, *, source='vctk', split='train', count=2, role='speech'):
    root = tmp_path / 'dataset'
    root.mkdir(exist_ok=True)
    shard = root / f'{source}-{split}.tar'
    rows = []
    with tarfile.open(shard, 'w') as archive:
        for i in range(count):
            wav = io.BytesIO()
            sf.write(wav, np.linspace(-.1, .1, 80 + i), 8000, format='WAV', subtype='FLOAT')
            data = wav.getvalue()
            source_id = f'wind_noise_{i}' if role == 'wind' else f'utterance_{i}'
            name = f'audio/{source}/{split}/{source_id}.wav'
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            rows.append(dict(id=f'{source}/{split}/{source_id}', source_id=source_id,
                             source=source, split=split, kind='noise' if role == 'wind' else role,
                             sample_rate=8000, frames=80 + i, channels=1,
                             sha256=hashlib.sha256(data).hexdigest(), shard=shard.name, path=name,
                             speaker_id=f'{source}:speaker{i}' if role == 'speech' else None,
                             text='Reference text' if i == 0 and role == 'speech' else None,
                             license='fixture-only', commercial_clearance='pending'))
    manifest = root / f'{source}-{split}.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return root, manifest, rows


def rewrite(manifest, rows):
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_plan_is_read_only_and_materialization_yields_urgent_inventories(tmp_path):
    root, manifest, original = pool(tmp_path)
    out = tmp_path / 'materialized'
    plan = make_plan(root, manifest, out, 'speech')
    assert not out.exists()
    result = materialize(plan)
    assert result['extracted'] == 2 and result['reused'] == 0
    inventory = read_scp(out / 'source.scp')
    assert set(inventory) == {'vctk:utterance_0', 'vctk:utterance_1'}
    assert (out / 'utt2spk').read_text() == 'vctk:utterance_0 vctk:speaker0\nvctk:utterance_1 vctk:speaker1\n'
    assert (out / 'text').read_text() == 'vctk:utterance_0 Reference text\n'
    records = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    assert all(r['commercial_clearance'] == 'pending' for r in records)
    assert result['commercial_clearance'] == 'not_attested'
    assert result['training_pairs_ready'] is False
    assert records[0]['id'] == original[0]['id']
    assert 'commercial_use' not in result
    assert 'rir_direct_path_samples' not in result


def test_identical_resume_reuses_verified_files_without_rewriting(tmp_path):
    root, manifest, _ = pool(tmp_path)
    plan = make_plan(root, manifest, tmp_path / 'out', 'speech')
    materialize(plan)
    times = {p: p.stat().st_mtime_ns for p in (tmp_path / 'out/audio').iterdir()}
    result = materialize(plan)
    assert result['reused'] == 2 and result['extracted'] == 0
    assert times == {p: p.stat().st_mtime_ns for p in times}


def test_generated_wind_split_scope_preserves_raw_ids_and_waveform_hashes(tmp_path):
    plans = []
    for split in ('train', 'validation'):
        root, manifest, original = pool(tmp_path, source='wind', split=split, role='wind', count=1)
        plan = make_plan(root, manifest, tmp_path / split, 'wind', wind_id_scope='split')
        materialize(plan)
        record = json.loads((tmp_path / split / 'records.jsonl').read_text())
        assert record['source_id'] == original[0]['source_id']
        assert record['sha256'] == original[0]['sha256']
        assert record['uid'].startswith('wind_noise:')
        plans.append(plan)
    assert plans[0]['records'][0]['uid'] != plans[1]['records'][0]['uid']
    # Fixture waveforms are identical: namespacing must not hide this evidence.
    assert plans[0]['records'][0]['sha256'] == plans[1]['records'][0]['sha256']
    with pytest.raises(ValueError, match='different'):
        materialize(make_plan(root, manifest, tmp_path / 'validation', 'wind'))


def test_split_local_wind_policy_cannot_hide_recorded_speech_ids(tmp_path):
    root, manifest, _ = pool(tmp_path)
    with pytest.raises(ValueError, match='independently generated wind'):
        make_plan(root, manifest, tmp_path / 'out', 'speech', wind_id_scope='split')


def test_changed_request_and_corrupt_resumed_audio_are_rejected(tmp_path):
    root, manifest, _ = pool(tmp_path)
    plan = make_plan(root, manifest, tmp_path / 'out', 'speech')
    materialize(plan)
    changed = make_plan(root, manifest, tmp_path / 'out', 'speech', max_items=1)
    with pytest.raises(ValueError, match='different'):
        materialize(changed)
    next((tmp_path / 'out/audio').iterdir()).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        materialize(plan)


def test_checksum_failure_does_not_publish_inventory_or_keep_unverified_file(tmp_path):
    root, manifest, rows = pool(tmp_path, count=1)
    rows[0]['sha256'] = '0' * 64
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='checksum'):
        materialize(make_plan(root, manifest, tmp_path / 'out', 'speech'))
    assert not (tmp_path / 'out/source.scp').exists()
    assert list((tmp_path / 'out/audio').iterdir()) == []


@pytest.mark.parametrize('field,value', [('path', '../escape.wav'), ('path', '/absolute.wav'),
                                        ('shard', '../outside.tar'), ('path', 'folder\\escape.wav')])
def test_archive_traversal_in_source_records_is_rejected(tmp_path, field, value):
    root, manifest, rows = pool(tmp_path)
    rows[0][field] = value
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='archive path|POSIX'):
        make_plan(root, manifest, tmp_path / 'out', 'speech')


@pytest.mark.parametrize('link', [False, True])
def test_dangerous_tar_headers_are_never_extracted(tmp_path, link):
    root, manifest, rows = pool(tmp_path, count=1)
    with tarfile.open(root / rows[0]['shard'], 'w') as archive:
        info = tarfile.TarInfo(rows[0]['path'] if link else '../escape.wav')
        if link:
            info.type, info.linkname = tarfile.SYMTYPE, '/etc/passwd'
        archive.addfile(info)
    with pytest.raises(ValueError, match='Unsafe archive|links/special'):
        materialize(make_plan(root, manifest, tmp_path / 'out', 'speech'))
    assert not (tmp_path / 'escape.wav').exists()
    assert not (tmp_path / 'out/source.scp').exists()


def test_missing_member_and_bad_audio_metadata_do_not_publish(tmp_path):
    root, manifest, rows = pool(tmp_path, count=1)
    rows[0]['path'] = 'missing.wav'
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='Missing 1'):
        materialize(make_plan(root, manifest, tmp_path / 'missing', 'speech'))
    rows[0]['path'] = 'audio/vctk/train/utterance_0.wav'
    rows[0]['sample_rate'] = 16000
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='metadata mismatch'):
        materialize(make_plan(root, manifest, tmp_path / 'bad-rate', 'speech'))


def test_subset_does_not_leak_unselected_transcripts_or_speakers(tmp_path):
    root, manifest, _ = pool(tmp_path, count=3)
    plan = make_plan(root, manifest, tmp_path / 'out', 'speech', max_items=1)
    result = materialize(plan)
    assert result['selected_count'] == 1
    assert 'speaker1' not in (tmp_path / 'out/utt2spk').read_text()
    assert 'utterance_1' not in (tmp_path / 'out/records.jsonl').read_text()
    assert len(list((tmp_path / 'out/audio').iterdir())) == 1


def test_same_source_id_across_splits_retains_identity_for_leakage_audit(tmp_path):
    paths = []
    for split in ['train', 'validation']:
        root, manifest, _ = pool(tmp_path, split=split, count=1)
        plan = make_plan(root, manifest, tmp_path / split, 'speech')
        materialize(plan)
        row = json.loads((tmp_path / split / 'records.jsonl').read_text())
        assert row['uid'] == 'vctk:utterance_0'
        audit = dict(split=split, speech_uid=row['uid'], speaker_id=row['speaker_id'],
                     source_speech_sha256=row['sha256'], noise_uid=f'noise-{split}',
                     source_noise_sha256=f'noise-hash-{split}')
        path = tmp_path / f'audit-{split}.jsonl'
        path.write_text(json.dumps(audit) + '\n')
        paths.append(path)
    with pytest.raises(ValueError, match='Split leakage'):
        audit_splits(paths)


def test_wind_ids_keep_dispatch_prefix_and_corpus_namespaces(tmp_path):
    root, manifest, _ = pool(tmp_path, source='wind', role='wind', count=1)
    plan = make_plan(root, manifest, tmp_path / 'out', 'wind')
    materialize(plan)
    assert (tmp_path / 'out/source.scp').read_text().startswith('wind_noise:wind:wind_noise_0 8000 ')
    assert not (tmp_path / 'out/utt2spk').exists()


def test_duplicate_identity_missing_speaker_and_multisplit_input_rejected(tmp_path):
    root, manifest, rows = pool(tmp_path)
    rows[1]['source_id'] = rows[0]['source_id']
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='Duplicate'):
        make_plan(root, manifest, tmp_path / 'out', 'speech')
    rows[1]['source_id'] = 'utterance_1'
    rows[1]['speaker_id'] = None
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='speaker_id'):
        make_plan(root, manifest, tmp_path / 'out', 'speech')
    rows[1]['speaker_id'] = 'speaker1'
    rows[1]['split'] = 'validation'
    rewrite(manifest, rows)
    with pytest.raises(ValueError, match='one source and split'):
        make_plan(root, manifest, tmp_path / 'out', 'speech')


def test_output_audio_symlink_cannot_escape(tmp_path):
    root, manifest, _ = pool(tmp_path)
    out = tmp_path / 'out'
    plan = make_plan(root, manifest, out, 'speech')
    out.mkdir()
    (out / 'request.json').write_text(json.dumps({k: v for k, v in plan.items() if k != 'records'}))
    outside = tmp_path / 'outside'
    outside.mkdir()
    (out / 'audio').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        materialize(plan)
    assert list(outside.iterdir()) == []


def stereo_pool(tmp_path, *, nonfinite_second_channel=False):
    root, manifest, rows = pool(tmp_path, count=1)
    waveform = np.stack([np.linspace(-.3, .3, 80), np.full(80, .8)], axis=1).astype(np.float32)
    if nonfinite_second_channel:
        waveform[10, 1] = np.nan
    wav = io.BytesIO()
    sf.write(wav, waveform, 8000, format='WAV', subtype='FLOAT')
    content = wav.getvalue()
    with tarfile.open(root / rows[0]['shard'], 'w') as archive:
        info = tarfile.TarInfo(rows[0]['path'])
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    rows[0]['channels'] = 2
    rows[0]['sha256'] = hashlib.sha256(content).hexdigest()
    rewrite(manifest, rows)
    return root, manifest, rows, waveform


def test_first_channel_policy_preserves_original_hash_and_exact_channel_zero(tmp_path):
    root, manifest, rows, waveform = stereo_pool(tmp_path)
    with pytest.raises(ValueError, match='explicit channel_policy'):
        make_plan(root, manifest, tmp_path / 'reject', 'speech')
    plan = make_plan(root, manifest, tmp_path / 'out', 'speech', channel_policy='first')
    result = materialize(plan)
    record = json.loads((tmp_path / 'out/records.jsonl').read_text())
    path = Path(record['materialized_path'])
    samples, rate = sf.read(path, dtype='float32')
    np.testing.assert_array_equal(samples, waveform[:, 0])
    assert rate == 8000 and sf.info(path).subtype == 'FLOAT'
    assert record['sha256'] == record['source_sha256'] == rows[0]['sha256']
    assert record['channels'] == record['source_channels'] == 2
    assert record['materialized_channels'] == 1
    assert record['materialized_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record['materialized_sha256'] != record['source_sha256']
    assert record['channel_transform'] == 'first_channel_float32_wav'
    assert result['channel_policy'] == 'first'
    assert next(iter(read_scp(tmp_path / 'out/source.scp').values()))[1] == path
    before = path.stat().st_mtime_ns
    resumed = materialize(make_plan(root, manifest, tmp_path / 'out', 'speech', channel_policy='first'))
    assert resumed['reused'] == 1 and resumed['extracted'] == 0
    assert path.stat().st_mtime_ns == before


def test_first_channel_policy_checks_all_source_channels_for_nonfinite_samples(tmp_path):
    root, manifest, _, _ = stereo_pool(tmp_path, nonfinite_second_channel=True)
    with pytest.raises(ValueError, match='Nonfinite audio'):
        materialize(make_plan(root, manifest, tmp_path / 'out', 'speech', channel_policy='first'))
    assert not (tmp_path / 'out/source.scp').exists()
    assert list((tmp_path / 'out/audio').iterdir()) == []


def test_first_channel_resume_requires_matching_original_source_verification(tmp_path):
    root, manifest, _, _ = stereo_pool(tmp_path)
    plan = make_plan(root, manifest, tmp_path / 'out', 'speech', channel_policy='first')
    materialize(plan)
    sidecar = next((tmp_path / 'out/verification').iterdir())
    metadata = json.loads(sidecar.read_text())
    metadata['source_sha256'] = '0' * 64
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='verification identity'):
        materialize(make_plan(root, manifest, tmp_path / 'out', 'speech', channel_policy='first'))


def test_channel_policy_is_part_of_resume_identity_even_for_mono_sources(tmp_path):
    root, manifest, _ = pool(tmp_path)
    materialize(make_plan(root, manifest, tmp_path / 'out', 'speech', channel_policy='first'))
    with pytest.raises(ValueError, match='different'):
        materialize(make_plan(root, manifest, tmp_path / 'out', 'speech'))


def test_only_rirs_may_use_unsplit_pool(tmp_path):
    root, manifest, _ = pool(tmp_path, role='rir', source='rirs', split='pool', count=1)
    result = materialize(make_plan(root, manifest, tmp_path / 'rirs', 'rir'))
    assert result['split'] == 'pool'
    root, manifest, _ = pool(tmp_path, role='speech', split='pool', count=1)
    with pytest.raises(ValueError, match='only RIRs may use pool'):
        make_plan(root, manifest, tmp_path / 'speech', 'speech')


def test_mls_speaker_namespaces_include_language_without_overwriting_raw_speaker(tmp_path):
    root, manifest, rows = pool(tmp_path, source='mls', count=2)
    for row, language in zip(rows, ['german', 'french']):
        row['source_id'] = f'mls_{language}_utterance1'
        row['speaker_id'] = '123'
    rewrite(manifest, rows)
    materialize(make_plan(root, manifest, tmp_path / 'out', 'speech'))
    assert (tmp_path / 'out/utt2spk').read_text().splitlines() == [
        'mls:mls_german_utterance1 mls:german:123',
        'mls:mls_french_utterance1 mls:french:123',
    ]
    records = [json.loads(line) for line in (tmp_path / 'out/records.jsonl').read_text().splitlines()]
    assert {r['speaker_id'] for r in records} == {'123'}
    assert {r['materialized_speaker_id'] for r in records} == {'mls:german:123', 'mls:french:123'}
