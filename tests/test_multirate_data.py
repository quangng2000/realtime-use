"""Data-only tests: no Trainer, optimizer, or training loop is invoked."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from realtime_use.data import PairedAudioDataset, collate_audio
from realtime_use.datamodule import PairedAudioDataModule
from realtime_use.distributed_data import RateBucketBatchSampler


def pairs(tmp_path, rates=(8000, 16000, 8000, 22050, 16000)):
    rows = []
    for index, rate in enumerate(rates):
        name = f'{index}.wav'
        sf.write(tmp_path / name, np.linspace(-.1, .1, rate // 10), rate, subtype='FLOAT')
        rows.append(dict(id=str(index), noisy=name, clean=name, sample_rate=rate))
    path = tmp_path / 'pairs.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def test_native_rate_crops_preserve_duration_and_collate_rejects_mixture(tmp_path):
    path = pairs(tmp_path)
    dataset = PairedAudioDataset(path, (8000, 16000, 22050), crop_seconds=.02, training=False)
    assert dataset.item_sample_rates == (8000, 16000, 8000, 22050, 16000)
    for index in range(len(dataset)):
        item = dataset[index]
        assert item['noisy'].numel() == round(.02 * item['sample_rate'])
        torch.testing.assert_close(item['clean'], item['noisy'])
    with pytest.raises(ValueError, match='one native sample_rate'):
        collate_audio([dataset[0], dataset[1]])
    assert collate_audio([dataset[0], dataset[2]])['sample_rate'] == 8000


@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('world_size,batch_size', [(1, 2), (2, 2), (4, 3)])
def test_rate_buckets_synchronize_rates_and_exact_validation_counts(shuffle, world_size, batch_size):
    rates = (8000, 16000, 8000, 22050, 16000, 16000, 8000)
    samplers = [RateBucketBatchSampler(rates, batch_size, num_replicas=world_size,
                                     rank=rank, shuffle=shuffle, seed=7) for rank in range(world_size)]
    batches = [list(sampler) for sampler in samplers]
    assert len({len(rank_batches) for rank_batches in batches}) == 1
    covered = []
    for step in range(len(batches[0])):
        rank_sizes, step_rates = [], set()
        for rank in range(world_size):
            batch = batches[rank][step]
            rank_sizes.append(len(batch))
            for entry in batch:
                index, valid = (entry, True) if shuffle else entry
                step_rates.add(rates[index])
                if valid:
                    covered.append(index)
        assert len(set(rank_sizes)) == 1
        assert len(step_rates) == 1
    assert set(covered) == set(range(len(rates)))
    if not shuffle:
        assert sorted(covered) == list(range(len(rates)))


def test_rate_buckets_epoch_contract_is_reproducible():
    sampler = RateBucketBatchSampler((8000,) * 20 + (16000,) * 20, 2, shuffle=True, seed=8)
    first = list(sampler)
    # This is the method Lightning discovers through batch_sampler.sampler.
    sampler.sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_multirate_datamodule_full_validation_and_per_rank_batches(tmp_path):
    manifest = pairs(tmp_path)
    config = dict(model=dict(sample_rate=16000, sample_rates=[8000, 16000, 22050]),
                  training=dict(crop_seconds=.02, batch_size=2),
                  data=dict(train_manifest=str(manifest), validation_manifest=str(manifest),
                            validation_full_utterance=True))
    modules = [PairedAudioDataModule(config) for _ in range(2)]
    for rank, module in enumerate(modules):
        module.setup('fit')
        module.trainer = SimpleNamespace(world_size=2, global_rank=rank)
        assert isinstance(module.train_dataloader().batch_sampler, RateBucketBatchSampler)
        for batch in module.train_dataloader():
            assert batch['noisy'].shape[-1] == round(batch['sample_rate'] * .02)
        for batch in module.val_dataloader():
            assert all(length == batch['sample_rate'] // 10 for length in batch['lengths'])
    counts = sum(int(batch['distributed_valid'].sum()) for module in modules for batch in module.val_dataloader())
    assert counts == 5


def test_transcript_and_language_metadata_survive_without_tensor_conversion(tmp_path):
    path = pairs(tmp_path, rates=(16000,))
    row = json.loads(path.read_text())
    row.update(transcript='A reference transcript.', language='en')
    path.write_text(json.dumps(row) + '\n')
    item = PairedAudioDataset(path, 16000, training=False)[0]
    assert item['transcript'] == row['transcript']
    assert item['language'] == 'en'
    batch = collate_audio([item])
    assert batch['transcript'] == ['A reference transcript.']
    assert batch['language'] == ['en']
