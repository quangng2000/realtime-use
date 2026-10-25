"""Equal-length validation shards without counting duplicated padding rows.

Every rank runs the same number of forward calls, including when there are fewer
examples than ranks. ``distributed_valid`` marks only the original dataset rows
for metric aggregation; padding is execution work, not additional evidence.
"""

from __future__ import annotations

from typing import Any, Iterator

from torch.utils.data import Dataset, Sampler


class PaddedValidationSampler(Sampler[tuple[int, bool]]):
    """Sequential rank shards, padded with explicitly invalid repeated indices."""

    def __init__(self, dataset: Dataset, *, num_replicas: int, rank: int) -> None:
        if isinstance(num_replicas, bool) or not isinstance(num_replicas, int) or num_replicas < 1:
            raise ValueError("num_replicas must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError("rank must be an integer in [0, num_replicas)")
        self.dataset_size = len(dataset)
        if not self.dataset_size:
            raise ValueError("Validation dataset must contain at least one example")
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = (self.dataset_size + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[tuple[int, bool]]:
        for position in range(self.rank, self.total_size, self.num_replicas):
            yield position % self.dataset_size, position < self.dataset_size

    def __len__(self) -> int:
        return self.num_samples


class ValidationDataset(Dataset):
    """Attach a validity flag without mutating the underlying audio example."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | tuple[int, bool]) -> dict[str, Any]:
        sample_index, valid = index if isinstance(index, tuple) else (index, True)
        return {**self.dataset[sample_index], "distributed_valid": valid}


class RateBucketBatchSampler(Sampler[list]):
    """Native-rate batches with the same rate and batch size on every DDP rank.

    Every rate forms global batches before rank sharding. Incomplete global
    batches repeat only enough rows to divide by world size; validation marks
    those repeats invalid. Training repeats are deliberate sampler padding,
    matching DistributedSampler's equal-work contract. No rate is dropped.
    """

    def __init__(self, sample_rates, batch_size: int, *, num_replicas: int = 1,
                 rank: int = 0, shuffle: bool = True, seed: int = 0) -> None:
        self.rates = tuple(sample_rates)
        if not self.rates or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                                 for value in self.rates):
            raise ValueError("sample_rates must be nonempty positive integer metadata")
        for name, value in (("batch_size", batch_size), ("num_replicas", num_replicas)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError("rank must be an integer in [0, num_replicas)")
        self.batch_size, self.num_replicas, self.rank = batch_size, num_replicas, rank
        self.shuffle, self.seed, self.epoch = shuffle, int(seed), 0
        self.buckets = {rate: [index for index, value in enumerate(self.rates) if value == rate]
                        for rate in sorted(set(self.rates))}
        # Lightning advances batch_sampler.sampler.set_epoch when automatic
        # distributed sampler replacement is disabled. This sampler owns epochs.
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        width = self.batch_size * self.num_replicas
        return sum((len(indices) + width - 1) // width for indices in self.buckets.values())

    def __iter__(self):
        import torch

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_batches = []
        width = self.batch_size * self.num_replicas
        for bucket in self.buckets.values():
            indices = list(bucket)
            if self.shuffle:
                indices = [indices[index] for index in torch.randperm(len(indices), generator=generator).tolist()]
            for start in range(0, len(indices), width):
                original = indices[start:start + width]
                total = ((len(original) + self.num_replicas - 1) // self.num_replicas) * self.num_replicas
                batch = [(original[position % len(original)], position < len(original))
                         for position in range(total)]
                global_batches.append(batch)
        if self.shuffle:
            global_batches = [global_batches[index] for index in
                              torch.randperm(len(global_batches), generator=generator).tolist()]
        for batch in global_batches:
            shard = batch[self.rank::self.num_replicas]
            yield [index for index, _ in shard] if self.shuffle else shard
