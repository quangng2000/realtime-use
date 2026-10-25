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
