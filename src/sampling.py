"""Class-aware sampling utilities for imbalanced LoveDA training masks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def target_class_indices(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], target_class_id: int
) -> list[int]:
    """Find dataset entries containing the requested remapped class ID."""
    if target_class_id < 0:
        raise ValueError("target_class_id must be non-negative.")

    mask_paths = getattr(dataset, "mask_paths", None)
    if mask_paths is not None:
        raw_label = target_class_id + 1
        return [
            index
            for index, mask_path in enumerate(mask_paths)
            if np.any(np.asarray(Image.open(Path(mask_path))) == raw_label)
        ]

    indices = []
    for index in range(len(dataset)):
        _, mask = dataset[index]
        if torch.any(mask == target_class_id):
            indices.append(index)
    return indices


def class_aware_sampler(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    target_class_id: int,
    target_weight: float,
    num_samples: int,
    seed: int,
) -> tuple[WeightedRandomSampler, list[int]]:
    """Favor images containing one rare class while preserving epoch size."""
    if target_weight <= 1:
        raise ValueError("target_weight must be greater than 1.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    indices = target_class_indices(dataset, target_class_id)
    if not indices:
        raise ValueError(f"No samples contain target class ID {target_class_id}.")

    weights = torch.ones(len(dataset), dtype=torch.double)
    weights[indices] = target_weight
    sampler = WeightedRandomSampler(
        weights,
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler, indices
