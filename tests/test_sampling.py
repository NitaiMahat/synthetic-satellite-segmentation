import unittest

import torch
from torch.utils.data import Dataset

from src.sampling import class_aware_sampler, target_class_indices


class VariedMaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.masks = [
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([[4, 1], [2, 3]]),
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([[0, 4], [2, 3]]),
        ]

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((3, 2, 2)), self.masks[index]


class ClassAwareSamplerTests(unittest.TestCase):
    def test_finds_only_images_with_target_class(self) -> None:
        dataset = VariedMaskDataset()
        self.assertEqual(target_class_indices(dataset, target_class_id=4), [1, 3])

    def test_sampler_is_deterministic_and_oversamples_target_images(self) -> None:
        dataset = VariedMaskDataset()
        sampler, indices = class_aware_sampler(
            dataset,
            target_class_id=4,
            target_weight=4.0,
            num_samples=1000,
            seed=42,
        )

        sampled_indices = list(sampler)
        target_draws = sum(index in indices for index in sampled_indices)
        self.assertEqual(len(sampled_indices), 1000)
        self.assertGreater(target_draws, 700)


if __name__ == "__main__":
    unittest.main()
