import unittest

import torch
from torch.utils.data import Dataset

from src.synthetic import BrightnessMixDataset, RotationMixDataset, rotate_90


class DummyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.image = torch.full((3, 2, 2), 0.5)
        self.mask = torch.tensor([[0, 1], [2, 255]])

    def __len__(self) -> int:
        return 90

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image.clone(), self.mask.clone()


class BrightnessMixDatasetTests(unittest.TestCase):
    def test_final_ratio_and_masks_are_preserved(self) -> None:
        dataset = BrightnessMixDataset(DummyDataset(), synthetic_ratio=0.1, seed=42)

        self.assertEqual(dataset.real_count, 90)
        self.assertEqual(dataset.synthetic_count, 10)
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.effective_synthetic_ratio, 0.1)

        image, mask = dataset[90]
        self.assertTrue(
            torch.isclose(image[0, 0, 0], torch.tensor(0.4))
            or torch.isclose(image[0, 0, 0], torch.tensor(0.6))
        )
        torch.testing.assert_close(mask, torch.tensor([[0, 1], [2, 255]]))


class RotationMixDatasetTests(unittest.TestCase):
    def test_rotation_keeps_image_and_mask_aligned(self) -> None:
        image = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
        mask = torch.tensor([[10, 20], [30, 40]], dtype=torch.int64)

        rotated_image, rotated_mask = rotate_90(image, mask, quarter_turns=1)

        torch.testing.assert_close(rotated_image, torch.tensor([[[2, 4], [1, 3]]], dtype=torch.float32))
        torch.testing.assert_close(rotated_mask, torch.tensor([[20, 40], [10, 30]], dtype=torch.int64))

    def test_final_ratio_and_rotation_metadata(self) -> None:
        dataset = RotationMixDataset(DummyDataset(), synthetic_ratio=0.1, seed=42)

        self.assertEqual(dataset.real_count, 90)
        self.assertEqual(dataset.synthetic_count, 10)
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.effective_synthetic_ratio, 0.1)
        self.assertEqual(dataset.metadata()[0]["transformation"], "rotation_90")
        self.assertIn(dataset.metadata()[0]["degrees_counterclockwise"], (90, 180, 270))


if __name__ == "__main__":
    unittest.main()
