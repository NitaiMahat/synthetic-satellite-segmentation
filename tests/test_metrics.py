import unittest

import torch

from src.metrics import segmentation_metrics


class SegmentationMetricsTests(unittest.TestCase):
    def test_metrics_ignore_no_data_pixels(self) -> None:
        predictions = torch.tensor([[0, 1, 1, 2], [0, 2, 1, 0]])
        targets = torch.tensor([[0, 1, 2, 2], [255, 2, 0, 0]])

        metrics = segmentation_metrics(predictions, targets, num_classes=3)

        expected_matrix = torch.tensor([[2, 1, 0], [0, 1, 0], [0, 1, 2]])
        torch.testing.assert_close(metrics["confusion_matrix"], expected_matrix)
        torch.testing.assert_close(metrics["pixel_accuracy"], torch.tensor(5 / 7))
        torch.testing.assert_close(
            metrics["per_class_iou"], torch.tensor([2 / 3, 1 / 3, 2 / 3])
        )
        torch.testing.assert_close(metrics["mean_iou"], torch.tensor(5 / 9))


if __name__ == "__main__":
    unittest.main()
