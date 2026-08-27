import unittest

import torch

from src.model import UNet


class UNetTests(unittest.TestCase):
    def test_output_matches_input_spatial_shape(self) -> None:
        model = UNet(num_classes=7, base_channels=8)
        inputs = torch.randn(1, 3, 64, 64)

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (1, 7, 64, 64))


if __name__ == "__main__":
    unittest.main()
