"""A compact U-Net baseline for LoveDA semantic segmentation."""

import torch
from torch import nn
from torch.nn import functional as functional


class DoubleConv(nn.Module):
    """Apply two convolution, normalization, and activation blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    """Small U-Net that predicts one logit map for each semantic class."""

    def __init__(self, num_classes: int = 7, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        self.encoder1 = DoubleConv(3, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.decoder3 = DoubleConv(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.decoder2 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        decoder3 = self._up_and_merge(bottleneck, encoder3, self.decoder3)
        decoder2 = self._up_and_merge(decoder3, encoder2, self.decoder2)
        decoder1 = self._up_and_merge(decoder2, encoder1, self.decoder1)
        return self.classifier(decoder1)

    @staticmethod
    def _up_and_merge(
        decoder_features: torch.Tensor,
        skip_features: torch.Tensor,
        block: nn.Module,
    ) -> torch.Tensor:
        decoder_features = functional.interpolate(
            decoder_features,
            size=skip_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return block(torch.cat((decoder_features, skip_features), dim=1))
