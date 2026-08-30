"""Label-preserving synthetic data for controlled LoveDA experiments."""

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Apply a brightness factor to a normalized image tensor without changing its mask."""
    if factor <= 0:
        raise ValueError("Brightness factors must be positive.")
    return (image * factor).clamp(0.0, 1.0)


@dataclass(frozen=True)
class SyntheticSample:
    """Metadata needed to reproduce one synthetic training sample."""

    source_index: int
    brightness_factor: float


class BrightnessMixDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Mix real samples with deterministic brightness-modified copies.

    ``synthetic_ratio`` is the proportion of synthetic entries in the final
    mixed dataset, not the number of copies added relative to real data.
    """

    def __init__(
        self,
        real_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        synthetic_ratio: float,
        brightness_factors: tuple[float, float] = (0.8, 1.2),
        seed: int = 42,
    ) -> None:
        if not 0 < synthetic_ratio < 1:
            raise ValueError("synthetic_ratio must be between 0 and 1.")
        if not brightness_factors:
            raise ValueError("At least one brightness factor is required.")

        self.real_dataset = real_dataset
        self.synthetic_ratio = synthetic_ratio
        self.brightness_factors = brightness_factors
        self.real_count = len(real_dataset)
        self.synthetic_count = round(self.real_count * synthetic_ratio / (1 - synthetic_ratio))
        if self.synthetic_count > self.real_count:
            raise ValueError("Synthetic ratio requires more source images than are available.")

        source_indices = random.Random(seed).sample(range(self.real_count), self.synthetic_count)
        self.synthetic_samples = tuple(
            SyntheticSample(
                source_index=source_index,
                brightness_factor=brightness_factors[index % len(brightness_factors)],
            )
            for index, source_index in enumerate(source_indices)
        )

    @property
    def effective_synthetic_ratio(self) -> float:
        return self.synthetic_count / len(self)

    def __len__(self) -> int:
        return self.real_count + self.synthetic_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")
        if index < self.real_count:
            return self.real_dataset[index]

        sample = self.synthetic_samples[index - self.real_count]
        image, mask = self.real_dataset[sample.source_index]
        return adjust_brightness(image, sample.brightness_factor), mask

    def metadata(self) -> list[dict[str, int | float | str]]:
        """Return serializable metadata for every synthetic sample."""
        return [
            {
                "source_index": sample.source_index,
                "transformation": "brightness",
                "brightness_factor": sample.brightness_factor,
            }
            for sample in self.synthetic_samples
        ]
