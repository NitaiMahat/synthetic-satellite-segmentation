"""PyTorch dataset utilities for LoveDA PNG image and mask pairs."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class LoveDADataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load LoveDA images and remapped semantic-segmentation masks.

    Raw LoveDA label 0 is No Data and becomes ``ignore_index``. Raw labels
    1 through 7 become contiguous class IDs 0 through 6.
    """

    ignore_index = 255
    num_classes = 7

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        if image_size is not None and (image_size[0] <= 0 or image_size[1] <= 0):
            raise ValueError("image_size must contain positive height and width values.")

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in: {self.image_dir}")

        self.mask_paths = []
        for image_path in self.image_paths:
            mask_path = self.mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for image '{image_path.name}': {mask_path}"
                )
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])

        if self.image_size is not None:
            height, width = self.image_size
            resize_size = (width, height)
            image = image.resize(resize_size, Image.Resampling.BILINEAR)
            mask = mask.resize(resize_size, Image.Resampling.NEAREST)

        raw_mask = np.array(mask)

        image_tensor = pil_to_tensor(image).to(torch.float32).div(255.0)
        raw_mask_tensor = torch.from_numpy(raw_mask).to(torch.int64)

        invalid_label_mask = (raw_mask_tensor < 0) | (raw_mask_tensor > 7)
        if torch.any(invalid_label_mask):
            invalid_labels = torch.unique(raw_mask_tensor[invalid_label_mask])
            raise ValueError(
                f"Unexpected label IDs in '{self.mask_paths[index].name}': "
                f"{invalid_labels.tolist()}"
            )

        mask_tensor = raw_mask_tensor - 1
        mask_tensor[raw_mask_tensor == 0] = self.ignore_index

        return image_tensor, mask_tensor
