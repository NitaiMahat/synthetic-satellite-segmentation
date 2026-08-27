"""Create original, target, prediction, and error-map baseline visualizations."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LoveDADataset
from src.model import UNet


CLASS_COLORS = ["lightgray", "red", "gray", "blue", "brown", "green", "yellow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "baseline_v1" / "best_checkpoint.pt",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "baseline_v1_validation_examples.png",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model = UNet(num_classes=7, base_channels=args.base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = LoveDADataset(
        PROJECT_ROOT / "data/raw/Val/Rural/images_png",
        PROJECT_ROOT / "data/raw/Val/Rural/masks_png",
        image_size=(args.image_size, args.image_size),
    )
    indices = np.linspace(0, len(dataset) - 1, args.num_examples, dtype=int)
    class_cmap = ListedColormap(CLASS_COLORS)
    class_cmap.set_bad("black")
    error_cmap = ListedColormap(["white", "red"])

    figure, axes = plt.subplots(args.num_examples, 4, figsize=(16, 4 * args.num_examples))
    for row, index in enumerate(indices):
        image, target = dataset[index]
        with torch.no_grad():
            prediction = model(image.unsqueeze(0)).argmax(dim=1).squeeze(0)

        image_array = image.permute(1, 2, 0).numpy()
        target_array = target.numpy()
        prediction_array = prediction.numpy()
        target_for_display = np.ma.masked_equal(target_array, LoveDADataset.ignore_index)
        errors = (prediction_array != target_array) & (
            target_array != LoveDADataset.ignore_index
        )

        axes[row, 0].imshow(image_array)
        axes[row, 0].set_title(f"Validation Image {index}")
        axes[row, 1].imshow(target_for_display, cmap=class_cmap, vmin=0, vmax=6)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(prediction_array, cmap=class_cmap, vmin=0, vmax=6)
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(errors, cmap=error_cmap, vmin=0, vmax=1)
        axes[row, 3].set_title("Error Map")

        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=150)
    print(f"Saved validation visualizations to: {args.output_path}")


if __name__ == "__main__":
    main()
