"""Train a real-only LoveDA baseline on the Rural split."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LoveDADataset
from src.metrics import mean_iou, per_class_iou, pixel_accuracy, confusion_matrix
from src.model import UNet
from src.synthetic import BrightnessMixDataset


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="baseline_v1_real_only_rural")
    parser.add_argument("--synthetic-ratio", type=float, default=0.0)
    parser.add_argument("--brightness-factors", type=float, nargs="+", default=[0.8, 1.2])
    parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "baseline_v1",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_images = 0
    matrix = torch.zeros((7, 7), dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            total_images += images.size(0)
            matrix += confusion_matrix(logits.argmax(dim=1), masks)

    return total_loss / total_images, matrix


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("epochs, batch-size, and image-size must be positive.")
    if not 0 <= args.synthetic_ratio < 1:
        raise ValueError("synthetic-ratio must be between 0 and 1.")

    set_seed(args.seed)
    device = torch.device(args.device)
    image_size = (args.image_size, args.image_size)
    real_train_dataset = LoveDADataset(
        PROJECT_ROOT / "data/raw/Train/Rural/images_png",
        PROJECT_ROOT / "data/raw/Train/Rural/masks_png",
        image_size=image_size,
    )
    val_dataset = LoveDADataset(
        PROJECT_ROOT / "data/raw/Val/Rural/images_png",
        PROJECT_ROOT / "data/raw/Val/Rural/masks_png",
        image_size=image_size,
    )
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    synthetic_details: dict[str, object] = {"type": "none"}
    if args.synthetic_ratio:
        train_dataset = BrightnessMixDataset(
            real_train_dataset,
            synthetic_ratio=args.synthetic_ratio,
            brightness_factors=tuple(args.brightness_factors),
            seed=args.seed,
        )
        samples_per_epoch = args.train_samples_per_epoch or len(real_train_dataset)
        if samples_per_epoch > len(train_dataset):
            raise ValueError("train-samples-per-epoch cannot exceed the mixed dataset size.")
        sampler = RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=samples_per_epoch,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
        synthetic_details = {
            "type": "brightness",
            "requested_final_ratio": args.synthetic_ratio,
            "effective_final_ratio": train_dataset.effective_synthetic_ratio,
            "real_samples": train_dataset.real_count,
            "synthetic_samples": train_dataset.synthetic_count,
            "mixed_dataset_samples": len(train_dataset),
            "brightness_factors": args.brightness_factors,
            "training_samples_per_epoch": samples_per_epoch,
        }
    else:
        train_dataset = real_train_dataset
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = UNet(num_classes=7, base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=LoveDADataset.ignore_index)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    history_path = args.output_dir / "history.json"
    config = {
        "experiment": args.experiment_name,
        "image_size": [args.image_size, args.image_size],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "seed": args.seed,
        "train_split": "Train/Rural",
        "validation_split": "Val/Rural",
        "model_architecture": "UNet(base_channels=" + str(args.base_channels) + ")",
        "num_classes": 7,
        "loss": "CrossEntropyLoss(ignore_index=255)",
        "synthetic_augmentation": synthetic_details,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.synthetic_ratio:
        (args.output_dir / "synthetic_samples.json").write_text(
            json.dumps(train_dataset.metadata(), indent=2), encoding="utf-8"
        )
    best_miou = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_loader.sampler)
        val_loss, matrix = evaluate(model, val_loader, criterion, device)
        val_miou = mean_iou(matrix).item()
        val_accuracy = pixel_accuracy(matrix).item()
        class_ious = per_class_iou(matrix).cpu().tolist()
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_miou": val_miou,
            "validation_pixel_accuracy": val_accuracy,
            "validation_per_class_iou": class_ious,
        }
        history.append(epoch_metrics)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | val mIoU: {val_miou:.4f} | "
            f"pixel accuracy: {val_accuracy:.4f}"
        )
        print(
            "Per-class IoU: "
            + ", ".join(f"{name}={score:.4f}" for name, score in zip(CLASS_NAMES, class_ious))
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_miou": val_miou,
                    "config": config,
                },
                checkpoint_path,
            )
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps(epoch_metrics, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
