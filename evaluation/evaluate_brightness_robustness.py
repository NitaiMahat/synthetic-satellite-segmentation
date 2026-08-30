"""Compare frozen segmentation checkpoints under controlled brightness shifts."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LoveDADataset
from src.metrics import mean_iou, per_class_iou, pixel_accuracy, confusion_matrix
from src.model import UNet
from src.synthetic import adjust_brightness


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]
CONDITIONS = {
    "Clean (1.0)": 1.0,
    "Darker (0.8)": 0.8,
    "Much darker (0.6)": 0.6,
    "Brighter (1.2)": 1.2,
    "Much brighter (1.4)": 1.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "baseline_v1_complete" / "best_checkpoint.pt",
    )
    parser.add_argument(
        "--brightness-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "synthetic_ratio_10_brightness" / "best_checkpoint.pt",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "brightness_robustness",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, base_channels: int, device: torch.device) -> UNet:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(num_classes=7, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def write_markdown(results: dict[str, dict[str, dict[str, float | list[float]]]], path: Path) -> None:
    baseline = results["real_only_baseline"]
    brightness = results["brightness_10_percent"]
    lines = [
        "# Brightness Robustness Evaluation",
        "",
        "Both frozen models were evaluated on the same unchanged Rural validation masks. "
        "Only image brightness was modified for the non-clean conditions.",
        "",
        "| Condition | Real-only mIoU | +10% Brightness mIoU | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        baseline_miou = baseline[condition]["miou"]
        brightness_miou = brightness[condition]["miou"]
        change = brightness_miou - baseline_miou
        lines.append(
            f"| {condition} | {baseline_miou:.2%} | {brightness_miou:.2%} | {change:+.2%} |"
        )

    lines += ["", "## Per-class IoU", ""]
    for condition in CONDITIONS:
        lines += [
            f"### {condition}",
            "",
            "| Class | Real-only | +10% Brightness | Change |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, baseline_iou, brightness_iou in zip(
            CLASS_NAMES,
            baseline[condition]["per_class_iou"],
            brightness[condition]["per_class_iou"],
        ):
            lines.append(
                f"| {name} | {baseline_iou:.2%} | {brightness_iou:.2%} | "
                f"{brightness_iou - baseline_iou:+.2%} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(results: dict[str, dict[str, dict[str, float | list[float]]]], path: Path) -> None:
    conditions_by_factor = sorted(CONDITIONS, key=CONDITIONS.get)
    factors = [CONDITIONS[name] for name in conditions_by_factor]
    baseline_scores = [results["real_only_baseline"][name]["miou"] for name in conditions_by_factor]
    brightness_scores = [
        results["brightness_10_percent"][name]["miou"] for name in conditions_by_factor
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(factors, baseline_scores, marker="o", linewidth=2, label="Real-only Baseline v1")
    plt.plot(factors, brightness_scores, marker="o", linewidth=2, label="+10% Brightness")
    plt.xlabel("Brightness factor")
    plt.ylabel("Validation mIoU")
    plt.title("Brightness Robustness: Frozen Model Comparison")
    plt.xticks(factors)
    plt.ylim(bottom=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("batch-size and image-size must be positive.")

    device = torch.device(args.device)
    dataset = LoveDADataset(
        PROJECT_ROOT / "data" / "raw" / "Val" / "Rural" / "images_png",
        PROJECT_ROOT / "data" / "raw" / "Val" / "Rural" / "masks_png",
        image_size=(args.image_size, args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    models = {
        "real_only_baseline": load_model(args.baseline_checkpoint, args.base_channels, device),
        "brightness_10_percent": load_model(args.brightness_checkpoint, args.base_channels, device),
    }
    matrices = {
        model_name: {
            condition: torch.zeros((7, 7), dtype=torch.int64, device=device)
            for condition in CONDITIONS
        }
        for model_name in models
    }

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            for condition, factor in CONDITIONS.items():
                adjusted_images = adjust_brightness(images, factor)
                for model_name, model in models.items():
                    predictions = model(adjusted_images).argmax(dim=1)
                    matrices[model_name][condition] += confusion_matrix(predictions, masks)

    results = {}
    for model_name, model_matrices in matrices.items():
        results[model_name] = {}
        for condition, matrix in model_matrices.items():
            results[model_name][condition] = {
                "brightness_factor": CONDITIONS[condition],
                "miou": mean_iou(matrix).item(),
                "pixel_accuracy": pixel_accuracy(matrix).item(),
                "per_class_iou": per_class_iou(matrix).cpu().tolist(),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, args.output_dir / "comparison.md")
    write_chart(results, args.output_dir / "miou_chart.png")
    print(f"Saved robustness results to: {args.output_dir}")


if __name__ == "__main__":
    main()
