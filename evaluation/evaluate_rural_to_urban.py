"""Evaluate frozen Rural-trained LoveDA models on Rural and unseen Urban validation data."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import LoveDADataset
from src.metrics import confusion_matrix, mean_iou, per_class_iou, pixel_accuracy
from src.model import UNet


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]
EXPERIMENTS = {
    "real_only_baseline": {
        42: PROJECT_ROOT / "models" / "baseline_v1_complete" / "best_checkpoint.pt",
        7: PROJECT_ROOT / "models" / "baseline_v1_seed7" / "best_checkpoint.pt",
        123: PROJECT_ROOT / "models" / "baseline_v1_seed123" / "best_checkpoint.pt",
    },
    "brightness_10_percent": {
        42: PROJECT_ROOT / "models" / "synthetic_ratio_10_brightness" / "best_checkpoint.pt",
        7: PROJECT_ROOT / "models" / "synthetic_ratio_10_brightness_seed7" / "best_checkpoint.pt",
        123: PROJECT_ROOT / "models" / "synthetic_ratio_10_brightness_seed123" / "best_checkpoint.pt",
    },
    "barren_focused_sampling": {
        42: PROJECT_ROOT / "models" / "class_aware_barren_seed42" / "best_checkpoint.pt",
        7: PROJECT_ROOT / "models" / "class_aware_barren_seed7" / "best_checkpoint.pt",
        123: PROJECT_ROOT / "models" / "class_aware_barren_seed123" / "best_checkpoint.pt",
    },
}
DISPLAY_NAMES = {
    "real_only_baseline": "Real-only baseline",
    "brightness_10_percent": "+10% brightness",
    "barren_focused_sampling": "Barren-focused sampling",
}
COLORS = ["black", "lightgray", "red", "gray", "blue", "brown", "green", "yellow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualization-count", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed checkpoint results from results.json in output-dir.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "rural_to_urban",
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


def evaluate_model(
    model: UNet, loader: DataLoader, device: torch.device, label: str
) -> dict[str, float | list[float]]:
    matrix = torch.zeros((7, 7), dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(loader, start=1):
            predictions = model(images.to(device)).argmax(dim=1)
            matrix += confusion_matrix(predictions, masks.to(device))
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(f"  {label}: batch {batch_index}/{len(loader)}")
    return {
        "miou": mean_iou(matrix).item(),
        "pixel_accuracy": pixel_accuracy(matrix).item(),
        "per_class_iou": per_class_iou(matrix).cpu().tolist(),
    }


def summarize_runs(runs: dict[str, dict[str, float | list[float]]]) -> dict[str, float | list[float]]:
    """Return across-seed mean and population standard deviation for one split."""
    miou = np.array([run["miou"] for run in runs.values()], dtype=float)
    accuracy = np.array([run["pixel_accuracy"] for run in runs.values()], dtype=float)
    class_iou = np.array([run["per_class_iou"] for run in runs.values()], dtype=float)
    return {
        "runs": len(runs),
        "miou_mean": float(np.nanmean(miou)),
        "miou_std": float(np.nanstd(miou)),
        "pixel_accuracy_mean": float(np.nanmean(accuracy)),
        "pixel_accuracy_std": float(np.nanstd(accuracy)),
        "per_class_iou_mean": np.nanmean(class_iou, axis=0).tolist(),
    }


def write_markdown(results: dict[str, object], path: Path) -> None:
    lines = [
        "# Rural-to-Urban Cross-Domain Evaluation",
        "",
        "All models were trained only on `Train/Rural`. `Val/Urban` was never used for training or checkpoint selection.",
        "The Rural result is in-domain performance; the Urban result measures transfer to an unseen environment.",
        "",
        "## Average Metrics Across Three Seeds",
        "",
        "| Experiment | Rural mIoU | Urban mIoU | Rural-to-Urban change | Urban pixel accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    experiments = results["experiments"]
    for experiment_name, experiment in experiments.items():
        rural = experiment["summary"]["Rural"]
        urban = experiment["summary"]["Urban"]
        change = urban["miou_mean"] - rural["miou_mean"]
        lines.append(
            f"| {DISPLAY_NAMES[experiment_name]} | {rural['miou_mean']:.2%} +/- {rural['miou_std']:.2%} | "
            f"{urban['miou_mean']:.2%} +/- {urban['miou_std']:.2%} | {change:+.2%} | "
            f"{urban['pixel_accuracy_mean']:.2%} |"
        )

    baseline_urban = experiments["real_only_baseline"]["summary"]["Urban"]
    lines += [
        "",
        "## Urban Comparison With Real-only Baseline",
        "",
        "| Experiment | Urban mIoU change | Urban pixel accuracy change |",
        "| --- | ---: | ---: |",
    ]
    for experiment_name, experiment in experiments.items():
        urban = experiment["summary"]["Urban"]
        lines.append(
            f"| {DISPLAY_NAMES[experiment_name]} | "
            f"{urban['miou_mean'] - baseline_urban['miou_mean']:+.2%} | "
            f"{urban['pixel_accuracy_mean'] - baseline_urban['pixel_accuracy_mean']:+.2%} |"
        )

    lines += ["", "## Mean Urban Per-class IoU", "", "| Class | Baseline | +10% Brightness | Barren-focused |", "| --- | ---: | ---: | ---: |"]
    urban_scores = [
        experiments[name]["summary"]["Urban"]["per_class_iou_mean"] for name in EXPERIMENTS
    ]
    for class_name, *scores in zip(CLASS_NAMES, *urban_scores):
        lines.append(f"| {class_name} | " + " | ".join(f"{score:.2%}" for score in scores) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chart(results: dict[str, object], path: Path) -> None:
    experiments = results["experiments"]
    labels = [DISPLAY_NAMES[name] for name in EXPERIMENTS]
    rural_scores = [experiments[name]["summary"]["Rural"]["miou_mean"] for name in EXPERIMENTS]
    urban_scores = [experiments[name]["summary"]["Urban"]["miou_mean"] for name in EXPERIMENTS]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, rural_scores, width, label="Val/Rural (seen environment)", color="#2E6F95")
    plt.bar(x + width / 2, urban_scores, width, label="Val/Urban (unseen environment)", color="#D97732")
    plt.xticks(x, labels)
    plt.ylabel("Mean mIoU across three seeds")
    plt.ylim(bottom=0)
    plt.title("Rural-to-Urban Generalization")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_visualizations(
    dataset: LoveDADataset,
    baseline: UNet,
    barren_focused: UNet,
    count: int,
    device: torch.device,
    path: Path,
) -> None:
    count = min(count, len(dataset))
    figure, axes = plt.subplots(count, 4, figsize=(14, 3.5 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    cmap = plt.matplotlib.colors.ListedColormap(COLORS)

    with torch.no_grad():
        for index in range(count):
            image, target = dataset[index]
            image_batch = image.unsqueeze(0).to(device)
            baseline_prediction = baseline(image_batch).argmax(dim=1).squeeze(0).cpu()
            barren_prediction = barren_focused(image_batch).argmax(dim=1).squeeze(0).cpu()
            # Convert model IDs back to LoveDA's display IDs: 0 is No Data, 1-7 are classes.
            target_display = target.clone() + 1
            target_display[target == LoveDADataset.ignore_index] = 0
            axes[index, 0].imshow(image.permute(1, 2, 0))
            axes[index, 1].imshow(target_display, cmap=cmap, vmin=0, vmax=7)
            axes[index, 2].imshow(baseline_prediction + 1, cmap=cmap, vmin=0, vmax=7)
            axes[index, 3].imshow(barren_prediction + 1, cmap=cmap, vmin=0, vmax=7)
            for axis in axes[index]:
                axis.axis("off")

    for axis, title in zip(axes[0], ["Urban image", "Ground truth", "Baseline prediction", "Barren-focused prediction"]):
        axis.set_title(title)
    figure.suptitle("Unseen Urban Validation Examples (seed 42)", y=0.995)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.image_size <= 0 or args.visualization_count <= 0:
        raise ValueError("batch-size, image-size, and visualization-count must be positive.")

    device = torch.device(args.device)
    datasets = {
        split: LoveDADataset(
            PROJECT_ROOT / "data" / "raw" / "Val" / split / "images_png",
            PROJECT_ROOT / "data" / "raw" / "Val" / split / "masks_png",
            image_size=(args.image_size, args.image_size),
        )
        for split in ("Rural", "Urban")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for split, dataset in datasets.items()
    }
    results_path = args.output_dir / "results.json"
    if args.resume and results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"Resuming completed runs from: {results_path}")
    else:
        results: dict[str, object] = {
        "metadata": {
            "training_split": "Train/Rural only",
            "evaluation_splits": {split: len(dataset) for split, dataset in datasets.items()},
            "image_size": [args.image_size, args.image_size],
            "base_channels": args.base_channels,
            "class_names": CLASS_NAMES,
            "note": "Urban masks are used only to calculate evaluation metrics.",
        },
        "experiments": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for experiment_name, checkpoints in EXPERIMENTS.items():
        experiment_results = results["experiments"].setdefault(
            experiment_name, {"checkpoints": {}, "runs": {}, "summary": {}}
        )
        for seed, checkpoint_path in checkpoints.items():
            seed_key = str(seed)
            completed_splits = experiment_results["runs"].get(seed_key, {})
            if all(split in completed_splits for split in datasets):
                print(f"Skipping completed {experiment_name}, seed {seed}.")
                continue
            print(f"Evaluating {experiment_name}, seed {seed}...")
            model = load_model(checkpoint_path, args.base_channels, device)
            experiment_results["checkpoints"][seed_key] = str(checkpoint_path.relative_to(PROJECT_ROOT))
            run_results = experiment_results["runs"].setdefault(seed_key, {})
            for split, loader in loaders.items():
                if split not in run_results:
                    run_results[split] = evaluate_model(
                        model, loader, device, f"{experiment_name} seed {seed} {split}"
                    )
            # Preserve completed work even if a later model is interrupted.
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        experiment_results["summary"] = {
            split: summarize_runs({seed: run[split] for seed, run in experiment_results["runs"].items()})
            for split in datasets
        }

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, args.output_dir / "comparison.md")
    write_chart(results, args.output_dir / "rural_urban_miou_chart.png")
    write_visualizations(
        datasets["Urban"],
        load_model(EXPERIMENTS["real_only_baseline"][42], args.base_channels, device),
        load_model(EXPERIMENTS["barren_focused_sampling"][42], args.base_channels, device),
        args.visualization_count,
        device,
        args.output_dir / "urban_validation_examples_seed42.png",
    )
    print(f"Saved Rural-to-Urban results to: {args.output_dir}")


if __name__ == "__main__":
    main()
