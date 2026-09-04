"""Compare three class-aware Barren oversampling runs against real-only baselines."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]
RUNS = [
    (
        42,
        PROJECT_ROOT / "models" / "baseline_v1_complete" / "best_metrics.json",
        PROJECT_ROOT / "models" / "class_aware_barren_seed42" / "best_metrics.json",
    ),
    (
        7,
        PROJECT_ROOT / "models" / "baseline_v1_seed7" / "best_metrics.json",
        PROJECT_ROOT / "models" / "class_aware_barren_seed7" / "best_metrics.json",
    ),
    (
        123,
        PROJECT_ROOT / "models" / "baseline_v1_seed123" / "best_metrics.json",
        PROJECT_ROOT / "models" / "class_aware_barren_seed123" / "best_metrics.json",
    ),
]


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(metrics: list[dict], key: str) -> float:
    return float(np.mean([item[key] for item in metrics]))


def mean_class_iou(metrics: list[dict]) -> list[float]:
    return np.mean([item["validation_per_class_iou"] for item in metrics], axis=0).tolist()


def write_markdown(results: dict, output_path: Path) -> None:
    averages = results["averages"]
    lines = [
        "# Class-Aware Barren Oversampling Comparison",
        "",
        "Each seed used the same Rural split, U-Net, 15 epochs, image size, and 1,366 "
        "training samples per epoch. The class-aware method sampled Barren-containing "
        "images with a weight of 4.0.",
        "",
        "| Seed | Baseline mIoU | Class-aware mIoU | Change | Baseline Barren IoU | Class-aware Barren IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results["per_seed"]:
        lines.append(
            f"| {item['seed']} | {item['baseline']['miou']:.2%} | "
            f"{item['class_aware']['miou']:.2%} | {item['miou_change']:+.2%} | "
            f"{item['baseline']['barren_iou']:.2%} | {item['class_aware']['barren_iou']:.2%} |"
        )

    lines += [
        "",
        "## Three-Seed Average",
        "",
        f"- Baseline mIoU: {averages['baseline']['miou']:.2%}",
        f"- Class-aware mIoU: {averages['class_aware']['miou']:.2%}",
        f"- mIoU change: {averages['miou_change']:+.2%}",
        f"- Baseline Barren IoU: {averages['baseline']['barren_iou']:.2%}",
        f"- Class-aware Barren IoU: {averages['class_aware']['barren_iou']:.2%}",
        f"- Barren IoU change: {averages['barren_iou_change']:+.2%}",
        f"- Pixel accuracy change: {averages['pixel_accuracy_change']:+.2%}",
        "",
        "## Interpretation",
        "",
        "Class-aware Barren oversampling produced nonzero Barren IoU in all three seeds "
        "while keeping average mIoU effectively unchanged. Pixel accuracy decreased, "
        "so this is a rare-class recall tradeoff rather than an across-the-board improvement.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(results: dict, output_path: Path) -> None:
    labels = [str(item["seed"]) for item in results["per_seed"]] + ["Average"]
    baseline_miou = [item["baseline"]["miou"] for item in results["per_seed"]]
    class_aware_miou = [item["class_aware"]["miou"] for item in results["per_seed"]]
    baseline_barren = [item["baseline"]["barren_iou"] for item in results["per_seed"]]
    class_aware_barren = [item["class_aware"]["barren_iou"] for item in results["per_seed"]]
    baseline_miou.append(results["averages"]["baseline"]["miou"])
    class_aware_miou.append(results["averages"]["class_aware"]["miou"])
    baseline_barren.append(results["averages"]["baseline"]["barren_iou"])
    class_aware_barren.append(results["averages"]["class_aware"]["barren_iou"])

    positions = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, baseline, class_aware, title in [
        (axes[0], baseline_miou, class_aware_miou, "Validation mIoU"),
        (axes[1], baseline_barren, class_aware_barren, "Barren IoU"),
    ]:
        axis.bar(positions - width / 2, baseline, width, label="Real-only baseline", color="#173A5E")
        axis.bar(positions + width / 2, class_aware, width, label="Class-aware Barren", color="#177E89")
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    figure.suptitle("Three-Seed Class-Aware Barren Oversampling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    per_seed = []
    baseline_metrics = []
    class_aware_metrics = []
    for seed, baseline_path, class_aware_path in RUNS:
        baseline = load_metrics(baseline_path)
        class_aware = load_metrics(class_aware_path)
        baseline_metrics.append(baseline)
        class_aware_metrics.append(class_aware)
        per_seed.append(
            {
                "seed": seed,
                "baseline": {
                    "best_epoch": baseline["epoch"],
                    "miou": baseline["validation_miou"],
                    "pixel_accuracy": baseline["validation_pixel_accuracy"],
                    "barren_iou": baseline["validation_per_class_iou"][4],
                },
                "class_aware": {
                    "best_epoch": class_aware["epoch"],
                    "miou": class_aware["validation_miou"],
                    "pixel_accuracy": class_aware["validation_pixel_accuracy"],
                    "barren_iou": class_aware["validation_per_class_iou"][4],
                },
                "miou_change": class_aware["validation_miou"] - baseline["validation_miou"],
            }
        )

    baseline_average = {
        "miou": mean_metric(baseline_metrics, "validation_miou"),
        "pixel_accuracy": mean_metric(baseline_metrics, "validation_pixel_accuracy"),
        "barren_iou": mean_class_iou(baseline_metrics)[4],
        "per_class_iou": mean_class_iou(baseline_metrics),
    }
    class_aware_average = {
        "miou": mean_metric(class_aware_metrics, "validation_miou"),
        "pixel_accuracy": mean_metric(class_aware_metrics, "validation_pixel_accuracy"),
        "barren_iou": mean_class_iou(class_aware_metrics)[4],
        "per_class_iou": mean_class_iou(class_aware_metrics),
    }
    results = {
        "experiment": "class_aware_barren_oversampling",
        "per_seed": per_seed,
        "averages": {
            "baseline": baseline_average,
            "class_aware": class_aware_average,
            "miou_change": class_aware_average["miou"] - baseline_average["miou"],
            "barren_iou_change": class_aware_average["barren_iou"] - baseline_average["barren_iou"],
            "pixel_accuracy_change": class_aware_average["pixel_accuracy"] - baseline_average["pixel_accuracy"],
        },
    }
    output_dir = PROJECT_ROOT / "evaluation" / "class_aware_barren_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, output_dir / "comparison.md")
    write_chart(results, output_dir / "metrics_chart.png")
    print(f"Saved class-aware comparison to: {output_dir}")


if __name__ == "__main__":
    main()
