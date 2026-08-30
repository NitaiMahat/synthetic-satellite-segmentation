"""Summarize a completed baseline run using its saved metrics."""

import argparse
import json
from pathlib import Path


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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = json.loads((args.run_dir / "history.json").read_text(encoding="utf-8"))
    best = json.loads((args.run_dir / "best_metrics.json").read_text(encoding="utf-8"))
    ranked_classes = sorted(
        zip(CLASS_NAMES, best["validation_per_class_iou"]), key=lambda item: item[1], reverse=True
    )
    strongest = ", ".join(name for name, _ in ranked_classes[:3])
    weakest = ", ".join(name for name, _ in ranked_classes[-3:])
    first_loss = history[0]["train_loss"]
    last_loss = history[-1]["train_loss"]

    note = f"""# Baseline v1 Complete Run Note

- Best validation mIoU: {best['validation_miou']:.4f} at epoch {best['epoch']}
- Pixel accuracy at the best epoch: {best['validation_pixel_accuracy']:.4f}
- Strongest classes by IoU: {strongest}
- Weakest classes by IoU: {weakest}
- Training loss changed from {first_loss:.4f} in epoch 1 to {last_loss:.4f} in epoch {history[-1]['epoch']}.

This run uses only real Rural LoveDA imagery. It is the reference point for later experiments with synthetic augmentation. The validation examples should be reviewed for visible failure patterns before interpreting the scores.
"""
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(note, encoding="utf-8")
    print(f"Saved baseline note to: {args.output_path}")


if __name__ == "__main__":
    main()
