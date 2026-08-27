"""Evaluation metrics for semantic segmentation."""

import torch


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 7,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Return a ``[num_classes, num_classes]`` target-by-prediction matrix.

    ``predictions`` and ``targets`` contain class IDs with the same shape.
    Target pixels equal to ``ignore_index`` are excluded from evaluation.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape, got "
            f"{predictions.shape} and {targets.shape}."
        )

    predictions = predictions.to(torch.int64).reshape(-1)
    targets = targets.to(torch.int64).reshape(-1)
    valid_target_mask = targets != ignore_index

    invalid_targets = valid_target_mask & ((targets < 0) | (targets >= num_classes))
    if torch.any(invalid_targets):
        raise ValueError(f"Unexpected target labels: {torch.unique(targets[invalid_targets]).tolist()}")

    predictions = predictions[valid_target_mask]
    targets = targets[valid_target_mask]
    if predictions.numel() == 0:
        raise ValueError("No valid target pixels remain after ignoring labels.")

    invalid_predictions = (predictions < 0) | (predictions >= num_classes)
    if torch.any(invalid_predictions):
        raise ValueError(
            f"Unexpected prediction labels: {torch.unique(predictions[invalid_predictions]).tolist()}"
        )

    encoded_pairs = targets * num_classes + predictions
    return torch.bincount(encoded_pairs, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def pixel_accuracy(matrix: torch.Tensor) -> torch.Tensor:
    """Calculate pixel accuracy from a target-by-prediction confusion matrix."""
    total_pixels = matrix.sum()
    if total_pixels == 0:
        raise ValueError("Pixel accuracy is undefined for an empty confusion matrix.")
    return matrix.diag().sum().to(torch.float32) / total_pixels


def per_class_iou(matrix: torch.Tensor) -> torch.Tensor:
    """Calculate IoU for every class; absent classes receive ``nan``."""
    true_positives = matrix.diag().to(torch.float32)
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - matrix.diag()

    iou = torch.full_like(true_positives, torch.nan)
    present_classes = union > 0
    iou[present_classes] = true_positives[present_classes] / union[present_classes]
    return iou


def mean_iou(matrix: torch.Tensor) -> torch.Tensor:
    """Calculate mIoU across classes present in the evaluated targets or predictions."""
    return torch.nanmean(per_class_iou(matrix))


def segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 7,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    """Calculate the confusion matrix, pixel accuracy, per-class IoU, and mIoU."""
    matrix = confusion_matrix(predictions, targets, num_classes, ignore_index)
    return {
        "confusion_matrix": matrix,
        "pixel_accuracy": pixel_accuracy(matrix),
        "per_class_iou": per_class_iou(matrix),
        "mean_iou": mean_iou(matrix),
    }
