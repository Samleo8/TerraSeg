from typing import Dict, Tuple

import torch


def get_confusion_matrix(
    gt_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    selected_id: int = 0,
    ignore_id: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get confusion matrix values.

    Args:
        gt_labels (torch.Tensor):  Ground truth point labels (0: ground, 1: non-ground, 2: ignore). Shape: (N,).
        pred_labels (torch.Tensor) : Predicted point labels (0: ground, 1: non-ground). Shape: (N,).
        selected_id (int) : Selected class ID.
        ignore_id (int) : Ignore class ID.

    Returns:
        tp, fn, fp, tn (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) : Number of TPs, FNs, FPs, and TNs (0-dimensional tensors).
    """
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"Shape mismatch: gt_labels {gt_labels.shape} vs pred_labels {pred_labels.shape}"
        )

    # Create a mask for valid pixels.
    valid_mask = gt_labels != ignore_id

    # Create raw binary masks for ground truth and predictions.
    gt = gt_labels == selected_id
    pr = pred_labels == selected_id

    # Calculate metrics using bitwise operations and the valid mask.
    tp = (gt & pr & valid_mask).sum()
    fn = (gt & ~pr & valid_mask).sum()
    fp = (~gt & pr & valid_mask).sum()
    tn = (~gt & ~pr & valid_mask).sum()
    return tp, fn, fp, tn


def get_standard_metrics(
    tp: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    tn: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute standard evaluation metrics for binary point segmentation.

    Args:
        tp (torch.Tensor) : True Positives. 0-D tensor.
        fn (torch.Tensor) : False Negatives. 0-D tensor.
        fp (torch.Tensor) : False Positives. 0-D tensor.
        tn (torch.Tensor) : True Negatives. 0-D tensor.
        eps (float) : Small value to prevent division by zero.

    Returns:
        Dict[str, float] : Dictionary containing the following metrics:
            - 'accuracy' : Overall accuracy.
            - 'iou' : Intersection-over-Union for the positive class.
            - 'precision' : Precision.
            - 'recall' : Recall.
            - 'f1' : F1 score.
    """
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * tp) / ((2 * tp) + fp + fn + eps)
    return {
        "accuracy": accuracy.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }
