import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.evaluate import get_confusion_matrix


@torch.no_grad()
def miou_vs_threshold(
    pred_probs: torch.Tensor,
    target_labels: torch.Tensor,
    thresholds: torch.Tensor | None = None,
    ignore_label: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Sweep decision thresholds and compute the symmetric mIoU at each one.

    Internally sorts the predicted probabilities once and uses cumulative TP/FP counts so the
    full sweep is O(N log N), independent of the number of thresholds.

    Args:
        pred_probs (torch.Tensor) : (N,) per-point predicted probabilities for the non-ground class (1).
        target_labels (torch.Tensor) : (N,) ground-truth labels in {0,1,ignore_label}.
        thresholds (torch.Tensor or None) : 1D tensor of thresholds to evaluate. Defaults to
            ``linspace(0,1,101)`` on the same device as ``pred_probs``.
        ignore_label (int) : Label value to mask out before computing IoU.

    Returns:
        thresholds (torch.Tensor) : (T,) thresholds evaluated.
        miou_per_threshold (torch.Tensor) : (T,) symmetric mIoU at each threshold.
        best_threshold (float) : Threshold maximizing mIoU.
        best_miou (float) : Maximum mIoU value.
    """
    device = pred_probs.device
    if thresholds is None:
        thresholds = torch.linspace(0.0, 1.0, 101, device=device)
    else:
        thresholds = torch.as_tensor(thresholds, device=device)

    # Mask out ignore points.
    bool_valid = target_labels != ignore_label
    probs = pred_probs[bool_valid].float()
    labels = target_labels[bool_valid].long()

    bool_pos = labels == 1
    num_pos = bool_pos.sum()
    num_neg = (~bool_pos).sum()

    if probs.numel() == 0:
        zero_per_th = torch.zeros_like(thresholds)
        return thresholds, zero_per_th, 0.0, 0.0

    # Sort probs descending and compute prefix counts.
    order = torch.argsort(probs, descending=True)
    probs_sorted = probs[order]
    bool_pos_sorted = bool_pos[order]

    tp_prefix = torch.cumsum(bool_pos_sorted, dim=0)
    fp_prefix = torch.cumsum(~bool_pos_sorted, dim=0)

    # For each threshold, count points with probability > threshold (k = position after sort).
    k = torch.searchsorted(-probs_sorted, -thresholds, right=False)

    k_clamped = torch.clamp(k, min=1) - 1
    tp = torch.where(k > 0, tp_prefix[k_clamped], torch.zeros_like(k, dtype=tp_prefix.dtype))
    fp = torch.where(k > 0, fp_prefix[k_clamped], torch.zeros_like(k, dtype=fp_prefix.dtype))
    fn = num_pos - tp

    # IoU for the positive class (non-ground).
    iou1 = tp.float() / (tp + fp + fn).float().clamp_min(1e-9)

    # IoU for the negative class (ground): TP0 = TN, FN0 = FP, FP0 = FN.
    tp0 = (num_neg - fp).float()
    fn0 = fp.float()
    fp0 = fn.float()
    iou0 = tp0 / (tp0 + fp0 + fn0).clamp_min(1e-9)

    miou = 0.5 * (iou0 + iou1)
    best_idx = miou.argmax()
    return (
        thresholds.detach(),
        miou.detach(),
        float(thresholds[best_idx].item()),
        float(miou.max().item()),
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    decision_thres: float = 0.5,
    ignore_label: int = 2,
) -> dict:
    """
    Evaluate a model over one full pass of a validation dataloader.

    Args:
        model (torch.nn.Module) : Model to evaluate (set to eval mode internally).
        val_loader (torch.utils.data.DataLoader) : Validation dataloader producing
            PTv3-style batches as built by :func:`terraseg_dataloading.collate_scans`.
        device (torch.device) : Device on which to run the model.
        decision_thres (float) : Decision threshold applied to the sigmoid of the logits.
        ignore_label (int) : Label value to mask out before metric computation.

    Returns:
        metrics (dict) : Per-class recall, precision, IoU, and the symmetric mIoU.
    """
    model.eval()

    total_tp0 = torch.zeros((), dtype=torch.float64, device=device)
    total_fn0 = torch.zeros((), dtype=torch.float64, device=device)
    total_fp0 = torch.zeros((), dtype=torch.float64, device=device)
    total_tp1 = torch.zeros((), dtype=torch.float64, device=device)
    total_fn1 = torch.zeros((), dtype=torch.float64, device=device)
    total_fp1 = torch.zeros((), dtype=torch.float64, device=device)

    for batch in val_loader:
        for key in ("coord", "feat", "offset", "target", "segment"):
            batch[key] = batch[key].to(device, non_blocking=True)

        bool_valid = batch["segment"] != ignore_label
        gt_labels = batch["segment"][bool_valid].long()
        pred_logits = model(batch)[bool_valid]
        pred_labels = (torch.sigmoid(pred_logits) > decision_thres).long()

        tp0, fn0, fp0, _ = get_confusion_matrix(
            gt_labels=gt_labels, pred_labels=pred_labels, selected_id=0, ignore_id=ignore_label
        )
        tp1, fn1, fp1, _ = get_confusion_matrix(
            gt_labels=gt_labels, pred_labels=pred_labels, selected_id=1, ignore_id=ignore_label
        )

        total_tp0 += tp0.to(torch.float64)
        total_fn0 += fn0.to(torch.float64)
        total_fp0 += fp0.to(torch.float64)
        total_tp1 += tp1.to(torch.float64)
        total_fn1 += fn1.to(torch.float64)
        total_fp1 += fp1.to(torch.float64)

    eps = 1e-9
    recall0 = (total_tp0 / (total_tp0 + total_fn0 + eps)).item()
    precision0 = (total_tp0 / (total_tp0 + total_fp0 + eps)).item()
    iou0 = (total_tp0 / (total_tp0 + total_fp0 + total_fn0 + eps)).item()
    recall1 = (total_tp1 / (total_tp1 + total_fn1 + eps)).item()
    precision1 = (total_tp1 / (total_tp1 + total_fp1 + eps)).item()
    iou1 = (total_tp1 / (total_tp1 + total_fp1 + total_fn1 + eps)).item()
    miou = 0.5 * (iou0 + iou1)

    return {
        "recall0": recall0,
        "precision0": precision0,
        "iou0": iou0,
        "recall1": recall1,
        "precision1": precision1,
        "iou1": iou1,
        "miou": miou,
    }
