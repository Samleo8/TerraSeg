import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(
    gt_sorted: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the gradient of the Lovász extension with respect to sorted errors.

    Args:
        gt_sorted (torch.Tensor) : (P,) float ground-truth labels in {0,1} sorted by descending error.

    Returns:
        jaccard (torch.Tensor) : (P,) gradient values.
    """
    p = gt_sorted.numel()
    if p == 0:
        return gt_sorted

    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)

    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_binary_from_errors(
    errors: torch.Tensor,
    fg: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Lovász loss for a single class (one-vs-rest formulation).

    Args:
        errors (torch.Tensor) : (P,) absolute errors |f(x_i) - y_i|.
        fg (torch.Tensor) : (P,) float ground-truth labels in {0,1}.

    Returns:
        loss (torch.Tensor) : Lovász loss value.
    """
    if fg.numel() == 0:
        return errors.sum() * 0.0

    errors_sorted, perm = torch.sort(errors, descending=True)
    fg_sorted = fg[perm]
    grad = _lovasz_grad(fg_sorted)
    return torch.dot(errors_sorted, grad)


def lovasz_softmax_binary_symmetric(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric Lovász-Softmax loss for binary segmentation, matching mIoU = (IoU0+IoU1) / 2.

    Args:
        logits (torch.Tensor) : (N,) per-point logits for the positive (non-ground) class.
        target_labels (torch.Tensor) : (N,) ground-truth labels in {0,1}.

    Returns:
        loss (torch.Tensor) : Symmetric Lovász loss value.
    """
    if target_labels.numel() == 0:
        return logits.sum() * 0.0

    p1 = torch.sigmoid(logits)

    # Class 0 (ground).
    fg0 = (target_labels == 0).float()
    p0 = 1.0 - p1
    err0 = (fg0 - p0).abs()
    loss0 = _lovasz_binary_from_errors(err0, fg0)
    
    # Class 1 (non-ground).
    fg1 = (target_labels == 1).float()
    err1 = (fg1 - p1).abs()
    loss1 = _lovasz_binary_from_errors(err1, fg1)

    return (loss0 + loss1) / 2.0


class BCELovaszPerScan(nn.Module):
    """
    Per-scan combined Binary Cross-Entropy + symmetric Lovász loss for ground segmentation.

    The loss is computed per scan (using the cumulative offset tensor) and averaged across
    the scans in the batch that contain at least one non-ignore point. Matches the loss
    described in the TerraSeg paper, Eq. (4): L = L_BCE + lambda * L_Lovasz.

    The BCE positive-class weight is exposed as a buffer ``pos_weight`` so the training loop
    can update it (e.g. with an EMA of the ground/non-ground ratio) by calling
    :meth:`set_pos_weight` after each optimizer step.

    Args:
        lovasz_weight (float) : Coefficient lambda in front of the Lovász term. Default: 1.0.
        ignore_label (int) : Label value to ignore in loss computation. Default: 2.
        pos_weight_init (float) : Initial value for the BCE positive-class weight. Default: 1.0.
    """

    def __init__(
        self,
        lovasz_weight: float = 1.0,
        ignore_label: int = 2,
        pos_weight_init: float = 1.0,
    ):
        super().__init__()
        assert lovasz_weight >= 0.0, "lovasz_weight must be non-negative!"
        self.lovasz_weight = float(lovasz_weight)
        self.ignore_label = int(ignore_label)

        self.register_buffer(
            "pos_weight", torch.tensor([float(pos_weight_init)], dtype=torch.float32)
        )

    def set_pos_weight(
        self,
        value: float,
    ) -> None:
        """
        Update the BCE positive-class weight in place.

        Args:
            value (float) : New positive-class weight.
        """
        self.pos_weight.fill_(float(value))

    @staticmethod
    def _offset_to_ranges(
        offset: torch.Tensor,
    ) -> list[tuple[int, int]]:
        """
        Convert cumulative offsets into a list of ``(start, end)`` ranges.

        Args:
            offset (torch.Tensor) : (B,) cumulative point counts per scan.

        Returns:
            ranges (list[tuple[int,int]]) : List of (start, end) index ranges.
        """
        ends = offset.detach().cpu().tolist()
        ranges = []
        start = 0
        for end in ends:
            ranges.append((start, int(end)))
            start = int(end)
        return ranges

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        offset: torch.Tensor,
    ) -> dict:
        """
        Compute the combined BCE + Lovász loss over a batch of scans.

        Args:
            pred_logits (torch.Tensor) : (N_tot,) per-point logits for the non-ground class.
            target_labels (torch.Tensor) : (N_tot,) ground-truth labels in {0,1,ignore_label}.
            offset (torch.Tensor) : (B,) cumulative point counts per scan.

        Returns:
            dict :
                - "loss" : Total batch loss (BCE + lovasz_weight * Lovász), averaged over used scans.
                - "bce" : Detached BCE component (mean over used scans).
                - "lovasz" : Detached Lovász component (mean over used scans).
                - "num_scans_used" : Number of scans contributing to the loss.
        """
        assert pred_logits.ndim == 1, "pred_logits must be a 1D tensor!"
        assert target_labels.ndim == 1, "target_labels must be a 1D tensor!"
        assert pred_logits.shape[0] == target_labels.shape[0], (
            "pred_logits and target_labels must have the same length!"
        )

        ranges = self._offset_to_ranges(offset)

        bce_sum = pred_logits.sum() * 0.0
        lovasz_sum = pred_logits.sum() * 0.0
        num_scans_used = 0

        for start, end in ranges:
            scan_logits = pred_logits[start:end]
            scan_labels = target_labels[start:end]

            bool_valid = scan_labels != self.ignore_label
            if not bool_valid.any():
                continue

            scan_logits = scan_logits[bool_valid]
            scan_labels = scan_labels[bool_valid]
            scan_targets_float = (scan_labels == 1).float()

            bce_scan = F.binary_cross_entropy_with_logits(
                input=scan_logits,
                target=scan_targets_float,
                pos_weight=self.pos_weight,
                reduction="mean",
            )

            if self.lovasz_weight > 0.0:
                lovasz_scan = lovasz_softmax_binary_symmetric(
                    logits=scan_logits, target_labels=(scan_labels == 1).long()
                )
            else:
                lovasz_scan = scan_logits.sum() * 0.0

            bce_sum = bce_sum + bce_scan
            lovasz_sum = lovasz_sum + lovasz_scan
            num_scans_used += 1

        if num_scans_used == 0:
            zero = pred_logits.sum() * 0.0
            return {
                "loss": zero,
                "bce": zero.detach(),
                "lovasz": zero.detach(),
                "num_scans_used": 0,
            }

        bce_mean = bce_sum / num_scans_used
        lovasz_mean = lovasz_sum / num_scans_used
        total = bce_mean + self.lovasz_weight * lovasz_mean

        return {
            "loss": total,
            "bce": bce_mean.detach(),
            "lovasz": lovasz_mean.detach(),
            "num_scans_used": num_scans_used,
        }
