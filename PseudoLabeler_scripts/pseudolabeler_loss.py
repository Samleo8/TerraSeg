import torch
from torch.nn.functional import huber_loss, mse_loss


def pseudolabeler_loss(
    pred_heights: torch.Tensor,
    target_heights: torch.Tensor,
    delta: float = 0.05,
) -> torch.Tensor:
    """
    Asymmetric ground-height loss.

    1. For nonnegative residuals r = (target - pred) >= 0 (prediction below 3D point),
    use Huber(r, delta), which is quadratic near 0 and linear beyond delta.
    2. For negative residuals r < 0 (prediction above 3D point),
    use MSE, which is quadratic.

    This loss function balances two physical regimes of a point cloud equally,
    regardless of the number of points in each regime. The goal is to obtain a
    height map that mediates between the two regimes, instead of a plane that
    achieves the lowest error per point.

    Args:
        pred_heights (torch.Tensor) : Predicted heights (N,). Unit: meters.
        target_heights (torch.Tensor) : Ground-truth heights (N,). Unit: meters.
        delta (float) : Threshold for Huber loss. Unit: meters.

    Returns:
        loss (torch.Tensor) : Mean of Huber and MSE loss terms. Unit: square meters (m^2).
    """
    assert pred_heights.shape == target_heights.shape, "Input tensors must have the same shape!"
    assert delta > 0, "Delta must be positive!"

    # Compute element-wise losses (no reduction yet).
    huber_elements = huber_loss(
        input=pred_heights,
        target=target_heights,
        delta=delta,
        reduction="none",
    )
    mse_elements = mse_loss(
        input=pred_heights,
        target=target_heights,
        reduction="none",
    )

    # Create boolean masks.
    res = target_heights - pred_heights
    bool_pos = res >= 0  # Real point ABOVE predicted height map.
    bool_neg = res < 0  # Real point BELOW predicted height map.

    # Safely count elements to avoid division by zero.
    pos_count = bool_pos.sum().clamp(min=1)
    neg_count = bool_neg.sum().clamp(min=1)

    # Compute independent means.
    huber_term = (huber_elements * bool_pos).sum() / pos_count
    mse_term = (mse_elements * bool_neg).sum() / neg_count

    # Combine.
    loss = (huber_term + mse_term) / 2
    return loss
