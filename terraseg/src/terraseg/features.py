import torch


def compute_terraseg_features(
    coord: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 3D per-point feature vector used by TerraSeg.

    Following TerraSeg paper section 3.4, the per-point feature vector is:
        (1) Constant ones channel.
        (2) Normalized height ``z / 5.0``.
        (3) Normalized horizontal range ``||(x, y)||_2 / 100.0``.

    Raw (x, y, z) coordinates are reserved for constructing PTv3's spatial voxel grid and are
    therefore NOT included as features.

    Args:
        coord (torch.Tensor) : (N, 3) per-point XYZ coordinates. Unit: meters.

    Returns:
        feat (torch.Tensor) : (N, 3) per-point feature vector in float32.
    """
    assert coord.ndim == 2 and coord.shape[1] == 3, (
        f"coord must have shape (N, 3), got {tuple(coord.shape)}!"
    )

    coord_f32 = coord.to(torch.float32)
    feat_ones = torch.ones_like(coord_f32[:, :1])
    feat_z_norm = coord_f32[:, 2:3] / 5.0
    xy_sq = torch.sum(coord_f32[:, :2] ** 2, dim=1, keepdim=True)
    feat_xy_range = torch.sqrt(xy_sq + 1e-12) / 100.0
    feat = torch.cat([feat_ones, feat_z_norm, feat_xy_range], dim=1)
    return feat
