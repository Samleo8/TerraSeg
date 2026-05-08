import torch
import torch.nn as nn


class PseudoLabeler(nn.Module):
    """
    PseudoLabeler.

    MLP that maps 2D coordinates (x,y) to scalar height value h.

    Note: Offline runtime optimization for a single point cloud!
    """

    METHOD_NAME = "PseudoLabeler"

    def __init__(
        self,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
    ):
        """
        Initialize network.

        Args:
            hidden_dim (int) : Number of hidden neurons.
            num_hidden_layers (int) : Number of hidden layers.
        """
        super().__init__()

        self.input_dim = 2
        self.output_dim = 1

        layers = []
        input_size = self.input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.SiLU())
            input_size = hidden_dim
        layers.append(nn.Linear(input_size, self.output_dim))

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        pc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predicts height for each (x,y) coordinate.

        Args:
            pc (torch.Tensor) : Point cloud with shape (N,2+), first two should be (x,y) coordinates. Unit: meters.

        Returns:
            pred_heights (torch.Tensor) : Predicted heights with shape (N,). Unit: meters.
        """
        assert pc.shape[1] >= 2, "Input point cloud must have shape (N,2+)!"

        xy = pc[:, :2]
        pred_heights = self.model(xy)[:, 0]
        return pred_heights

    @torch.no_grad()
    def get_ground_bool(
        self,
        pc: torch.Tensor,
        inlier_thres: float = 0.40,
    ) -> torch.Tensor:
        """
        Get ground segmentation boolean tensor.

        Args:
            pc (torch.Tensor) : Point cloud with shape (N,3+), first three should be (x,y,z) coordinates. Unit: meters.
            inlier_thres (float) : Maximum height distance for ground points in meters.

        Returns:
            bool_ground (bool) : Boolean tensor with shape (N,). Unit: 1.
        """
        assert pc.shape[1] >= 3, "Input point cloud must have shape (N,3+)!"

        pred_heights = self.forward(pc=pc)
        target_heights = pc[:, 2]

        bool_ground = (target_heights - pred_heights) <= inlier_thres
        return bool_ground

    @torch.no_grad()
    def remove_ego_points(
        self,
        pc: torch.Tensor,
        R_thres: float = 5.00,
        H_thres: float = 0.50,
    ) -> torch.Tensor:
        """
        Remove ego points from the point cloud.

        Args:
            pc (torch.Tensor) : Point cloud with shape (N,3+), first three should be (x,y,z) coordinates. Unit: meters.
            R_thres (float) : Radius threshold for ego points in meters.
            H_thres (float) : Height threshold for ego points in meters.

        Returns:
            egoremoved_pc (torch.Tensor) : Point cloud with ego points removed, shape (M,3+), first three are (x,y,z) coordinates. Unit: meters.
        """
        assert pc.shape[1] >= 3, "Input point cloud must have shape (N,3+)!"
        assert R_thres > 0.0, "Variable R_thres must be positive!"
        assert H_thres > 0.0, "Variable H_thres must be positive!"

        bool_radius = (pc[:, 0] ** 2 + pc[:, 1] ** 2) <= R_thres**2
        bool_height = pc[:, 2] >= H_thres
        egoremoved_pc = pc[~(bool_radius & bool_height), :]
        return egoremoved_pc

    @torch.no_grad()
    def preprocess_denoise_pc(
        self,
        pc: torch.Tensor,
        bottom_thres: float = 0.005,
    ) -> torch.Tensor:
        """
        Denoise point cloud by removing bottom points.

        Args:
            pc (torch.Tensor) : Point cloud with shape (N,3+), first three should be (x,y,z) coordinates. Unit: meters.
            bottom_thres (float) : Fraction for removing bottom points (denoising). Unit: 1.

        Returns:
            denoised_pc (torch.Tensor) : Denoised point cloud with shape (M,3+), first three are (x,y,z) coordinates. Unit: meters.
        """
        assert pc.dtype in [torch.float32, torch.float64], (
            f"Point cloud must be float32 or float64 for quantile calculation, but got {pc.dtype}!"
        )
        assert pc.shape[1] >= 3, "Input point cloud must have shape (N,3+)!"
        assert 0.0 <= bottom_thres <= 1.0, "Variable bottom_thres must be in [0,1]!"

        cutoff_z = torch.quantile(pc[:, 2], bottom_thres)
        bool_denoised = pc[:, 2] >= cutoff_z
        denoised_pc = pc[bool_denoised, :]
        return denoised_pc

    @torch.no_grad()
    def postprocess_recover_non_ground(
        self,
        pc: torch.Tensor,
        pred_labels: torch.Tensor,
        vxy: float = 0.50,
        H_p1: float = 1.5,
        H_p2: float = 1.5,
        tau: float = 0.05,
    ) -> torch.Tensor:
        """
        Recover non-ground points that were incorrectly classified as ground.

        Logic (per pillar):
        1) Predict pillar center height h using model.
        2) Consider points with z in window [h - H_p1, h + H_p2].
        3) If pillar has BOTH ground (0) and non-ground (1) inside that window,
            then any point with z >= (h_bottom + tau) becomes new non-ground.
        4) Points cannot go from non-ground to ground (conservative OR).

        Args:
            pc (torch.Tensor) : Point cloud with shape (N,3+), first three should be (x,y,z) coordinates. Unit: meters.
            pred_labels (torch.Tensor) : Predicted labels with shape (N,), with 0 := ground, 1 := non-ground. Unit: 1.
            vxy (float) : Pillar size in x and y directions. Unit: meters.
            H_p1 (float) : Height below predicted pillar height to consider points. Unit: meters.
            H_p2 (float) : Height above predicted pillar height to consider points. Unit: meters.
            tau (float) : Margin above pillar bottom height to consider points as non-ground. Unit: meters.

        Returns:
            new_pred_labels (torch.Tensor) : Post-processed non-ground boolean tensor with shape (N,), with 0 := ground, 1 := non-ground. Unit: 1.
        """
        assert pc.shape[1] >= 3, "Input point cloud must have shape (N,3+)!"
        assert pred_labels.shape[0] == pc.shape[0], (
            "Input label pred_labels must have shape (N,) matching pc!"
        )
        assert vxy > 0.0, "Variable vxy must be positive!"
        assert H_p1 > 0.0, "Variable H_p1 must be positive!"
        assert H_p2 > 0.0, "Variable H_p2 must be positive!"
        assert tau >= 0.0, "Variable tau must be non-negative!"

        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        xmin, xmax, ymin = x.min(), x.max(), y.min()

        # Pillarization.
        ix = torch.floor((x - xmin) / vxy).to(torch.int64)
        iy = torch.floor((y - ymin) / vxy).to(torch.int64)
        nx = int(torch.floor((xmax - xmin) / vxy).item()) + 1  # Number of pillars in x direction.
        pid = ix + iy * nx  # Shape (N,).

        # Unique pillars and centers.
        uniq_pid, inv = torch.unique(pid, return_inverse=True)
        iy_u = torch.div(uniq_pid, nx, rounding_mode="floor")
        ix_u = uniq_pid - iy_u * nx
        xy_centers = torch.stack(
            [xmin + (ix_u.to(x.dtype) + 0.5) * vxy, ymin + (iy_u.to(y.dtype) + 0.5) * vxy], dim=1
        )  # Shape (P,2).

        # Predict height once per pillar.
        pred_heights = self.forward(xy_centers)  # Shape (P,).
        pillar_heights = pred_heights[inv]  # Shape (N,).

        # Inside vertical window.
        bool_inside = (z >= (pillar_heights - H_p1)) & (z <= (pillar_heights + H_p2))  # Shape (N,).

        # Sort by pillar.
        order = torch.argsort(inv)
        inv_sorted = inv[order]
        z_sorted = z[order]
        pred_labels_sorted = pred_labels[order]
        bool_inside_sorted = bool_inside[order]

        # Lengths per pillar.
        lengths = torch.bincount(inv_sorted, minlength=uniq_pid.numel())

        # Per-pillar counts (use FLOAT input to segment_reduce).
        ground_count = torch.segment_reduce(
            data=(bool_inside_sorted & (pred_labels_sorted == 0)).float(),
            reduce="sum",
            lengths=lengths,
        )
        nonground_count = torch.segment_reduce(
            data=(bool_inside_sorted & (pred_labels_sorted == 1)).float(),
            reduce="sum",
            lengths=lengths,
        )
        bool_both = (ground_count > 0.0) & (nonground_count > 0.0)  # Shape (P,).

        # Per-pillar bottom height.
        z_masked = torch.where(
            bool_inside_sorted, z_sorted, torch.full_like(z_sorted, float("inf"))
        )
        pillar_bottom_height = torch.segment_reduce(
            data=z_masked, reduce="min", lengths=lengths
        )  # Shape (P,).

        # Map pillar-wise values back to points.
        bool_both_pt = bool_both[inv]
        pillar_bottom_height_pt = pillar_bottom_height[inv]

        # Candidates to flip (only if inside, only if pillar has both ground and non-ground points, and only if above bottom+tau).
        new_nonground = bool_inside & bool_both_pt & (z >= (pillar_bottom_height_pt + tau))

        # Conservative OR update (only add new non-ground).
        new_pred_labels = (pred_labels == 1) | new_nonground
        new_pred_labels = new_pred_labels.to(pred_labels.dtype)
        return new_pred_labels
