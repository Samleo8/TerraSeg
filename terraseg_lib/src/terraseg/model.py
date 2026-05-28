import torch
import torch.nn as nn

from ptv3 import PointTransformerV3

# Named PTv3 backbone configurations for the two TerraSeg variants released in the paper.
# Reported parameter counts: TerraSeg-B ~46M, TerraSeg-S ~12M.
#
# TerraSeg-B uses the default PTv3 sizes (omitting the four size kwargs from the
# PointTransformerV3 constructor leaves them at the upstream defaults).
# TerraSeg-S halves the channel widths and quarters the patch sizes at every stage.
TERRASEG_B_CONFIG: dict = {
    "backbone_dim": 64,
    "ptv3_kwargs": {},
}

TERRASEG_S_CONFIG: dict = {
    "backbone_dim": 32,
    "ptv3_kwargs": {
        "enc_channels": (16, 32, 64, 128, 256),
        "enc_patch_size": (256, 256, 256, 256, 256),
        "dec_channels": (32, 32, 64, 128),
        "dec_patch_size": (256, 256, 256, 256),
    },
}


class TerraSeg(nn.Module):
    """
    TerraSeg: self-supervised, domain-agnostic model for LiDAR ground segmentation.

    Architecture:
        (1) Point Transformer v3 (PTv3) backbone with dataset-specific normalization disabled
            (``pdnorm_conditions=None``) so the model learns universal geometric priors.
        (2) Lightweight MLP classification head producing per-point binary logits:
            LayerNorm -> Linear -> GELU -> Linear.

    Note:
        Input features must match the 3-dimensional vector described in the paper:
        (constant ones, normalized height z/5.0, normalized horizontal range ||xy||/100.0).
        Raw (x, y, z) coordinates are reserved for constructing PTv3's spatial voxel grid.

    Args:
        input_dim (int) : Input per-point feature dimension. Default: 3.
        backbone_dim (int) : PTv3 output feature dimension (i.e. ``dec_channels[0]``). Must
            match the first decoder channel. Default: 64 (the TerraSeg-B value).
        num_classes (int) : Number of output logits. Default: 1 (BCE for binary ground/non-ground).
        ptv3_kwargs (dict or None) : Optional additional keyword arguments forwarded verbatim to
            :class:`ptv3.PointTransformerV3`. Use this to set ``enc_channels``,
            ``enc_patch_size``, ``dec_channels``, ``dec_patch_size`` for variants like TerraSeg-S.
    """

    def __init__(
        self,
        input_dim: int = 3,
        backbone_dim: int = 64,
        num_classes: int = 1,
        ptv3_kwargs: dict | None = None,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.backbone_dim = int(backbone_dim)
        self.num_classes = int(num_classes)

        ptv3_init_kwargs = {"in_channels": self.input_dim, "pdnorm_conditions": None}
        if ptv3_kwargs is not None:
            ptv3_init_kwargs.update(ptv3_kwargs)

        if "dec_channels" in ptv3_init_kwargs:
            assert ptv3_init_kwargs["dec_channels"][0] == self.backbone_dim, (
                f"backbone_dim ({self.backbone_dim}) must equal dec_channels[0] "
                f"({ptv3_init_kwargs['dec_channels'][0]})!"
            )

        self.backbone = PointTransformerV3(**ptv3_init_kwargs)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, self.backbone_dim // 4, bias=False),
            nn.GELU(),
            nn.Linear(self.backbone_dim // 4, self.num_classes),
        )

    def forward(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch (dict) : Batched point cloud in PTv3 input format with keys:
                - "coord" (torch.Tensor) : (N_tot,3) XYZ coordinates. Unit: meters.
                - "feat" (torch.Tensor) : (N_tot,input_dim) per-point features.
                - "offset" (torch.Tensor) : (B,) cumulative point counts per scan.
                - "grid_size" (float) : Voxel grid size. Unit: meters.
                - "condition" (list[str]) : Per-scan dataset name tags.

        Returns:
            pred_logits (torch.Tensor) : (N_tot,) per-point ground/non-ground logits.
        """
        ptv3_in = {
            "coord": batch["coord"],
            "feat": batch["feat"],
            "offset": batch["offset"],
            "grid_size": batch["grid_size"],
            "condition": batch["condition"],
        }
        point_features = self.backbone(ptv3_in).feat
        pred_logits = self.cls_head(point_features).squeeze(-1)
        return pred_logits


def build_terraseg(
    variant: str,
    input_dim: int = 3,
    num_classes: int = 1,
) -> TerraSeg:
    """
    Build a :class:`TerraSeg` model for the requested released variant.

    Args:
        variant (str) : Either ``"B"`` (Base, ~46M parameters, default PTv3 sizes) or ``"S"``
            (Small, ~12M parameters, halved channels and quartered patch sizes).
        input_dim (int) : Input per-point feature dimension. Default: 3.
        num_classes (int) : Number of output logits. Default: 1.

    Returns:
        model (TerraSeg) : Instantiated TerraSeg model.
    """
    variant = variant.upper()
    if variant == "B":
        config = TERRASEG_B_CONFIG
    elif variant == "S":
        config = TERRASEG_S_CONFIG
    else:
        raise ValueError(f"Unknown TerraSeg variant '{variant}'. Expected 'B' or 'S'.")

    return TerraSeg(
        input_dim=input_dim,
        backbone_dim=config["backbone_dim"],
        num_classes=num_classes,
        ptv3_kwargs=config["ptv3_kwargs"],
    )
