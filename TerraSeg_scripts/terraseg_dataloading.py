import math

import numpy as np
import torch
from torch.utils.data import Sampler

from terraseg import compute_terraseg_features

# Default per-scan augmentation configuration used during training.
DEFAULT_AUG_CFG: dict = {
    "max_rot_deg": (2.00, 2.00, 10.00),
    "max_translation": (0.20, 0.20, 0.10),
    "jitter_std": 0.005,
    "jitter_clip": 0.02,
    "drop_ratio": 0.10,
    "scale_range": (0.90, 1.10),
    "flip_prob": 0.50,
    "z_rot_full": True,
    "drop_apply_prob": 0.20,
}


# Dataset sampling probabilities described in the TerraSeg paper (section A.1).
GROUP_PROBS: dict = {
    # Primary datasets (50 percent total).
    "nuScenes__main": 0.20,
    "KITTI__main": 0.20,
    "WaymoPerception__main": 0.10,
    # Large datasets (30 percent total).
    "AV2_Lidar": 0.10,
    "ONCE": 0.10,
    "ZOD": 0.10,
    # Small datasets (15 percent total).
    "AevaScenes": 0.05,
    "Lyft": 0.05,
    "TruckScenes": 0.05,
    # Very small dataset (2.5 percent).
    "PandaSet": 0.0125,
    "VoD": 0.0125,
    # Fallback group (2.5 percent total).
    "else": 0.025,
}


def rand_se3_augment(
    pc: torch.Tensor,
    max_rot_deg: tuple[float, float, float] = (2.00, 2.00, 10.00),
    max_translation: tuple[float, float, float] = (0.20, 0.20, 0.10),
    jitter_std: float = 0.005,
    jitter_clip: float = 0.02,
    drop_ratio: float = 0.10,
    scale_range: tuple[float, float] = (0.90, 1.10),
    flip_prob: float = 0.5,
    z_rot_full: bool = True,
    drop_apply_prob: float = 0.20,
    g: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply random SE(3) augmentation to a point cloud.

    Args:
        pc (torch.Tensor) : (N,3) input point cloud. Unit: meters.
        max_rot_deg (tuple[float,float,float]) : Maximum roll, pitch, yaw magnitudes in degrees.
        max_translation (tuple[float,float,float]) : Maximum X, Y, Z translation. Unit: meters.
        jitter_std (float) : Per-point Gaussian jitter standard deviation. Unit: meters.
        jitter_clip (float) : Maximum absolute jitter magnitude after clipping. Unit: meters.
        drop_ratio (float) : Fraction of points dropped when dropout is applied. Unit: 1.
        scale_range (tuple[float,float]) : Isotropic scaling range (min, max). Unit: 1.
        flip_prob (float) : Probability of flipping along each of the X and Y axes. Unit: 1.
        z_rot_full (bool) : If True, yaw is sampled uniformly in [-pi, pi].
        drop_apply_prob (float) : Probability of applying dropout to this scan. Unit: 1.
        g (torch.Generator or None) : Optional random generator for reproducibility.

    Returns:
        pc_aug (torch.Tensor) : (M,3) augmented point cloud.
        bool_keep (torch.Tensor or None) : (N,) boolean mask of kept points if dropout was
            applied, otherwise None.
    """
    if g is None:
        g = torch.Generator(device=pc.device)
        g.manual_seed(torch.seed())

    # Dropout (optionally applied to this scan).
    bool_keep = None
    bool_apply_drop = (drop_ratio > 0.0) and (
        torch.rand((), device=pc.device, generator=g) < drop_apply_prob
    )
    if bool_apply_drop and pc.numel() > 0:
        bool_keep = torch.rand((pc.shape[0],), device=pc.device, generator=g) > drop_ratio
        if bool_keep.any():
            pc = pc[bool_keep]
        else:
            # Keep at least one point.
            keep_idx = torch.randint(pc.shape[0], (1,), device=pc.device, generator=g)
            bool_keep = torch.zeros((pc.shape[0],), dtype=torch.bool, device=pc.device)
            bool_keep[keep_idx] = True
            pc = pc[keep_idx]

    # Isotropic random scaling.
    if scale_range is not None:
        scale_lo, scale_hi = scale_range
        r = torch.rand((), device=pc.device, generator=g)
        scale_factor = (scale_lo + (scale_hi - scale_lo) * r).to(dtype=pc.dtype)
        pc = pc * scale_factor

    # Random rotation: full yaw + small roll and pitch.
    to_rad = math.pi / 180.0
    rp = torch.rand(2, device=pc.device, generator=g) * 2.0 - 1.0  # Two values in [-1,1].
    roll = rp[0] * (max_rot_deg[0] * to_rad)
    pitch = rp[1] * (max_rot_deg[1] * to_rad)
    if z_rot_full:
        yaw = (torch.rand((), device=pc.device, generator=g) * 2.0 - 1.0) * math.pi
    else:
        yaw = (torch.rand((), device=pc.device, generator=g) * 2.0 - 1.0) * (
            max_rot_deg[2] * to_rad
        )

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    R = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=pc.dtype,
        device=pc.device,
    )
    pc = pc @ R.T

    # Random flips along X and Y axes.
    if torch.rand((), device=pc.device, generator=g) < flip_prob:
        pc[:, 0] = -pc[:, 0]
    if torch.rand((), device=pc.device, generator=g) < flip_prob:
        pc[:, 1] = -pc[:, 1]

    # Small random translation.
    max_t = torch.tensor(max_translation, device=pc.device, dtype=pc.dtype)
    t = (torch.rand(3, device=pc.device, generator=g) * 2.0 - 1.0) * max_t
    pc = pc + t

    # Per-point Gaussian jitter (clipped).
    if jitter_std > 0.0:
        noise = torch.randn(pc.shape, device=pc.device, dtype=pc.dtype, generator=g) * jitter_std
        noise = noise.clamp_(-jitter_clip, jitter_clip)
        pc = pc + noise

    return pc, bool_keep


def collate_scans(
    samples: list[dict],
    grid_size: float = 0.05,
    ignore_label: int = 2,
    is_train: bool = False,
    aug_cfg: dict | None = None,
) -> dict:
    """
    Collate a list of OmniLiDAR scan samples into a single PTv3-ready batch dict.
 
    Per-point features used by TerraSeg: a constant ones channel, a normalized height channel
    ``z / 5.0``, and a normalized horizontal range channel ``||(x,y)||_2 / 100.0``. Raw (x,y,z)
    coordinates are kept separately and passed through PTv3 as the ``coord`` tensor.
 
    Empty scans (e.g. the ~3k AV2_Lidar placeholder files that OmniLiDAR keeps to preserve
    sequence indexing; the paper's statistics exclude them) are dropped here. They would
    otherwise add zero-length entries to ``offset`` and could collapse a minibatch to zero
    points when ``batch_scans`` is small, which crashes PTv3 in ``serialization``.
 
    Args:
        samples (list[dict]) : List of dataset samples from :class:`utils.dataset.OmniLiDARDataset`.
        grid_size (float) : Voxel grid size passed to PTv3. Unit: meters.
        ignore_label (int) : Label value used when a sample has no targets. Default: 2.
        is_train (bool) : If True, apply SE(3) augmentation.
        aug_cfg (dict or None) : Augmentation hyperparameters. Falls back to ``DEFAULT_AUG_CFG``.
 
    Returns:
        batch (dict) : Batched dict with keys ``coord``, ``feat``, ``offset``, ``target``,
            ``segment``, ``condition``, ``grid_size``, ``ignore_label``. ``coord`` has shape
            ``(0, 3)`` and ``offset`` is empty when every scan in ``samples`` was empty; the
            training loop should detect this and skip the step.
    """
    coords: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    offsets: list[int] = []
    conditions: list[str] = []
    num_acc = 0
 
    # Reproducible per-worker random generator for augmentation.
    worker_info = torch.utils.data.get_worker_info()
    g = torch.Generator()
    g.manual_seed(worker_info.seed if worker_info is not None else torch.seed())
 
    cfg = aug_cfg if aug_cfg is not None else DEFAULT_AUG_CFG
 
    for sample in samples:
        pc = sample["x"].cpu().float()
 
        # Drop empty scans before anything else: they cannot contribute to the batch, they
        # would add a zero-length entry to ``offset``, and they trigger downstream crashes
        # in PTv3 when they happen to constitute the entire minibatch.
        if pc.shape[0] == 0:
            continue
 
        target = (
            sample["target_pseudolabels"]
            if sample.get("target_pseudolabels") is not None
            else sample.get("target_labels")
        )
        if target is not None:
            target = target.cpu().long()
 
        # SE(3) augmentation during training.
        if is_train:
            pc, bool_keep = rand_se3_augment(
                pc=pc,
                max_rot_deg=cfg.get("max_rot_deg", (2.00, 2.00, 10.00)),
                max_translation=cfg.get("max_translation", (0.20, 0.20, 0.10)),
                jitter_std=cfg.get("jitter_std", 0.005),
                jitter_clip=cfg.get("jitter_clip", 0.02),
                drop_ratio=cfg.get("drop_ratio", 0.10),
                scale_range=cfg.get("scale_range", (0.90, 1.10)),
                flip_prob=cfg.get("flip_prob", 0.50),
                z_rot_full=cfg.get("z_rot_full", True),
                drop_apply_prob=cfg.get("drop_apply_prob", 0.20),
                g=g,
            )
            if target is not None and bool_keep is not None:
                target = target[bool_keep]
 
        # Fill ignore_label for samples without targets.
        if target is None:
            target = torch.full((pc.shape[0],), ignore_label, dtype=torch.long)
 
        coords.append(pc)
        labels.append(target)
        num_acc += pc.shape[0]
        offsets.append(num_acc)
        conditions.append(sample["source_dataset"])
 
    # Concatenate.
    coord = torch.cat(coords, dim=0) if coords else torch.empty(0, 3, dtype=torch.float32)
    target = torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long)
    offset = (
        torch.tensor(offsets, dtype=torch.long) if offsets else torch.empty(0, dtype=torch.long)
    )
 
    # 3D per-point features used by TerraSeg (shared with the deployment predictor).
    feat = compute_terraseg_features(coord=coord)
 
    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "target": target,
        "segment": target,
        "condition": conditions,
        "grid_size": grid_size,
        "ignore_label": ignore_label,
    }


def dataset_to_group(
    dataset_name: str,
    sensor_name: str,
) -> str:
    """
    Map a (dataset, sensor) pair to a TerraSeg sampling group, as defined in paper (section A.1).

    Args:
        dataset_name (str) : Source dataset name (e.g. "nuScenes", "WaymoPerception").
        sensor_name (str) : Sensor name within the dataset (e.g. "TOP" for Waymo).

    Returns:
        group (str) : Sampling group key in :data:`GROUP_PROBS`.
    """
    if dataset_name == "nuScenes":
        return "nuScenes__main"
    if dataset_name in ("KITTI360", "SemanticKITTI"):
        return "KITTI__main"
    if dataset_name == "WaymoPerception" and sensor_name == "TOP":
        return "WaymoPerception__main"
    if dataset_name == "AevaScenes":
        return "AevaScenes"
    if dataset_name == "AV2_Lidar":
        return "AV2_Lidar"
    if dataset_name == "Lyft":
        return "Lyft"
    if dataset_name == "ONCE":
        return "ONCE"
    if dataset_name == "PandaSet":
        return "PandaSet"
    if dataset_name == "TruckScenes":
        return "TruckScenes"
    if dataset_name == "VoD":
        return "VoD"
    if dataset_name == "ZOD":
        return "ZOD"
    return "else"


class BalancedGroupSampler(Sampler):
    """
    Weighted dataset sampler that yields a fixed number of indices per epoch.

    Each index is drawn by first sampling a dataset group (according to ``group_probs``),
    then drawing without replacement from that group until the group is exhausted, at
    which point that group is re-shuffled. This matches the sampling described in the
    TerraSeg paper (section A.1).

    Args:
        group_indices (dict[str, list[int]]) : Mapping from group name to the global indices
            belonging to that group.
        group_probs (dict[str, float]) : Mapping from group name to its sampling probability.
            Probabilities are renormalized to sum to 1.
        total_samples (int) : Number of indices yielded per epoch.
        rng_seed (int or None) : Optional seed for the internal numpy RNG, for reproducibility.
    """

    def __init__(
        self,
        group_indices: dict,
        group_probs: dict,
        total_samples: int,
        rng_seed: int | None = None,
    ):
        assert total_samples > 0, "total_samples must be positive!"
        assert set(group_indices.keys()) == set(group_probs.keys()), (
            "group_indices and group_probs must have the same keys!"
        )
        for name, indices in group_indices.items():
            assert len(indices) > 0, f"Group '{name}' is empty!"

        self.group_indices = group_indices
        self.total_samples = int(total_samples)

        prob_sum = sum(group_probs.values())
        self.group_probs = {k: v / prob_sum for k, v in group_probs.items()}

        self.group_names = list(self.group_probs.keys())
        self.prob_array = np.array(
            [self.group_probs[k] for k in self.group_names], dtype=np.float64
        )

        self._rng = np.random.default_rng(rng_seed)
        self._reset_iterators()

    def _reset_iterators(
        self,
    ) -> None:
        """
        Shuffle each group's indices and reset internal cursors.
        """
        self.group_iters = {
            name: self._rng.permutation(np.asarray(indices)).tolist()
            for name, indices in self.group_indices.items()
        }

    def __iter__(
        self,
    ):
        """
        Yield ``total_samples`` global indices for one training epoch.
        """
        for _ in range(self.total_samples):
            chosen_group = self._rng.choice(self.group_names, p=self.prob_array)
            if len(self.group_iters[chosen_group]) == 0:
                self.group_iters[chosen_group] = self._rng.permutation(
                    np.asarray(self.group_indices[chosen_group])
                ).tolist()
            yield int(self.group_iters[chosen_group].pop())

    def __len__(
        self,
    ) -> int:
        """
        Return the number of indices yielded per epoch.
        """
        return self.total_samples


def build_group_indices(
    dataset_pairs: list[tuple[str, str]],
    cum_blocks: list[int],
    group_probs: dict | None = None,
) -> tuple[dict, dict]:
    """
    Build per-group global index lists from an :class:`OmniLiDARDataset`'s metadata.

    Args:
        dataset_pairs (list[tuple[str,str]]) : The dataset's ``pairs`` attribute, listing
            ``(split_name, sensor_name)`` blocks in order.
        cum_blocks (list[int]) : The dataset's ``cum_blocks`` attribute (cumulative scan counts).
        group_probs (dict or None) : Sampling probabilities to use. Empty groups are dropped from
            this dict and from the returned ``group_indices``. Defaults to :data:`GROUP_PROBS`.

    Returns:
        group_indices (dict[str, list[int]]) : Mapping from group name to global indices.
        group_probs_used (dict[str, float]) : Filtered probabilities matching the non-empty groups.
    """
    probs = dict(group_probs) if group_probs is not None else dict(GROUP_PROBS)
    indices: dict[str, list[int]] = {name: [] for name in probs.keys()}

    start_idx = 0
    for i, (split_name, sensor_name) in enumerate(dataset_pairs):
        end_idx = cum_blocks[i]
        dataset_name = split_name.split("/")[0]
        group = dataset_to_group(dataset_name=dataset_name, sensor_name=sensor_name)
        if group not in indices:
            indices[group] = []
            probs.setdefault(group, 0.0)
        indices[group].extend(range(start_idx, end_idx))
        start_idx = end_idx

    indices = {name: idxs for name, idxs in indices.items() if len(idxs) > 0}
    probs_used = {name: p for name, p in probs.items() if name in indices}
    return indices, probs_used
