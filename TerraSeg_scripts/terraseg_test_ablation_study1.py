import json
import random
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import numpy as np
import torch
from tabulate import tabulate
from terraseg import build_terraseg, replace_bn1d_with_gn
from terraseg_dataloading import collate_scans
from torch.utils.data import DataLoader
from utils.dataset import OmniLiDARDataset
from utils.splits import eval_splits

# Paths.
results_root = Path("PUT_YOUR_DIRECTORY_HERE/TerraSeg__ablation_study1")
omnilidar_root = Path("PUT_YOUR_DIRECTORY_HERE/OmniLiDAR")
checkpoint_path = Path("PUT_YOUR_DIRECTORY_HERE/TerraSeg-B-or-S/best.pth")

assert str(results_root) != "PUT_YOUR_DIRECTORY_HERE/TerraSeg__ablation_study1", (
    "Folder for storing TerraSeg results. Change to directory in your file system!"
)
assert str(omnilidar_root) != "PUT_YOUR_DIRECTORY_HERE/OmniLiDAR", (
    "Directory to OmniLiDAR dataset. Change to directory in your file system!"
)
assert str(checkpoint_path) != "PUT_YOUR_DIRECTORY_HERE/TerraSeg-B-or-S/best.pth", (
    "Path to TerraSeg checkpoint. Change to checkpoint file in your file system!"
)
assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}!"

results_root.mkdir(exist_ok=True, parents=True)

# Released TerraSeg variant to evaluate. Set to "B" for the accurate Base model (~46M params)
# or "S" for the efficient Small model (~12M params). Must match the trained checkpoint.
VARIANT = "B"

# Hyperparameters that must match the trained model.
METHOD_NAME = f"TerraSeg-{VARIANT}"
SEED = 42
BATCH_SCANS = 1  # Per-scan metrics require batch_size=1.
NUM_WORKERS = 8
GRID_SIZE = 0.05
IGNORE_LABEL = 2
INPUT_DIM = 3

# Ablation hyperparameters.
SIGMA_RANGE = 50.0  # Cylindrical radius used to crop ground points before PCA. Unit: meters.
SIGMA_RESOLUTION = 0.5  # BEV cell size used for the cell-median statistic. Unit: meters.
SIGMA_CUTOFF = 0.40  # Flat / non-flat partition cutoff (paper Table 8). Unit: meters.

# Device and determinism.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Results directory.
run_id = checkpoint_path.parent.name
run_dir = results_root / f"{METHOD_NAME}__ablation_study1" / run_id
run_dir.mkdir(exist_ok=True, parents=True)
log_path = run_dir / "log.txt"
if log_path.exists():
    log_path.unlink()


def log_msg(msg: str) -> None:
    """
    Print a message and append it to the run log file.

    Args:
        msg (str) : Message to log.
    """
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


log_msg(f"Evaluating checkpoint: {checkpoint_path}")

# Load checkpoint.
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
saved_epoch = checkpoint.get("epoch", "?")
saved_step = checkpoint.get("global_step", "?")
decision_thres = float(checkpoint.get("decision_thres", 0.5))
log_msg(
    f"Checkpoint metadata: epoch={saved_epoch} global_step={saved_step} "
    f"decision_thres={decision_thres:.3f}\n"
)

# Model.
model = build_terraseg(variant=VARIANT, input_dim=INPUT_DIM, num_classes=1)
model = replace_bn1d_with_gn(model=model, groups_default=32)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()


@torch.no_grad()
def pca3_fast(
    X: torch.Tensor,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast PCA for (N, 3) point coordinates.

    Args:
        X (torch.Tensor) : (N, 3) float tensor of 3D coordinates.
        center (bool) : If True, subtract the per-axis mean before PCA.

    Returns:
        mu (torch.Tensor) : (3,) mean used for centering, or zeros if ``center=False``.
        components (torch.Tensor) : (3, 3) matrix whose rows are the principal axes
            (PC1, PC2, PC3) in descending order of explained variance.
        explained_var (torch.Tensor) : (3,) per-component variances, descending.
    """
    assert X.ndim == 2 and X.shape[1] == 3, "X must have shape (N, 3)!"
    X = X.float()
    mu = X.mean(dim=0) if center else torch.zeros(3, device=X.device, dtype=X.dtype)
    Xc = X - mu
    denom = max(Xc.shape[0] - 1, 1)
    C = (Xc.T @ Xc) / denom
    evals, evecs = torch.linalg.eigh(C)
    evals = evals.flip(0)
    evecs = evecs.flip(1)
    components = evecs.T
    explained_var = evals.clamp_min(0)
    return mu, components, explained_var


@torch.no_grad()
def per_scan_sigma(
    pc_lidar: torch.Tensor,
    bool_ground: torch.Tensor,
    cyl_radius: float = SIGMA_RANGE,
    spatial_resolution: float = SIGMA_RESOLUTION,
) -> float:
    """
    Compute the per-scan ground-elevation σ used in Table 8 of the TerraSeg paper.

    Pipeline:
    1. Keep ground points (``bool_ground``) within a cylindrical radius
        ``cyl_radius`` of the sensor.
    2. Run a 3D PCA on those points and select the principal component whose
        absolute cosine with the world z-axis is largest. This is the local
        ground-plane normal.
    3. Center only in z and rotate the points so the selected component
        becomes the new z-axis. This removes global ground tilt.
    4. In the rotated frame, bin points into BEV cells of size
        ``spatial_resolution`` and compute the median rotated-z per cell.
    5. Return the standard deviation of the cell medians over non-empty cells.

    Args:
        pc_lidar (torch.Tensor) : (N, 3) raw point cloud on the predictor's device.
        bool_ground (torch.Tensor) : (N,) boolean mask selecting ground points.
        cyl_radius (float) : Cylindrical xy-radius for the ground-point crop. Unit: meters.
        spatial_resolution (float) : BEV cell size. Unit: meters.

    Returns:
        sigma (float) : Per-scan σ in meters, or ``float("nan")`` if there are not
            enough ground points to evaluate PCA or fill any BEV cell.
    """
    bool_cyl = torch.linalg.norm(pc_lidar[:, :2], dim=1) < cyl_radius
    ground_points = pc_lidar[bool_cyl & bool_ground]
    if ground_points.shape[0] < 3:
        return float("nan")

    mu, components, _ = pca3_fast(X=ground_points, center=True)

    # Pick the principal component most aligned with the world z-axis.
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=pc_lidar.device).view(3, 1)
    cos_sims = torch.abs(components @ z_axis).flatten()
    selected_idx = int(torch.argmax(cos_sims).item())

    # Center only in z; rotate to PCA frame.
    mu_z_only = mu.clone()
    mu_z_only[:2] = 0.0
    projected = (ground_points - mu_z_only.view(1, 3)) @ components.T
    proj_z = projected[:, selected_idx]
    proj_xy = projected[:, [i for i in (0, 1, 2) if i != selected_idx]]

    # BEV cell medians in the rotated frame.
    xmin, xmax = -cyl_radius, cyl_radius
    ymin, ymax = -cyl_radius, cyl_radius
    nx = int(np.ceil((xmax - xmin) / spatial_resolution))
    ny = int(np.ceil((ymax - ymin) / spatial_resolution))

    ix = torch.floor((proj_xy[:, 0] - xmin) / spatial_resolution).long()
    iy = torch.floor((proj_xy[:, 1] - ymin) / spatial_resolution).long()
    bool_inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not bool_inside.any():
        return float("nan")
    ix, iy = ix[bool_inside], iy[bool_inside]
    z = proj_z[bool_inside]
    lin_ids = (iy * nx + ix).cpu().numpy()
    z_np = z.cpu().numpy()

    # Group by linear cell id, take median per non-empty cell, std across cells.
    order = np.argsort(lin_ids)
    sorted_ids = lin_ids[order]
    sorted_z = z_np[order]
    _, split_idxs = np.unique(sorted_ids, return_index=True)
    z_groups = np.split(sorted_z, split_idxs[1:])
    medians = np.array([np.median(g) for g in z_groups])
    if medians.size < 2:
        return float("nan")
    return float(np.std(medians, ddof=1))


def per_scan_confusion(
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    ignore_label: int = IGNORE_LABEL,
) -> dict:
    """
    Compute per-scan TP, FP, FN for the binary ground / non-ground task.

    Args:
        pred_labels (torch.Tensor) : (N,) per-point predictions in ``{0, 1}``.
        target_labels (torch.Tensor) : (N,) per-point ground-truth labels, possibly
            containing ``ignore_label``.
        ignore_label (int) : Label value to exclude from the confusion counts.

    Returns:
        cm (dict) : ``{"TP0", "FP0", "FN0", "TP1", "FP1", "FN1"}`` as Python ints.
    """
    bool_valid = target_labels != ignore_label
    pred = pred_labels[bool_valid]
    tgt = target_labels[bool_valid]
    tp0 = int(((pred == 0) & (tgt == 0)).sum().item())
    fp0 = int(((pred == 0) & (tgt == 1)).sum().item())
    fn0 = int(((pred == 1) & (tgt == 0)).sum().item())
    tp1 = int(((pred == 1) & (tgt == 1)).sum().item())
    fp1 = int(((pred == 1) & (tgt == 0)).sum().item())
    fn1 = int(((pred == 0) & (tgt == 1)).sum().item())
    return {"TP0": tp0, "FP0": fp0, "FN0": fn0, "TP1": tp1, "FP1": fp1, "FN1": fn1}


def aggregate_miou(scan_ids: list, per_scan_records: dict) -> tuple[float, float, float]:
    """
    Aggregate IoU0, IoU1, and mIoU over a list of scan IDs by summing
    confusion-matrix elements (NOT averaging per-scan mIoU).

    Args:
        scan_ids (list[str]) : Scan IDs to include in the aggregate.
        per_scan_records (dict) : Mapping ``scan_id -> {"sigma", "TP0", ..., "FN1"}``.

    Returns:
        iou0 (float) : Ground IoU in [0, 1], or ``nan`` if no valid scans.
        iou1 (float) : Non-ground IoU in [0, 1], or ``nan`` if no valid scans.
        miou (float) : Mean IoU = (iou0 + iou1) / 2.
    """
    if not scan_ids:
        return float("nan"), float("nan"), float("nan")
    tp0 = fp0 = fn0 = tp1 = fp1 = fn1 = 0
    for sid in scan_ids:
        r = per_scan_records.get(sid)
        if r is None:
            continue
        tp0 += r["TP0"]
        fp0 += r["FP0"]
        fn0 += r["FN0"]
        tp1 += r["TP1"]
        fp1 += r["FP1"]
        fn1 += r["FN1"]
    iou0 = tp0 / max(tp0 + fp0 + fn0, 1)
    iou1 = tp1 / max(tp1 + fp1 + fn1, 1)
    miou = 0.5 * (iou0 + iou1)
    return iou0, iou1, miou


# Validation datasets and dataloaders. batch_size=1 because the ablation is per-scan.
val_datasets = {
    dataset_name: OmniLiDARDataset(
        root=omnilidar_root,
        splits={dataset_name: split_name},
        pseudolabels_root="",
        remove_ego_points=False,
        compute_scans_metadata=False,
    )
    for dataset_name, split_name in eval_splits.items()
}
val_loaders = {
    dataset_name: DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SCANS,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda samples: collate_scans(
            samples=samples,
            grid_size=GRID_SIZE,
            ignore_label=IGNORE_LABEL,
            is_train=False,
            aug_cfg=None,
        ),
        drop_last=False,
    )
    for dataset_name, val_dataset in val_datasets.items()
}

# Per-scan inference + σ + confusion matrix accumulation.
all_records: dict = {}  # ``{dataset_name: {scan_id: record}}``.
for dataset_name, val_loader in val_loaders.items():
    log_msg(
        f"[{datetime.now().strftime('%H:%M:%S')}] Per-scan inference on "
        f"{dataset_name} ({len(val_loader.dataset)} scans)..."
    )
    dataset_records: dict = {}
    for batch_idx, batch in enumerate(val_loader):
        # Move tensors to device.
        for key in ("coord", "feat", "offset", "target", "segment"):
            batch[key] = batch[key].to(device, non_blocking=True)

        with torch.inference_mode():
            pred_logits = model(batch)
        pred_probs = torch.sigmoid(pred_logits.float())
        pred_labels = (pred_probs > decision_thres).long()

        target_labels = batch["segment"]
        coord = batch["coord"]

        # Per-scan σ (uses GT ground points, as in the paper).
        bool_ground_gt = target_labels == 0
        sigma = per_scan_sigma(pc_lidar=coord, bool_ground=bool_ground_gt)

        # Per-scan confusion matrix.
        cm = per_scan_confusion(pred_labels=pred_labels, target_labels=target_labels)

        scan_id = f"scan_{batch_idx:06d}"
        dataset_records[scan_id] = {"sigma": sigma, **cm}

    all_records[dataset_name] = dataset_records
    log_msg(
        f"[{datetime.now().strftime('%H:%M:%S')}] {dataset_name} | "
        f"median σ = {np.nanmedian([r['sigma'] for r in dataset_records.values()]):.4f} m"
    )

    # Per-scan dump so the user can audit the values.
    per_scan_json_path = run_dir / f"{dataset_name}__per_scan.json"
    with open(per_scan_json_path, "w") as f:
        json.dump(dataset_records, f, indent=2)

# Aggregate: per-dataset, per σ-partition mIoU.
log_msg(f"\nFlat / non-flat partition cutoff: σ ≤ {SIGMA_CUTOFF:.2f} m vs σ > {SIGMA_CUTOFF:.2f} m")
rows = []
for dataset_name, dataset_records in all_records.items():
    sigmas = np.array([r["sigma"] for r in dataset_records.values()], dtype=np.float64)
    scan_ids = list(dataset_records.keys())
    bool_valid = ~np.isnan(sigmas)
    valid_ids = [sid for sid, ok in zip(scan_ids, bool_valid) if ok]
    valid_sigmas = sigmas[bool_valid]

    flat_ids = [sid for sid, s in zip(valid_ids, valid_sigmas) if s <= SIGMA_CUTOFF]
    non_flat_ids = [sid for sid, s in zip(valid_ids, valid_sigmas) if s > SIGMA_CUTOFF]

    _, _, miou_flat = aggregate_miou(scan_ids=flat_ids, per_scan_records=dataset_records)
    _, _, miou_non_flat = aggregate_miou(scan_ids=non_flat_ids, per_scan_records=dataset_records)

    rows.append(
        [
            dataset_name,
            f"{100 * miou_flat:.2f}" if not np.isnan(miou_flat) else "nan",
            f"{100 * miou_non_flat:.2f}" if not np.isnan(miou_non_flat) else "nan",
            f"{len(flat_ids)}",
            f"{len(non_flat_ids)}",
        ]
    )

table = tabulate(
    rows,
    headers=[
        "Dataset",
        f"mIoU (σ ≤ {SIGMA_CUTOFF:.2f} m)",
        f"mIoU (σ > {SIGMA_CUTOFF:.2f} m)",
        "N flat",
        "N non-flat",
    ],
    tablefmt="pretty",
    stralign="center",
    numalign="center",
)
log_msg("\n" + table + "\n")

results_path = run_dir / "results.txt"
with open(results_path, "w") as f:
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Decision threshold: {decision_thres:.4f}\n")
    f.write(
        f"σ pipeline: cylindrical radius {SIGMA_RANGE:.1f} m, PCA tilt-corrected, "
        f"BEV cell size {SIGMA_RESOLUTION:.2f} m, partition cutoff {SIGMA_CUTOFF:.2f} m\n\n"
    )
    f.write(table + "\n")
log_msg(f"Wrote results table to {results_path}")
