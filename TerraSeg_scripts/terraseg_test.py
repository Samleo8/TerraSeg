import random
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import numpy as np
import torch
from tabulate import tabulate
from terraseg_dataloading import collate_scans
from terraseg_utils import evaluate_loader
from torch.utils.data import DataLoader
from utils.dataset import OmniLiDARDataset
from utils.splits import eval_splits

from terraseg import build_terraseg, replace_bn1d_with_gn

# Paths.
results_root = Path("PUT_YOUR_DIRECTORY_HERE/TerraSeg_results")
omnilidar_root = Path("PUT_YOUR_DIRECTORY_HERE/OmniLiDAR")
checkpoint_path = Path("PUT_YOUR_DIRECTORY_HERE/TerraSeg-B-or-S/best.pth")

assert str(results_root) != "PUT_YOUR_DIRECTORY_HERE/TerraSeg_results", (
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
BATCH_SCANS = 4
NUM_WORKERS = 8
GRID_SIZE = 0.05
IGNORE_LABEL = 2
INPUT_DIM = 3

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

# Results directory. Derive the eval folder from the training run id (the parent folder of the
# checkpoint) so re-evaluating the same checkpoint overwrites the previous evaluation results
# instead of accumulating new timestamped folders. Mirrors the training layout exactly:
# training writes to `results_root / METHOD_NAME / <run_id>`, evaluation writes to
# `results_root / f"{METHOD_NAME}__eval" / <run_id>`.
run_id = checkpoint_path.parent.name
run_dir = results_root / f"{METHOD_NAME}__eval" / run_id
run_dir.mkdir(exist_ok=True, parents=True)
log_path = run_dir / "log.txt"
if log_path.exists():
    log_path.unlink()  # Fresh log on each evaluation run.


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
state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict", checkpoint))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

# Validation datasets and dataloaders.
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

# Run evaluation.
rows = []
mious = []
for dataset_name, val_loader in val_loaders.items():
    metrics = evaluate_loader(
        model=model,
        val_loader=val_loader,
        device=device,
        decision_thres=decision_thres,
        ignore_label=IGNORE_LABEL,
    )
    mious.append(metrics["miou"])
    log_msg(
        f"[{datetime.now().strftime('%H:%M:%S')}] {dataset_name} | "
        f"recall0 {100 * metrics['recall0']:.2f} "
        f"precision0 {100 * metrics['precision0']:.2f} "
        f"IoU0 {100 * metrics['iou0']:.2f} "
        f"recall1 {100 * metrics['recall1']:.2f} "
        f"precision1 {100 * metrics['precision1']:.2f} "
        f"IoU1 {100 * metrics['iou1']:.2f} "
        f"mIoU {100 * metrics['miou']:.2f}"
    )
    rows.append(
        [
            dataset_name,
            f"{100 * metrics['recall0']:.2f}",
            f"{100 * metrics['precision0']:.2f}",
            f"{100 * metrics['iou0']:.2f}",
            f"{100 * metrics['recall1']:.2f}",
            f"{100 * metrics['precision1']:.2f}",
            f"{100 * metrics['iou1']:.2f}",
            f"{100 * metrics['miou']:.2f}",
        ]
    )

mean_miou = float(np.mean(mious)) if mious else float("nan")
rows.append(["MEAN", "", "", "", "", "", "", f"{100 * mean_miou:.2f}"])

table = tabulate(
    rows,
    headers=[
        "Dataset",
        "Recall0",
        "Precision0",
        "IoU0",
        "Recall1",
        "Precision1",
        "IoU1",
        "mIoU",
    ],
    tablefmt="pretty",
    stralign="center",
    numalign="center",
)
log_msg("\n" + table + "\n")

results_path = run_dir / "results.txt"
with open(results_path, "w") as f:
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Decision threshold: {decision_thres:.4f}\n\n")
    f.write(table + "\n")
log_msg(f"Wrote results table to {results_path}")
