import random
import sys
from pathlib import Path

import numpy as np
import torch
from pseudolabeler_loss import pseudolabeler_loss
from pseudolabeler_model import PseudoLabeler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import OmniLiDARDataset
from utils.evaluate import get_confusion_matrix, get_standard_metrics
from utils.splits import eval_splits

# Paths.
results_root = Path("PUT_YOUR_DIRECTORY_HERE/PseudoLabeler__ablation_study1__base")
omnilidar_root = Path("PUT_YOUR_DIRECTORY_HERE/OmniLiDAR")

assert str(results_root) != "PUT_YOUR_DIRECTORY_HERE/PseudoLabeler__ablation_study1__base", (
    "Folder for storing PseudoLabeler results. Change to directory in your file system!"
)
assert str(omnilidar_root) != "PUT_YOUR_DIRECTORY_HERE/OmniLiDAR", (
    "Directory to OmniLiDAR dataset. Change to directory in your file system!"
)

project_root = Path.cwd().parent
sys.path.append(str(project_root))

results_root.mkdir(exist_ok=True, parents=True)
omnilidar_root.mkdir(exist_ok=True, parents=True)

# Device and Determinism.
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Create dataset.
dataset = OmniLiDARDataset(
    root=omnilidar_root,
    splits=eval_splits,
    pseudolabels_root="",  # Empty: We need to create these pseudo-labels first!
    remove_ego_points=False,
    compute_scans_metadata=True,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda batch: batch[0],
    pin_memory=True,
)

# Predict.
for sample_dict in tqdm(dataloader, desc="Scans"):
    # Check if pseudo-labels already exist.
    file_name = str(sample_dict["x_dir"].name).replace("pointcloud", "labels")
    labels_path = (
        results_root
        / sample_dict["source_dataset"]
        / sample_dict["source_split"]
        / "labels"
        / file_name
    )
    labels_path = labels_path.with_suffix(".npy")
    if labels_path.exists():
        continue

    # Get data.
    pc_lidar = sample_dict["x"].to(device, non_blocking=True)

    # Set seeds.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Get model.
    model = PseudoLabeler().to(device)
    model.train()

    # Optimize.
    egoremoved_pc = model.remove_ego_points(pc=pc_lidar)
    pc = egoremoved_pc.clone()

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=50)

    ema_loss = None
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    loss_history = []
    ema_loss_history = []

    for _ in range(2500):
        # Train step.
        optimizer.zero_grad()
        pred_heights = model(pc)
        target_heights = pc[:, 2]
        loss = pseudolabeler_loss(pred_heights, target_heights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step(loss.item())

        # Bookkeeping.
        current_loss = loss.item()
        loss_history.append(current_loss)
        ema_loss = current_loss if ema_loss is None else 0.90 * ema_loss + 0.10 * current_loss
        ema_loss_history.append(ema_loss)

        # Early stopping.
        if ema_loss < best_loss:
            best_loss = ema_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 150:
                break

    # Predict.
    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        model.eval()
        pc_lidar = sample_dict["x"].to(device)
        bool_ground = model.get_ground_bool(pc=pc_lidar)
        pred_labels = (~bool_ground).to(torch.uint8)

    # Save pseudo-labels.
    file_name = str(sample_dict["x_dir"].name).replace("pointcloud", "labels")
    labels_path = (
        results_root
        / sample_dict["source_dataset"]
        / sample_dict["source_split"]
        / "labels"
        / file_name
    )
    labels_path = labels_path.with_suffix(".npy")
    labels_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(labels_path, pred_labels.cpu().numpy())


# Eval.
for dataset_name, split_name in zip(eval_splits.keys(), eval_splits.values()):
    dataset = OmniLiDARDataset(
        root=omnilidar_root,
        splits={dataset_name: split_name},
        pseudolabels_root=results_root,
        remove_ego_points=False,
        compute_scans_metadata=True,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: batch[0],
        pin_memory=True,
    )

    total_tp0, total_fn0, total_fp0, total_tn0 = (
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
    )
    total_tp1, total_fn1, total_fp1, total_tn1 = (
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
    )
    for sample_dict in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        target_labels = sample_dict["target_labels"]
        target_pseudolabels = sample_dict["target_pseudolabels"]

        tp0, fn0, fp0, tn0 = get_confusion_matrix(
            gt_labels=target_labels,
            pred_labels=target_pseudolabels,
            selected_id=0,
            ignore_id=2,
        )

        tp1, fn1, fp1, tn1 = get_confusion_matrix(
            gt_labels=target_labels,
            pred_labels=target_pseudolabels,
            selected_id=1,
            ignore_id=2,
        )

        total_tp0 += tp0
        total_fn0 += fn0
        total_fp0 += fp0
        total_tn0 += tn0

        total_tp1 += tp1
        total_fn1 += fn1
        total_fp1 += fp1
        total_tn1 += tn1

    metrics_dict0 = get_standard_metrics(
        tp=total_tp0,
        fn=total_fn0,
        fp=total_fp0,
        tn=total_tn0,
    )

    metrics_dict1 = get_standard_metrics(
        tp=total_tp1,
        fn=total_fn1,
        fp=total_fp1,
        tn=total_tn1,
    )

    miou = (metrics_dict0["iou"] + metrics_dict1["iou"]) / 2

    print(f"{dataset_name} Metrics:")
    print(
        f"Class 0 (ground) - Recall: {100 * metrics_dict0['recall']:.2f} | Precision: {100 * metrics_dict0['precision']:.2f} | IoU: {100 * metrics_dict0['iou']:.2f}"
    )
    print(
        f"Class 1 (non-ground) - Recall: {100 * metrics_dict1['recall']:.2f} | Precision: {100 * metrics_dict1['precision']:.2f} | IoU: {100 * metrics_dict1['iou']:.2f}"
    )
    print(f"mIoU: {100 * miou:.2f}\n")
