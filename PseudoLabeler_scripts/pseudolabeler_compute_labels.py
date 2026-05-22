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
from utils.splits import train_splits

# Paths.
results_root = Path("PUT_YOUR_DIRECTORY_HERE/PseudoLabeler")
omnilidar_root = Path("PUT_YOUR_DIRECTORY_HERE/OmniLiDAR")

assert str(results_root) != "PUT_YOUR_DIRECTORY_HERE/PseudoLabeler", (
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
    splits=train_splits,
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
    if pc_lidar.shape[0] == 0:
        empty_pred_labels = np.array([], dtype=np.uint8)
        labels_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(labels_path, empty_pred_labels)
        continue

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
    if egoremoved_pc.shape[0] == 0:
        ignore_pred_labels = np.full((pc_lidar.shape[0],), fill_value=2, dtype=np.uint8)
        labels_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(labels_path, ignore_pred_labels)
        continue
    
    denoised_pc = model.preprocess_denoise_pc(pc=egoremoved_pc)
    pc = denoised_pc.clone()

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
        model.to(device)
    with torch.no_grad():
        model.eval()
        pc_lidar = sample_dict["x"].to(device)
        bool_ground = model.get_ground_bool(pc=pc_lidar)
        pred_labels = (~bool_ground).to(torch.uint8)
        postprocessed_pred_labels = model.postprocess_recover_non_ground(
            pc=pc_lidar, pred_labels=pred_labels
        )

    # Save pseudo-labels.
    labels_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(labels_path, postprocessed_pred_labels.cpu().numpy())
