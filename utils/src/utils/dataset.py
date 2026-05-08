import bisect
import hashlib
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import Dataset
from torchrobotics.ground import filter_by_radius_origin


class OmniLiDARDataset(Dataset):
    """
    The OmniLiDAR dataset: Many LiDAR datasets unified into a single format.

    Note:
        This dataset strictly assumes perfect file contiguity. Within any sequence,
        `scan_num` must start at 0 and increment sequentially with zero missing frames.

    Args:
        root (str or Path) : Root directory of OmniLiDAR dataset.
        splits (dict) : Dataset splits (e.g. {'nuScenes': 'train_scans',}).
        pseudolabels_root (str or Path) : Root directory containing pseudo-labels.
        remove_ego_points (bool) : Whether to remove ego points from the LiDAR scans.
        compute_scans_metadata (bool) : Whether to compute the scans metadata during initialization.
    """

    def __init__(
        self,
        root: str | Path,
        splits: dict,
        pseudolabels_root: str | Path,
        remove_ego_points: bool = False,
        compute_scans_metadata: bool = False,
    ):
        assert isinstance(splits, dict), (
            f"The 'splits' argument must be a dictionary, got {type(splits)}."
        )

        self.splits = []
        for dataset_name, split_data in splits.items():
            assert isinstance(dataset_name, str), (
                f"Dataset keys must be strings. Got: {type(dataset_name)}"
            )
            split_list = [split_data] if isinstance(split_data, str) else split_data
            assert isinstance(split_list, list), (
                f"Split values must be strings or lists of strings. Got: {type(split_data)}"
            )
            for split_name in split_list:
                assert isinstance(split_name, str), (
                    f"Split names must be strings. Got: {type(split_name)}"
                )
                self.splits.append(f"{dataset_name}/{split_name}")

        # General setup.
        self.root = Path(root)
        self.ego_removal_radii_root = self.root / "egoremovalradii"
        self.transform_root = self.root / "transforms"
        self.pseudolabels_root = Path(pseudolabels_root)
        self.remove_ego_points = remove_ego_points
        self.compute_scans_metadata = compute_scans_metadata

        # Get sequence sizes.
        self.get_scans_metadata()

        # Process sizes for fast indexing.
        self.pairs = [
            (split_name, sensor_name)
            for split_name in self.splits
            for sensor_name in self.scans_metadata[split_name]
        ]
        block_totals = [
            sum(self.scans_metadata[split_name][sensor_name].values())
            for split_name, sensor_name in self.pairs
        ]
        self.cum_blocks = list(itertools.accumulate(block_totals))

        self.seq_info = {
            (split_name, sensor_name): (
                list(seq_counts),
                list(itertools.accumulate(seq_counts.values())),
            )
            for split_name in self.scans_metadata
            for sensor_name, seq_counts in self.scans_metadata[split_name].items()
        }

        # Get ego removal radii.
        ego_removal_radii_files = [
            file_dir for file_dir in self.ego_removal_radii_root.glob("*_egoremovalradii.json")
        ]
        ego_removal_radii = {
            str(filename.name).split("_egoremovalradii.json")[0]: json.loads(
                (self.ego_removal_radii_root / filename).read_text()
            )
            for filename in ego_removal_radii_files
        }
        ego_removal_radii = {
            split_name: {sensor_name: float(radius) for sensor_name, radius in data.items()}
            for split_name, data in ego_removal_radii.items()
        }
        self.ego_removal_radii = ego_removal_radii

        # Get transforms.
        transform_files = list(self.transform_root.glob("*_transforms.json"))
        transforms = {
            str(filename.name).split("_transforms.json")[0]: json.loads(
                (self.transform_root / filename).read_text()
            )
            for filename in transform_files
        }
        transforms = {
            split_name: {
                sensor_name: torch.tensor(values, dtype=torch.float32).view(4, 4)
                for sensor_name, values in data.items()
            }
            for split_name, data in transforms.items()
        }
        self.transforms = transforms

    def get_scans_metadata(
        self,
    ):
        """
        Get number of scans for each sequence in the dataset.
        """
        metadata_dir = self.root / "scansmetadata"
        os.makedirs(metadata_dir, exist_ok=True)

        # Create a stable, order-independent hash of the splits.
        stable_splits_repr = json.dumps(sorted(self.splits)).encode("utf-8")
        split_hash = hashlib.md5(stable_splits_repr).hexdigest()[:12]

        metadata_path = metadata_dir / f"scans_metadata__{split_hash}.json"

        if self.compute_scans_metadata:
            scans_metadata = {}
            for split_name in self.splits:
                base_dir = self.root / split_name
                pc_dir = base_dir / "pointcloud"
                new_data = defaultdict(lambda: defaultdict(int))
                for p in pc_dir.glob("pointcloud__sensor_*__seq_num_*__scan_num_*.npy"):
                    _, sensor_s, seq_s, scan_s = p.stem.split("__")
                    sensor_name = sensor_s.split("sensor_")[1]
                    seq_num = int(seq_s.split("seq_num_")[1])
                    scan_num = int(scan_s.split("scan_num_")[1])
                    if scan_num + 1 > new_data[sensor_name][seq_num]:  # Note: scan_num starts at 0.
                        new_data[sensor_name][seq_num] = scan_num + 1
                new_data = {sensor_name: dict(seqs) for sensor_name, seqs in new_data.items()}
                scans_metadata[split_name] = new_data

            # Sort the sequences.
            self.scans_metadata = {
                split_name: {
                    sensor_name: dict(sorted(values.items()))
                    for sensor_name, values in sensors.items()
                }
                for split_name, sensors in scans_metadata.items()
            }

            # Make folder and save the metadata.
            metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(self.scans_metadata, f, indent=4)

        else:
            # Load if file exists.
            assert metadata_path.exists(), (
                f"Metadata not found at {metadata_path} for splits {self.splits}. "
                f"Run with compute_scans_metadata=True first."
            )

            with open(metadata_path, "r") as f:
                loaded_metadata = json.load(f)

            self.scans_metadata = {
                split_name: {
                    sensor_name: {int(seq_num): count for seq_num, count in seqs.items()}
                    for sensor_name, seqs in sensors.items()
                }
                for split_name, sensors in loaded_metadata.items()
            }

    def __len__(
        self,
    ) -> int:
        """
        Return number of scans.
        """
        return self.cum_blocks[-1] if self.cum_blocks else 0

    def __getitem__(
        self,
        idx: int,
    ) -> dict:
        """
        Get point cloud and labels (if exist).

        Args:
            idx (int) : Global scan index.

        Return:
            scan_dict (dict) : Dictionary containing point cloud, labels (if exist), map (if exist), and meta data.
        """
        assert idx >= 0, "Scan index must be non-negative!"
        assert idx < len(self), "Scan index is higher than total number of scans!"

        # Pick block
        b = bisect.bisect_right(a=self.cum_blocks, x=idx)
        split_name, sensor_name = self.pairs[b]
        rel_idx = idx - (self.cum_blocks[b - 1] if b > 0 else 0)

        # Pick sequence.
        _, cums = self.seq_info[(split_name, sensor_name)]
        seq_num = bisect.bisect_right(cums, rel_idx)
        scan_num = rel_idx - (cums[seq_num - 1] if seq_num > 0 else 0)

        # Get point cloud, labels (if exist), map (if exist), and pseudo-labels (if exist).
        pc_dir = self.get_pointcloud_dir(
            split_name=split_name, sensor_name=sensor_name, seq_num=seq_num, scan_num=scan_num
        )
        labels_dir = self.get_labels_dir(
            split_name=split_name, sensor_name=sensor_name, seq_num=seq_num, scan_num=scan_num
        )
        pseudolabels_dir = self.get_pseudolabels_dir(
            split_name=split_name, sensor_name=sensor_name, seq_num=seq_num, scan_num=scan_num
        )

        pc = torch.from_numpy(np.load(pc_dir, mmap_mode="r").copy()).float()
        original_length = pc.shape[0]

        try:
            labels = torch.from_numpy(np.load(labels_dir, mmap_mode="r").copy())
            target_labels_dir_out = labels_dir
        except FileNotFoundError:
            labels = None
            target_labels_dir_out = None
        try:
            pseudolabels = torch.from_numpy(np.load(pseudolabels_dir, mmap_mode="r").copy())
            pseudolabels_dir_out = pseudolabels_dir
        except FileNotFoundError:
            pseudolabels = None
            pseudolabels_dir_out = None

        # Meta data.
        source_dataset = split_name.split("/")[0]
        source_split = split_name.split("/")[1]

        # Remove ego points.
        radius = self.ego_removal_radii[source_dataset][sensor_name]
        if self.remove_ego_points:
            bool_valid = filter_by_radius_origin(pc=pc, radius=radius, mode="keep_outer")
            pc = pc[bool_valid]
            labels = labels[bool_valid] if labels is not None else None
            pseudolabels = pseudolabels[bool_valid] if pseudolabels is not None else None
            ids_valid = torch.nonzero(bool_valid, as_tuple=True)[0]
        else:
            ids_valid = torch.arange(original_length, device=pc.device)

        # Transform points.
        T_target_source = self.transforms[source_dataset][sensor_name]
        R = T_target_source[:3, :3]
        t = T_target_source[:3, 3]
        pc_transformed = (R @ pc.T).T + t

        # Create dict.
        scan_dict = {
            "x": pc_transformed,
            "target_labels": labels,  # 0: ground; 1: non-ground; 2: ignore.
            "target_pseudolabels": pseudolabels,  # 0: ground; 1: non-ground.
            "x_dir": pc_dir,
            "target_labels_dir": target_labels_dir_out,
            "pseudolabels_dir": pseudolabels_dir_out,
            "original_pc_length": original_length,
            "ids_valid": ids_valid,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "sensor_name": sensor_name,
            "seq_num": seq_num,
            "scan_num": scan_num,
            "global_idx": idx,
            "ego_removal_radius": radius,
        }
        return scan_dict

    def __repr__(
        self,
    ) -> str:
        """
        Display dataset statistics.

        Return:
            table (str) : Overview of the dataset.
        """
        headers = [
            "DATASET_NAME",
            "ORIGINAL_SPLIT_NAME",
            "SENSOR_NAME",
            "NUMBER OF SEQUENCES",
            "NUMBER OF POINT CLOUDS",
        ]
        rows = []
        for key, (scene_ids, cum_sweeps_count) in self.seq_info.items():
            dataset_name, split_name = key[0].split("/")
            sensor_name = key[1]
            rows.append(
                [
                    dataset_name,
                    split_name,
                    sensor_name,
                    f"{len(scene_ids):,}",
                    f"{cum_sweeps_count[-1]:,}",
                ]
            )
        total_sequences = sum(int(r[3].replace(",", "")) for r in rows)
        total_pointclouds = sum(int(r[4].replace(",", "")) for r in rows)
        rows.append(["TOTAL", "", "", f"{total_sequences:,}", f"{total_pointclouds:,}"])

        table = tabulate(
            rows, headers=headers, tablefmt="pretty", stralign="center", numalign="center"
        )
        lines = table.splitlines()
        sep_line = lines[2]
        insert_idx = len(lines) - 2
        lines.insert(insert_idx, sep_line)
        body = "\n".join(lines)
        return f"OmniLiDAR Dataset Overview\n\n{body}"

    def get_pointcloud_dir(
        self,
        split_name: str,
        sensor_name: str,
        seq_num: int,
        scan_num: int,
    ) -> Path:
        """
        Get point cloud directory.

        Args:
            split_name (str) : Split name.
            sensor_name (str) : Sensor name.
            seq_num (int) : Sequence number.
            scan_num (int) : scan number.

        Return:
            pc_dir (Path) : Point cloud directory.
        """
        assert seq_num >= 0, "Sequence number must be non-negative!"
        assert scan_num >= 0, "Scan number must be non-negative!"

        folder_dir = self.root / split_name / "pointcloud"
        filename = f"pointcloud__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy"
        return folder_dir / filename

    def get_labels_dir(
        self,
        split_name: str,
        sensor_name: str,
        seq_num: int,
        scan_num: int,
    ) -> Path:
        """
        Get labels directory.

        Args:
            split_name (str) : Split name.
            sensor_name (str) : Sensor name.
            seq_num (int) : Sequence number.
            scan_num (int) : scan number.

        Return:
            labels_dir (Path) : Labels directory.
        """
        assert seq_num >= 0, "Sequence number must be non-negative!"
        assert scan_num >= 0, "Scan number must be non-negative!"

        folder_dir = self.root / split_name / "labels"
        filename = f"labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy"
        return folder_dir / filename

    def get_pseudolabels_dir(
        self,
        split_name: str,
        sensor_name: str,
        seq_num: int,
        scan_num: int,
    ) -> Path:
        """
        Get pseudo-labels directory.

        Args:
            split_name (str) : Split name.
            sensor_name (str) : Sensor name.
            seq_num (int) : Sequence number.
            scan_num (int) : scan number.

        Return:
            pseudolabels_dir (Path) : Pseudo-labels directory.
        """
        assert seq_num >= 0, "Sequence number must be non-negative!"
        assert scan_num >= 0, "Scan number must be non-negative!"

        folder_dir = self.pseudolabels_root / split_name / "labels"
        filename = f"labels__sensor_{sensor_name}__seq_num_{str(seq_num).zfill(8)}__scan_num_{str(scan_num).zfill(8)}.npy"
        return folder_dir / filename
