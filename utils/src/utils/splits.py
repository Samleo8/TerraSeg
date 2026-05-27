train_mini_splits = {
    "KITTI360": [
        "train_scans",
        "test_scans",
    ],
    "nuScenes": "train_scans",
    "SemanticKITTI": "train_scans",
    "WaymoPerception": "train_scans__TOP",
}

train_splits_no_vod = {
    "AevaScenes": "train_scans",
    "AV2_Lidar": [
        "train_scans",
        "val_scans",
        "test_scans",
    ],
    "KITTI360": [
        "train_scans",
        "test_scans",
    ],
    "Lyft": [
        "train_scans__part1",
        "train_scans__part2",
        "test_scans",
    ],
    "nuScenes": "train_scans",
    "ONCE": "train_scans",
    "PandaSet": [
        "train_scans",
        "test_scans",
    ],
    "SemanticKITTI": "train_scans",
    "TruckScenes": "train_scans",
    "WaymoPerception": [
        "train_scans__TOP",
        "train_scans__OTHERS",
    ],
    "ZOD": "train_scans",
}

train_splits = {
    "AevaScenes": "train_scans",
    "AV2_Lidar": [
        "train_scans",
        "val_scans",
        "test_scans",
    ],
    "KITTI360": [
        "train_scans",
        "test_scans",
    ],
    "Lyft": [
        "train_scans__part1",
        "train_scans__part2",
        "test_scans",
    ],
    "nuScenes": "train_scans",
    "ONCE": "train_scans",
    "PandaSet": [
        "train_scans",
        "test_scans",
    ],
    "SemanticKITTI": "train_scans",
    "TruckScenes": "train_scans",
    "VoD": [
        "train_scans",
        "val_scans",
        "test_scans",
    ],
    "WaymoPerception": [
        "train_scans__TOP",
        "train_scans__OTHERS",
    ],
    "ZOD": "train_scans",
}

eval_splits = {
    "nuScenes": "val_scans",
    "SemanticKITTI": "val_scans",
    "WaymoPerception": "val_scans",
}
