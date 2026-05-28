# TerraSeg (library)


This workspace member contains the **library code** for TerraSeg: the model definition, BatchNorm-to-GroupNorm swap, feature engineering, and a single-frame predictor.

It is depended on by both the training scripts (`TerraSeg_scripts/`) and the ROS2 node (`TerraSeg_ros2/`), so the same TerraSeg model and inference pipeline is reused everywhere.


## Public API


```python
from terraseg import (
    TerraSeg,                   # nn.Module
    TERRASEG_B_CONFIG,          # dict
    TERRASEG_S_CONFIG,          # dict
    build_terraseg,             # Variant -> TerraSeg
    replace_bn1d_with_gn,       # See paper section 3.4
    compute_terraseg_features,  # (N,3) coord -> (N,3) feat
    TerraSegPredictor,          # Load checkpoint, predict single-frame
)
```


## When to use which


* For **training and offline evaluation**, see [TerraSeg_scripts](../TerraSeg_scripts/).
* For **online deployment on a robot**, see [TerraSeg_ros2](../TerraSeg_ros2) (subscribes to a LiDAR `PointCloud2` topic and publishes per-point labels).
* For **embedding TerraSeg into a custom Python application**, use the `TerraSegPredictor` class from this library directly.
