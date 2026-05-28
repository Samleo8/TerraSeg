# TerraSeg ROS2 node


This package wraps **TerraSeg** as a real-time ROS2 node. It subscribes to a
`sensor_msgs/PointCloud2` LiDAR topic, runs the trained TerraSeg-B or TerraSeg-S model, and
publishes a labeled `PointCloud2` with an added `uint8 label` field (0 = ground, 1 = non-ground).

The ROS2 node reuses the same model definition and inference pipeline as the training scripts,
via the shared `terraseg` workspace library.


## 🧠 Topic API


* **Input:** `<input_topic>` of type `sensor_msgs/PointCloud2` (default `/lidar/points`).
  The point cloud must be in the TerraSeg-standardized frame: `z = 0` approximately
  ground-aligned and the positive `x`-axis pointing in the robot's forward direction.

* **Output:** `<output_topic>` of type `sensor_msgs/PointCloud2` (default `/terraseg/segmented`).
  Carries fields `x`, `y`, `z` (float32, meters) and `label` (uint8, with `0 = ground`,
  `1 = non-ground`). Downstream nodes can filter by the `label` field to isolate the ground
  or non-ground subset for free-space estimation, object discovery, etc.


## ⚙️ Step 1: Prerequisites


* A working ROS2 installation (tested with Humble and Jazzy). It can be installed natively
  on the host or provided via a Singularity / Docker container. Step 2 covers both cases.
* The TerraSeg workspace synced with `uv sync` (provides `torch`, `ptv3`, `huggingface_hub`,
  and the shared `terraseg` library).
* The PTv3 backbone vendored under `ptv3/src/ptv3/` (see [ptv3/README.md](../ptv3/README.md)).
* A trained TerraSeg checkpoint, either:
  * a local `best.pth` produced by `TerraSeg_scripts/terraseg_train.py`, or
  * the released weights on Hugging Face: `hf://TedLentsch/TerraSeg/terraseg_s.pth` or
    `hf://TedLentsch/TerraSeg/terraseg_b.pth`. These are downloaded automatically on the
    first launch and cached locally.


## 🛠️ Step 2: Build


The node is built with `colcon` from the workspace root. The exact recipe depends on whether
ROS2 is installed natively on the host or provided via a container. Both are common; pick
the one that matches your setup.

### Native ROS2

When ROS2 is installed on the host (typically under `/opt/ros/${ROS_DISTRO}/`):

```bash
cd PUT_YOUR_DIRECTORY_HERE/TerraSeg
source /opt/ros/${ROS_DISTRO}/setup.bash
source .venv/bin/activate
colcon build --packages-select terraseg_ros2 --symlink-install
source install/setup.bash
```

The `--symlink-install` flag lets you edit the Python sources in place without rebuilding.

### Containerized ROS2 (Singularity, Docker, etc.)

When ROS2 lives inside a container (common on shared clusters where you cannot
`apt install ros-${ROS_DISTRO}-*`), `colcon` is a tool in the *container's* system Python.
It generates the node's entry-point script with a shebang pointing at the container's
`/usr/bin/python3`, not at the uv venv, so the launched node fails at startup with
`ModuleNotFoundError: No module named 'terraseg'`.

The fix is to patch the shebang to the venv's Python after each `colcon build`. The
`build.sh` script below does the full rebuild in the correct order and applies the patch:

We ship `build.sh` at the workspace root. **Source it, do not execute it.** Sourcing runs
the script in your current shell so the `source /opt/ros/…/setup.bash`,
`source .venv/bin/activate`, and `source install/setup.bash` calls actually stick:

```bash
source build.sh
ros2 launch terraseg_ros2 terraseg.launch.py
```

If you accidentally run it with `./build.sh` instead, the build itself succeeds but the
sourced environment is lost when the subshell exits. `ros2` will report `command not found`
afterwards. The script detects this and aborts with an explanatory error.

The patched shebang makes the node's entry-point invoke the venv's Python, which sees both
the `terraseg` library (from the venv's `site-packages`) and the ROS2 message types (from
`PYTHONPATH`, set by sourcing the container's ROS2 setup).

> **Note on `set -eo pipefail` (no `-u`).** ROS2's `setup.bash` references variables like
> `AMENT_TRACE_SETUP_FILES` before they exist; under `set -u` (treat unset variables as
> errors) this aborts the script. The remaining flags (`-e`, `-o pipefail`) still catch real
> failures.


## 🛠️ Step 3: Configure


Edit `config/terraseg.yaml`. The most relevant fields are `variant` (`"S"` or `"B"`),
`checkpoint_path` (either a local path or an `hf://` URI), `compile_model` (default `true`),
and the input / output topics.


## 🚀 Step 4: Run


Launch the node:

```bash
ros2 launch terraseg_ros2 terraseg.launch.py
```

Or, to override the default config file at launch time:

```bash
ros2 launch terraseg_ros2 terraseg.launch.py \
    config:=/absolute/path/to/your/terraseg.yaml
```

Bag replay during development:

```bash
ros2 bag play your_lidar.bag --topics /lidar/points
```

Visualize in RViz: add a `PointCloud2` display on `/terraseg/segmented` and colour by the
`label` field.


## ⚡ Realtime and `torch.compile`


The node runs entirely in Python on top of PyTorch in **FP32**. Lower-precision dtypes
(BF16, FP16) are not supported because PTv3's sparse-convolution path becomes numerically
unstable at reduced precision. There is no precision knob.

The single throughput lever is `compile_model`:

| Default | Behaviour |
| --- | --- |
| `compile_model: true` | Wraps the model with `torch.compile(mode="reduce-overhead")` for kernel fusion and CUDA Graph capture. Adds a one-time compile cost at the first scan (a few seconds); subsequent scans are markedly faster. |

Per the TerraSeg paper (Tables 3-5), on an NVIDIA A100 GPU:

* TerraSeg-S: 17 - 50 Hz across the three benchmark datasets.
* TerraSeg-B: 10 - 28 Hz across the three benchmark datasets.

The Small variant comfortably keeps up with typical 10 Hz LiDAR streams; the Base variant is
borderline at 10 Hz on a desktop GPU. Throughput on embedded GPUs (e.g. Jetson Orin) is
significantly lower; keeping `compile_model: true` is strongly recommended in that case.


## ❓ Why not TensorRT or a C++ rewrite?


We deliberately keep the ROS2 node in Python:

* **PTv3's compute graph is unfriendly to TensorRT.** It relies on sparse convolutions
  (spconv) and optionally FlashAttention. Neither has a stock ONNX / TensorRT translation,
  so a TRT port would require writing and maintaining custom plugins.
* **Python is not the bottleneck.** Single-stream LiDAR processing is GPU-bound; the
  Python interpreter and the GIL are inactive while CUDA kernels run. A C++ rewrite with
  `libtorch` would reduce per-call CPU overhead by a few hundred microseconds, which is
  invisible at 10 - 50 Hz scan rates.
* **`torch.compile` captures most of the available speedup** through kernel fusion and
  CUDA-graph reuse, without the maintenance cost of a TRT or C++ port.


## 📂 Repository Structure


* `package.xml`: ROS2 package manifest.
* `setup.py` / `setup.cfg`: ament_python build configuration.
* `terraseg_ros2/terraseg_node.py`: the node implementation (subscriber + publisher).
* `terraseg_ros2/pointcloud_conversion.py`: PointCloud2 ↔ torch.Tensor utilities.
* `launch/terraseg.launch.py`: default launch file.
* `config/terraseg.yaml`: default node parameters.
