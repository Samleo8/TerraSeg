# PseudoLabeler Scripts


This repository contains the standalone Python scripts for the **PseudoLabeler** module. PseudoLabeler is a self-supervised method that generates high-quality ground and non-ground labels through per-scan runtime optimization. It is designed to label raw LiDAR scans in the [OmniLiDAR](../OmniLiDAR_scripts/) dataset without requiring any human-annotated training data.

To ensure absolute reproducibility, this project strictly utilizes [uv](https://github.com/astral-sh/uv) as its package manager.


## 🧠 What does PseudoLabeler do?


PseudoLabeler executes a three-stage pipeline to generate precise ground segmentation labels:

1. **Pre-processing (Denoising):** To prevent negative noise (like multi-path reflections) from artificially lowering the estimated elevation map, the point cloud is denoised by:
    * Computing a global **0.5% lower height quantile**.
    * Filtering out all points falling below this threshold.

2. **Runtime Optimization:** The module estimates a continuous Bird’s-Eye-View (BEV) elevation map using a Multilayer Perceptron (MLP) with SiLU activations:
    * **Loss Function:** An asymmetric loss that penalizes points below the predicted surface quadratically, while using a Huber penalty to ignore above-ground objects.
    * **Initial Labeling:** Points are initially classified based on a **0.40m distance threshold** from the predicted surface.

3. **Post-processing (Non-Ground recovery):** To fix over-segmentation (e.g. when the bottom of a car tire is mislabeled as ground), a pillar-based refinement step is applied:
    * The scan is discretized into **0.50m x 0.50m pillars**.
    * If a pillar contains both ground and non-ground points within a **+1.5m vertical window**, points exceeding the pillar's minimum elevation by more than **0.05m** are reclassified as non-ground.


## ⚙️ Step 1: Prerequisites & setup


1. **Install `uv`** (if you haven't already). For Windows/Mac instructions, see the official `uv` documentation.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Sync the Base Environment**:
Navigate to the root directory of the repository (e.g. `TerraSeg/`) and sync the base project environment. This reads the root uv.lock file to instantly build the shared workspace `.venv`.
```bash
cd PUT_YOUR_DIRECTORY_HERE/TerraSeg
uv sync
```


## 🛠️ Step 2: Configure the scripts


Before running the labeling process, edit `pseudolabeler_compute_labels.py` to point to your local OmniLiDAR dataset directory. Locate the results and OmniLiDAR root variables near the top of the file. Replace the placeholder text with your actual directory paths:

```python
results_root   = Path("PUT_YOUR_DIRECTORY_HERE/PseudoLabeler")
omnilidar_root = Path("PUT_YOUR_DIRECTORY_HERE/OmniLiDAR")
```

*Note 1: Ensure you leave the `.mkdir()` commands intact so the script can build the proper subdirectories.*
*Note 2: The `OmniLiDAR` folder is created using the code in the `OmniLiDAR_scripts` folder.*


## 🚀 Step 3: Run PseudoLabeler


Navigate into the scripts folder, and use `uv run` to execute the pseudo-labeling. Do **not** use the standard `python script.py` command. `uv run` will automatically detect the workspace's virtual environment (which includes the linked `terraseg-utils`) and execute the script safely.

```bash
cd PseudoLabeler_scripts
uv run pseudolabeler_compute_labels.py
```

*Note 1: Because this method performs runtime optimization, it is computationally intensive. Labels are generated offline at a rate of approximately **0.3 Hz**.*


## 📂 Repository Structure


**Core Scripts:**
* `pseudolabeler_compute_labels.py`: The primary execution script.
* `pseudolabeler_model.py`: Defines the MLP architecture of PseudoLabeler.
* `pseudolabeler_loss.py`: Implements the asymmetric Huber loss function.


**Ablation Studies:**
* `pseudolabeler_compute_ablation_study1__*.py`: Scripts to evaluate the impact of pre-processing and post-processing steps individually.
* `pseudolabeler_compute_ablation_study2.py`: Analyzes sensitivity regarding pillar sizes (`vxy`) and vertical recovery thresholds (`tau`).
