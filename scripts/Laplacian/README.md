# Dataset Building Pipeline (Laplacian Frame Selection)

This module is responsible for the first stage of the preprocessing pipeline. It analyzes decoded video frames, computes **Laplacian sharpness scores**, detects motion blur segments, and automatically constructs synchronized dataset scenes.

The output is a paired dataset of OIS and Non-OIS frames ready for the **Alignment Pipeline**.

---

# 1. Environment Setup

It is recommended to use a virtual environment (`.venv`) to avoid library conflicts.

### Create and Activate Environment
```bash
# Create
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies
```bash
pip install opencv-python numpy rawpy tqdm
```

---

# 2. Pipeline Overview

The script performs a "temporal seek" to find matching moments in OIS and Non-OIS footage:

```text
MotionCam decoded_frames/  (Delivered by Watcher)
          ↓
RAW decoding (.dng → RGB)
          ↓
Laplacian Scoring (Sharpness detection)
          ↓
Cross-Camera Synchronization
          ↓
Scene Construction (Automatic Folder creation)
          ↓
dataset/Method/scene_xxx/
```

---

# 3. Running the Script

The script supports three distinct modes depending on your workflow.

### Mode 1: One-Shot Processing
Processes everything currently in the `decoded_frames` folder and exits.
```bash
python scripts/Laplacian/1lap.py
```

### Mode 2: Full Pipeline (One-Shot)
Automatically proceeds to Alignment (`align4.py`) and Interpolation (`n256.py`) after processing.
```bash
python scripts/Laplacian/1lap.py --mode full
```

### Mode 3: Continuous Watcher (Recommended)
Stays open and automatically processes new captures as they arrive from the Drive Watcher.
```bash
python scripts/Laplacian/1lap.py --watch
```

---

# 4. Features & Logic

### Laplacian Sharpness Scoring
Uses the **Variance of the Laplacian** to identify:
* **Reference Frames**: High variance (Steady/Sharp).
* **Target Frames**: Low variance (Motion-Blurred).

### Dynamic Progress Tracking
In **Watcher Mode (`--watch`)**, the script maintains a **Persistent Master Bar**.
* **Automatic Scaling**: If the Drive Watcher adds new captures, the "Total" count on the progress bar will increase in real-time (e.g., `5/250` becomes `5/260`) without needing to restart the script.
* **Resumption**: Uses `dataset_state.json` to ensure it never processes the same capture twice, even after a crash or restart.

---

# 5. Automated Workflow (Drive to Dataset)

For a powerful, "hands-off" experience, run these two scripts in separate terminals:

1. **Terminal A (Drive Sync)**: `python scripts/Watcher/watcher.py`
   * Monitors Google Drive → Downloads, extracts, and cleans files into `decoded_frames`.
2. **Terminal B (Scene Selection)**: `python scripts/Laplacian/1lap.py --watch`
   * Monitors `decoded_frames` → Automatically produces final `dataset` scenes.

---

# 6. Folder Structures

### Input (`decoded_frames/`)
Folders are organized by capture method, containing `ois` and `nonois` subfolders.
```text
workspace/data/decoded_frames/
  HandShake Method/
    capture_001/
      ois/
      nonois/
```

### Output (`dataset/`)
Generated scenes are saved with standardized naming.
```text
workspace/data/dataset/
  HandShake Method/
    scene_001/
      ois_sharp.dng
      ois_blur.dng
      nonois_sharp.dng
      nonois_blur.dng
```

---

# 7. Documentation Logs

* **`logs/laplacian/`**: Contains raw sharpness scores for every frame.
* **`logs/scene_selection_log.csv`**: Maps every dataset scene back to the original source frames for research traceability.