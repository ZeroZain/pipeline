# Dataset Building Pipeline (Laplacian Frame Selection)

This module is responsible for the first stage of the preprocessing pipeline. It analyzes decoded video frames, computes **Laplacian sharpness scores**, detects motion blur segments, and automatically constructs synchronized dataset scenes.

The output is a paired dataset of OIS and Non-OIS frames ready for the **Alignment Pipeline**.

---

# 1. Environment Setup

To ensure all computer vision libraries are isolated and don't conflict with other projects, it is recommended to use a virtual environment (`.venv`).

### Create the Environment

Run this command in your project root:

```bash
python -m venv .venv

```

### Activate the Environment

You must activate the environment **every time** you start a new terminal session:

* **Windows (CMD):** `.venv\Scripts\activate`
* **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
* **macOS / Linux:** `source .venv/bin/activate`

### Install Dependencies

Once activated, install the required packages:

```bash
pip install opencv-python numpy rawpy tqdm

```

---

# 2. Pipeline Overview

The script performs a "temporal seek" to find matching moments in OIS and Non-OIS footage:

```text
MotionCam decoded_frames/
          ↓
RAW decoding (.dng → RGB)
          ↓
Laplacian Scoring (Variance detection)
          ↓
Cross-Camera Synchronization (Search Window)
          ↓
Scene Construction (Automatic Folder creation)
          ↓
dataset/scene_xxx/

```

---

# 3. Input Folder Structure

Place your decoded MotionCam frames in the following hierarchy:

```text
project_root/
├── decoded_frames/
│   ├── capture_001/
│   │   ├── ois/ (frame_000001.dng, ...)
│   │   └── nonois/ (frame_000001.dng, ...)

```

* **Synchronization:** Ensure the `ois` and `nonois` folders contain frames from the same recording event.

---

# 4. Logic & Features

### Laplacian Sharpness Scoring

The script uses the **Variance of the Laplacian** to determine image sharpness.

* **High Variance:** Steady, sharp frame (Reference).
* **Low Variance:** Motion-blurred frame (Target).

### Cross-Camera Search Window

Due to hardware timing differences, OIS and Non-OIS frames might be slightly desynchronized. The script uses a `SEARCH_WINDOW = 3` to look for the best matching sharp/blur frames in the neighboring indices of the secondary camera.

### Progress Tracking (`tqdm`)

Since decoding RAW files is CPU-intensive, the script provides real-time progress bars:

* **Overall Progress:** Tracks how many capture folders are finished.
* **Scoring Progress:** Tracks the frame-by-frame analysis of the current folder.

---

# 5. How to Run

Ensure your virtual environment is active, then run:

```bash
python scripts/laplacian/dataset_builder.py

```

### Resuming Progress

The script maintains a `dataset/dataset_state.json` file. If the process is interrupted, it will automatically skip already processed captures and continue numbering scenes from where it left off.

---

# 6. Output Summary

### Final Dataset

Each generated scene folder (e.g., `scene_001`) contains:

* `ois_sharp.dng` / `ois_blur.dng`
* `nonois_sharp.dng` / `nonois_blur.dng`

### Documentation Logs

* **`logs/laplacian/`**: Contains raw sharpness scores for every frame (useful for plotting shake profiles).
* **`logs/scene_selection_log.csv`**: Maps every dataset scene back to the original frame filename for research traceability.