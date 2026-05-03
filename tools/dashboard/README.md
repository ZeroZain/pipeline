# Scene Dashboard

## Overview
The Scene Dashboard is a Streamlit-based visual inspection and data cleaning workstation for the motion deblurring preprocessing pipeline. It provides tools for reviewing scene quality, managing dataset categorization, and exporting the final structured dataset.

## Thesis Alignment: Data Cleaning & Multi-Stage Validation
This dashboard directly implements the "Data Cleaning" and "Multi-Stage Validation Framework" (Tables 13 & 14) described in the methodology:
* **Quality Scorecard:** Enforces geometric alignment (Flow ROI p90), photometric alignment, and color alignment (Delta E, SSIM) thresholds.
* **Image Pair Cleaning:** Serves as the UI to "Redundancy control" and "Pair locking" (Assigning Keep/Reject and Dataset Splits).
* **Manual Spot Check:** Provides a "Manual Frame Selection" fallback for when automated Laplacian scoring fails due to complex motion.

## Setup

### Dependencies
```bash
pip install streamlit pandas numpy opencv-python rawpy matplotlib altair
```

### Running the Dashboard
From the project root:
```bash
streamlit run tools/dashboard/app.py
```
The dashboard reads from `workspace/logs/` and `workspace/data/` relative to the project root.

---

## Features

### 1. Scene Inspector
The primary workspace for reviewing individual scenes. For each scene:
* **Quality Scorecard** — Flow ROI p90, Min SSIM, Delta E, and composite quality score
* **Laplacian Deltas** — Sharp-to-blur sharpness drop for OIS and Non-OIS
* **Alignment Confidence** — Average inlier ratio from geometric alignment
* **Scene Review Panel** — Mark scenes as KEEP, REJECT, or FLAG with optional notes
* **Manage Scene Data** — Assign dataset split (Training, Validation, Testing, Unassigned) and change capture method per scene
* **Laplacian Curves** — Interactive charts showing sharpness over time with selected frame markers
* **Selected Frames** — Visual display of sharp, drop, and blur frames from OIS and Non-OIS sequences
* **Manual Frame Selection** — Fallback UI to re-select valid sharp/blur pairs when objective extraction fails.
* **Scene Logs** — Geo, Photo, Color, Interpolation, Failure, and Review History tabs

### 2. Data Management (Scene Set Builder)
Global workspace management tools to curate collections of KEEP scenes:
* **Create & Load Sets** — Curate specific versions of your dataset (e.g., experiment_v1).
* **Bulk Actions** — Include, Exclude, or Flag multiple scenes at once.
* **Batch Split Assignment** — Assign dataset splits (Training, Validation, Testing) to multiple scenes simultaneously.

### 3. Export Dataset
Export reviewed scenes into the final training-ready folder structure:
* Only exports scenes marked as **KEEP** with an assigned split (Training, Validation, or Testing)
* Scenes with **Unassigned** split are skipped
* Exports to `dataset_ois/` and `dataset_nonois/` with `train/val/test` subdirectories
* Generates a manifest log at `logs/dataset_export_log.csv`

---

## Dataset Split Assignment
Dataset splits are **not** determined by folder structure. The folder hierarchy is organized by capture method only. Splits are assigned through the dashboard (Per-scene or Batch) and applied dynamically via overrides.

| Split | Export Key | Description |
| --- | --- | --- |
| Unassigned | *(not exported)* | Default for all new scenes |
| Training | `train` | Training set |
| Validation | `val` | Validation set |
| Testing | `test` | Test set |

---

## Folder Structure
The dashboard reads from the workspace (`dataset_1080`, `aligned`, `decoded_frames`) and logs (`geo_log.csv`, `laplacian/*.csv`, etc.) to generate the interactive UI.

## Export Output Structure
```text
workspace/
  data/
    dataset_ois/
      train/
        input/    ← ois_blur.jpg 
        target/   ← ois_sharp.jpg 
      val/
      test/
    dataset_nonois/
      train/
        input/    ← nonois_blur.jpg
        target/   ← nonois_sharp.jpg
      val/
      test/
```

## Scoring Criteria
* **Laplacian Score** — Variance of the Laplacian for sharpness measurement
* **Flow ROI p90** — Residual misalignment after geometric alignment (threshold: 15.0)
* **SSIM** — Structural similarity after color alignment (threshold: 0.70)
* **Delta E** — CIELAB color difference (no hard threshold)
* **Quality Score** — Composite ranking: `0.5 × flow + 0.3 × (1 − ssim) × 100 + 0.2 × deltaE`
