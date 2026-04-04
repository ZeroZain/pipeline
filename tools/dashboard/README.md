# Scene Dashboard

## Overview

The Scene Dashboard is a Streamlit-based visual inspection and data cleaning workstation for the motion deblurring preprocessing pipeline. It provides tools for reviewing scene quality, managing dataset categorization, and exporting the final structured dataset.

---

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
* **Scene Logs** — Geo, Photo, Color, Interpolation, Failure, and Review History tabs

### 2. Data Management & History

Global workspace management tools:

* **Global Review History** — View and clear all manual review decisions
* **Auto-Fix Methods** — Automatically reclassify HandShake ↔ Sliding based on linear motion detection
* **Batch Split Assignment** — Assign dataset splits to multiple scenes at once, filtered by method and/or current split
* **Manual Overrides History** — View and clear all split/method overrides

### 3. Export Dataset

Export reviewed scenes into the final training-ready folder structure:

* Only exports scenes marked as **KEEP** with an assigned split (Training, Validation, or Testing)
* Scenes with **Unassigned** split are skipped
* Exports to `dataset_ois/` and `dataset_nonois/` with `train/val/test` subdirectories
* Generates a manifest log at `logs/dataset_export_log.csv`

---

## Dataset Split Assignment

Dataset splits are **not** determined by folder structure. The folder hierarchy is organized by capture method only:

```
Method/scene_xxx/
```

Splits are assigned through the dashboard:

* **Per-scene**: In the Scene Inspector → Manage Scene Data → "Assign to Split"
* **Batch**: In Data Management → Batch Split Assignment

Available splits:

| Split | Export Key | Description |
| --- | --- | --- |
| Unassigned | *(not exported)* | Default for all new scenes |
| Training | `train` | Training set |
| Validation | `val` | Validation set |
| Testing | `test` | Test set |

Split assignments are stored as overrides in `workspace/logs/scene_overrides.csv` and are applied dynamically — they never modify the original workspace files.

---

## Folder Structure

The dashboard reads from the following workspace paths:

```text
workspace/
  data/
    dataset_256/
      gt_ois/
        HandShake Method/
          scene_xxx/
            ois_sharp.jpg, ois_blur.jpg, nonois_sharp.jpg, nonois_blur.jpg
        Sliding Method/
          ...
        Vibration Method/
          ...
    aligned/
      gt_ois/
        color/
          HandShake Method/
            scene_xxx/
              ...
    decoded_frames/
      HandShake Method/
        capture_xxx/
          ois/
          nonois/
  logs/
    geo_log.csv
    photo_log.csv
    color_log.csv
    scene_fail_log.csv
    scene_selection_log.csv
    interpolation_log.csv
    manual_review.csv
    scene_overrides.csv
    dataset_export_log.csv
    laplacian/
      <capture_name>_ois.csv
      <capture_name>_nonois.csv
```

---

## Export Output Structure

```text
workspace/
  data/
    dataset_ois/
      train/
        input/    ← ois_blur.jpg (001.jpg, 002.jpg, ...)
        target/   ← ois_sharp.jpg (001.jpg, 002.jpg, ...)
      val/
      test/

    dataset_nonois/
      train/
        input/    ← nonois_blur.jpg
        target/   ← nonois_sharp.jpg
      val/
      test/
```

---

## Sidebar Filters

* **Split** — Filter by assigned dataset split
* **Method** — Filter by capture method (HandShake, Sliding, Vibration)
* **Find scene** — Text search across scene names
* **Sort scenes by** — Scene order, Worst quality, Highest flow, Lowest SSIM, Most failures
* **Review Status** — Filter by KEEP, REJECT, FLAG, NOT REVIEWED
* **Quality Status** — Filter by Clean, Needs Attention, Has Failures
* **Failed Phase** — Filter scenes that failed specific pipeline phases (geo, ssim, etc.)
* **Flow ROI p90 range** — Slider filter
* **Min SSIM range** — Slider filter

---

## Scoring Criteria

The dashboard includes a detailed expander in the sidebar ("How Scoring Works") explaining:

* **Laplacian Score** — Variance of the Laplacian for sharpness measurement
* **Flow ROI p90** — Residual misalignment after geometric alignment (threshold: 15.0)
* **SSIM** — Structural similarity after color alignment (threshold: 0.70)
* **Delta E** — CIELAB color difference (no hard threshold)
* **Quality Score** — Composite ranking: `0.5 × flow + 0.3 × (1 − ssim) × 100 + 0.2 × deltaE`

---

## Configuration

Key constants in `app.py`:

| Constant | Default | Description |
| --- | --- | --- |
| `FLOW_THRESHOLD` | 15.0 | Flow ROI p90 pass/fail threshold |
| `SSIM_THRESHOLD` | 0.70 | Minimum SSIM pass/fail threshold |
| `SPLIT_OPTIONS` | Unassigned, Training, Validation, Testing | Available split categories |

---

## Notes

* The dashboard caches loaded data using `@st.cache_data` for performance
* Review decisions append to `manual_review.csv` — multiple reviews per scene are allowed, with the latest being the active one
* Overrides are stored in `scene_overrides.csv` and apply dynamically without modifying source files
* The batch split assignment tool allows bulk categorization for efficient workflow
