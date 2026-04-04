# Overview

This pipeline supports the dataset preparation workflow for the thesis on motion deblurring using paired OIS and non-OIS captures from MotionCam.

Its purpose is to transform raw decoded capture sequences into a structured benchmark dataset where each scene contains:

* a sharp reference frame
* a motion-blurred frame
* samples from both OIS and non-OIS recordings

The overall objective is to produce image pairs that are suitable for controlled comparison, preprocessing, and downstream deblurring experiments.

---

# Pipeline Workflow

The dataset is prepared through the following stages:

```text
MotionCam capture
        ->
CaptureSync (pairing and organization → Google Drive)
        ->
Watcher (download, extract, and clean from Google Drive)
        ->
Decoded frame collection
        ->
Laplacian-based frame analysis
        ->
Scene construction
        ->
Alignment pipeline
        ->
Resolution standardization
        ->
Dashboard (review, split assignment, and export)
        ->
Final benchmark dataset
```

The pipeline spans two machines. On the **capture device**, CaptureSync matches and organizes the raw OIS/non-OIS captures into a structured Google Drive folder. On the **processing machine**, the Watcher monitors Google Drive for new captures, copies them locally, extracts the ZIP archives, and cleans out non-DNG files to produce the decoded frame collection. From there, the system identifies useful motion events through Laplacian sharpness analysis, selects representative sharp and blurred frames, aligns the paired images, and standardizes them to a fixed resolution. The Dashboard then provides visual inspection, quality review, dataset split assignment, and one-click export to produce the final benchmark dataset.

---

# CaptureSync

CaptureSync is the entry point of the pipeline. It is a Python script that organizes raw video frame folders from two source devices (OIS and non-OIS) into a structured dataset on Google Drive.

## Purpose

After each capture session, the OIS and non-OIS recordings land in separate incoming folders. CaptureSync automates the tedious process of:

* **Pairing** — matching OIS and non-OIS folders by timestamp (within ±2 seconds tolerance) to account for real-world capture delays between devices
* **Classifying** — assigning each pair to a capture method (HandShake, Sliding, or Vibration) via round-robin assignment
* **Organizing** — creating numbered `capture_xxx` folders with `ois/` and `nonois/` subfolders on Google Drive
* **Indexing** — scanning existing captures per method to determine the next available capture index, with configurable start indices to prevent overwrites

## How It Works

1. Raw captures are placed in `workspace/data/encoded_frames/incoming/ois/` and `incoming/nonois/`
2. The script validates folder names (format: `YYMMDD_HHMMSS_VIDEO_XXmm`)
3. Matched pairs are moved into the Google Drive structure under the assigned method
4. Unmatched or invalid folders are moved to `failed/` for manual review
5. All operations are logged to `logs/run_YYYYMMDD_HHMMSS.txt`

Files are **moved**, not copied — `incoming/` becomes empty after processing.

## Google Drive Output

```text
G:\My Drive\Thesis or Crisis\Videos\Dataset Capture\
  HandShake Method\
    capture_xxx\
      ois\
      nonois\
  Sliding Method\
    capture_xxx\
      ois\
      nonois\
  Vibration Method\
    capture_xxx\
      ois\
      nonois\
```

There are no dataset split folders on Google Drive. Split categorization is handled later in the Dashboard.

> See `scripts/CaptureSync/README.md` for full setup, usage, and troubleshooting details.

---

# Watcher

The Watcher is the bridge between the capture device and the processing machine. It is a continuously running Python script that monitors the Google Drive sync folder for new captures uploaded by CaptureSync, then downloads, extracts, and cleans them into local decoded frames ready for the rest of the pipeline.

## Purpose

While CaptureSync runs on the capture device to organize and upload raw data to Google Drive, the Watcher runs on the processing machine to:

* **Monitor** — scan the Google Drive Desktop sync folder every 15 seconds for new `capture_xxx` folders across all three methods
* **Validate** — verify each capture has both `ois/` and `nonois/` subfolders with at least one ZIP file each before processing
* **Wait for stability** — ensure ZIP file sizes stop changing (upload complete) before extraction, with a configurable timeout (default 300 seconds)
* **Extract** — copy captures to a local staging area, recursively extract all ZIP files, and flatten nested folders
* **Clean** — remove all non-DNG files, leaving only the raw sensor frames
* **Deliver** — move the cleaned capture into `workspace/data/decoded_frames/` preserving the method folder structure

## How It Works

1. The Watcher scans `G:\My Drive\Thesis or Crisis\Videos\Dataset Capture\` for `capture_xxx` folders under each method
2. It skips captures already present in `decoded_frames/` (duplicate protection)
3. Structurally ready captures are queued for processing by worker threads (2 workers by default)
4. Each capture is copied to `workspace/data/staging/` before extraction (safe staging)
5. After extraction and cleaning, the result is moved to `decoded_frames/`
6. If processing fails, the Watcher retries once; if it fails again, it remembers the failed source signature and skips until the source files change

## Console Status

After each scan, the Watcher prints a compact summary:

```text
[SCAN] found=24 ready=20 enqueued=2 busy=1 processed=18 incomplete=4 failed_hold=0 queued=2 active=1 failed=0
```

> See `scripts/Watcher/README.md` for full setup, reliability behavior, and troubleshooting details.

---

# Decoded Frame Collection

After the Watcher processes the raw captures from Google Drive, the extracted DNG frames are stored locally in the workspace:

```text
workspace/
  data/
    decoded_frames/
      HandShake Method/
        capture_xxx/
          ois/
          nonois/
      Sliding Method/
        capture_xxx/
          ois/
          nonois/
      Vibration Method/
        capture_xxx/
          ois/
          nonois/
```

---

# Scene Construction

Scene generation is based on Laplacian sharpness analysis applied to decoded frames.

In this stage, the pipeline:

* evaluates frame sharpness across each capture sequence
* detects transitions from sharp to blurred content
* selects representative sharp and blur frames for the motion event
* matches OIS and non-OIS samples from the same capture context
* stores the selected frames as reusable scene folders

This allows the dataset to be built from naturally occurring motion segments rather than from manual frame selection.

## Dataset Scene Structure

Each generated scene contains four corresponding images:

```text
scene_xxx/
  ois_sharp.dng
  ois_blur.dng
  nonois_sharp.dng
  nonois_blur.dng
```

Where:

| File | Description |
| --- | --- |
| `ois_sharp` | sharp frame from the OIS capture |
| `ois_blur` | blurred frame from the OIS capture |
| `nonois_sharp` | sharp frame from the non-OIS capture |
| `nonois_blur` | blurred frame from the non-OIS capture |

These scenes are designed to preserve correspondence between stabilization conditions while capturing the same motion event.

After scene construction, the paired dataset is stored as:

```text
workspace/
  data/
    dataset/
      HandShake Method/
        scene_xxx/
          ois_sharp.dng
          ois_blur.dng
          nonois_sharp.dng
          nonois_blur.dng
      Sliding Method/
        scene_xxx/
          ...
      Vibration Method/
        scene_xxx/
          ...
```

---

# Alignment Pipeline

After scene construction, the dataset undergoes a three-stage alignment process:

1. geometric alignment
2. photometric normalization
3. color alignment

These stages ensure that paired images are spatially consistent and visually comparable before they are used for evaluation or model training.

The alignment stage reduces variation caused by viewpoint mismatch, brightness differences, and color inconsistency so that the remaining differences better reflect motion blur characteristics.

Aligned outputs are exported as JPEG images:

```text
workspace/
  data/
    aligned/
      gt_ois/
        color/
          HandShake Method/
            scene_xxx/
              ois_sharp.jpg
              ois_blur.jpg
              nonois_sharp.jpg
              nonois_blur.jpg
          Sliding Method/
            scene_xxx/
              ...
          Vibration Method/
            scene_xxx/
              ...
```

---

# Resolution Standardization

After alignment, the dataset is standardized to a fixed spatial resolution.

This stage performs:

1. inward center cropping to avoid invalid border regions introduced by alignment
2. bicubic resizing to **256 x 256 pixels**

This ensures that all samples have uniform dimensions and are suitable for consistent benchmarking and learning-based experiments.

```text
workspace/
  data/
    dataset_256/
      gt_ois/
        HandShake Method/
          scene_xxx/
            ois_sharp.jpg
            ois_blur.jpg
            nonois_sharp.jpg
            nonois_blur.jpg
        Sliding Method/
          scene_xxx/
            ...
        Vibration Method/
          scene_xxx/
            ...
```

---

# Dashboard

The Dashboard is a Streamlit-based visual inspection and data cleaning workstation. It serves as the final quality gate before the dataset is used for training and evaluation.

## Purpose

The Dashboard brings together all pipeline outputs and logs into a single interactive interface, enabling:

* **Scene Inspection** — reviewing individual scenes with quality scorecards (Flow ROI p90, Min SSIM, Delta E, composite score), Laplacian sharpness curves, selected frame previews, and alignment logs
* **Quality Review** — marking scenes as KEEP, REJECT, or FLAG with optional notes, filtering by quality status, failed pipeline phases, and metric ranges
* **Dataset Split Assignment** — assigning scenes to Training, Validation, Testing, or Unassigned splits, either individually or in batch, without modifying the folder structure
* **Data Management** — viewing review and override history, auto-fixing method classifications, and batch operations
* **Dataset Export** — one-click export of KEEP scenes with assigned splits into the final `dataset_ois/` and `dataset_nonois/` folder structure with `train/val/test` subdirectories

## Dataset Split Assignment

Dataset split categorization (Training, Validation, Testing) is **not** determined by folder structure. Instead, the Dashboard provides tools to:

* assign individual scenes to a split (Training, Validation, Testing, or Unassigned)
* batch-assign scenes by method or current split status
* export only scenes with assigned splits into the final folder structure

This decouples the capture organization from the split assignment, allowing flexible re-categorization without moving files. Split assignments are stored as overrides in `workspace/logs/scene_overrides.csv` and are applied dynamically.

## Export Output

```text
workspace/
  data/
    dataset_ois/
      train/
        input/    <- ois_blur.jpg (001.jpg, 002.jpg, ...)
        target/   <- ois_sharp.jpg (001.jpg, 002.jpg, ...)
      val/
      test/

    dataset_nonois/
      train/
        input/    <- nonois_blur.jpg
        target/   <- nonois_sharp.jpg
      val/
      test/
```

> See `tools/dashboard/README.md` for full setup, features, scoring criteria, and configuration details.

---

# Directory Structure

To keep the repository root clean, generated files are grouped under a single `workspace/` parent folder.

The top-level project layout:

```text
pipeline/
  scripts/
    CaptureSync/
  tools/
    dashboard/
  workspace/
    data/
    logs/
    debug/
  Overview.md
```

Within `workspace/`, the pipeline organizes data by capture method. Dataset split categorization (Training, Validation, Testing) is handled in the Dashboard, not via folder structure.

---

# Logging and Traceability

The pipeline records processing decisions to support reproducibility and dataset auditing.

The logs are organized as:

```text
workspace/
  logs/
    laplacian/
      <capture_name>_ois.csv
      <capture_name>_nonois.csv
    scene_selection_log.csv
    geo_log.csv
    photo_log.csv
    color_log.csv
    scene_fail_log.csv
    interpolation_log.csv
    manual_review.csv
    scene_overrides.csv
    dataset_export_log.csv
```

Generated logs cover:

* frame-level sharpness measurements
* selected scene contents
* geometric alignment validation
* photometric alignment validation
* color alignment validation
* interpolation results
* scene-level failures during preprocessing
* manual review decisions (KEEP / REJECT / FLAG)
* dataset split and method overrides
* export manifests

This traceability is important for thesis documentation because each generated sample can be related back to its original capture and preprocessing outcome.

---

# Incremental Dataset Growth

The system is designed to support repeated runs as new captures become available.

A state file tracks previously processed captures and the next available scene ID, allowing the dataset to grow incrementally without rebuilding everything from the beginning.

This makes the workflow practical for long-running data collection across capture methods.

---

# Research Purpose

This pipeline was designed to produce a structured paired dataset for studying motion blur under stabilized and non-stabilized conditions.

By combining:

* automatic scene extraction
* paired OIS and non-OIS samples
* alignment and normalization
* fixed-resolution output

the dataset becomes suitable for fair benchmarking of motion deblurring methods in the thesis setting.
