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
Decoded frame collection
        ->
Capture organization and extraction
        ->
Laplacian-based frame analysis
        ->
Scene construction
        ->
Alignment pipeline
        ->
Resolution standardization
        ->
Final benchmark dataset
```

The pipeline begins with paired capture folders from OIS and non-OIS devices. After the decoded frames are collected, the system identifies useful motion events, selects representative sharp and blurred frames, aligns the paired images, and standardizes them to a fixed resolution for benchmarking and model training.

---

# Directory Structure

To keep the repository root clean, generated files are grouped under a single `workspace/` parent folder.

The top-level project layout becomes:

```text
pipeline/
  scripts/
  workspace/
    data/
    logs/
    debug/
  Overview.md
```

Within `workspace/`, the pipeline preserves the dataset split and capture method throughout preprocessing.

The decoded capture collection is organized as:

```text
workspace/
  data/
    decoded_frames/
      Training Set/
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
      Validation Set/
        ...
```

After scene construction, the paired dataset is stored as:

```text
workspace/
  data/
    dataset/
      Training Set/
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
      Validation Set/
        ...
```

After alignment, the processed dataset becomes:

```text
workspace/
  data/
    aligned/
      gt_ois/
        color/
          Training Set/
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
          Validation Set/
            ...
```

After resolution standardization, the final benchmark dataset is stored as:

```text
workspace/
  data/
    dataset_256/
      gt_ois/
        Training Set/
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
        Validation Set/
          ...
```

---

# Dataset Scene Structure

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

---

# Alignment Pipeline

After scene construction, the dataset undergoes a three-stage alignment process:

1. geometric alignment
2. photometric normalization
3. color alignment

These stages ensure that paired images are spatially consistent and visually comparable before they are used for evaluation or model training.

The alignment stage reduces variation caused by viewpoint mismatch, brightness differences, and color inconsistency so that the remaining differences better reflect motion blur characteristics.

Aligned outputs are exported as JPEG images for the next stage of preprocessing.

---

# Resolution Standardization

After alignment, the dataset is standardized to a fixed spatial resolution.

This stage performs:

1. inward center cropping to avoid invalid border regions introduced by alignment
2. bicubic resizing to **256 x 256 pixels**

This ensures that all samples have uniform dimensions and are suitable for consistent benchmarking and learning-based experiments.

The resulting output becomes the final benchmark-ready dataset.

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
```

Generated logs cover:

* frame-level sharpness measurements
* selected scene contents
* geometric alignment validation
* photometric alignment validation
* color alignment validation
* interpolation results
* scene-level failures during preprocessing

This traceability is important for thesis documentation because each generated sample can be related back to its original capture and preprocessing outcome.

---

# Incremental Dataset Growth

The system is designed to support repeated runs as new captures become available.

A state file tracks previously processed captures and the next available scene ID, allowing the dataset to grow incrementally without rebuilding everything from the beginning.

This makes the workflow practical for long-running data collection across training, validation, and testing splits.

---

# Research Purpose

This pipeline was designed to produce a structured paired dataset for studying motion blur under stabilized and non-stabilized conditions.

By combining:

* automatic scene extraction
* paired OIS and non-OIS samples
* alignment and normalization
* fixed-resolution output

the dataset becomes suitable for fair benchmarking of motion deblurring methods in the thesis setting.
