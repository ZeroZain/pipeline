# Overview

The pipeline processes decoded MotionCam frames and automatically constructs dataset scenes using Laplacian sharpness analysis.

Each scene contains:

* a sharp reference image
* a motion-blurred image
* captured from both OIS and Non-OIS devices

Scenes are generated automatically based on detected motion events within each capture.

After scene generation, the dataset undergoes a multi-stage preprocessing pipeline consisting of:

* geometric alignment
* photometric normalization
* color normalization
* bicubic interpolation to a fixed resolution of **256 × 256 pixels**

These steps ensure that all image pairs are spatially aligned, photometrically normalized, color balanced, and resolution standardized before benchmarking.

---

# Pipeline Workflow

The dataset is created using the following pipeline:

```
MotionCam Capture
        ↓
MotionCam Decoder
        ↓
decoded_frames/
        ↓
RAW decoding (.dng → RGB)
        ↓
Laplacian sharpness scoring
        ↓
Blur segment detection
        ↓
Scene generation
        ↓
dataset/scene_x
        ↓
Alignment pipeline
(geometric → photometric → color)
        ↓
aligned dataset
        ↓
Bicubic interpolation (256 × 256)
        ↓
final benchmark dataset
```

---

# Dataset Scene Format

Each generated scene initially contains four RAW images:

```
dataset/
└── scene_001/
    ├── ois_sharp.dng
    ├── ois_blur.dng
    ├── nonois_sharp.dng
    └── nonois_blur.dng
```

Where:

| File           | Description                        |
| -------------- | ---------------------------------- |
| `ois_sharp`    | sharp frame from OIS capture       |
| `ois_blur`     | blurred frame from OIS capture     |
| `nonois_sharp` | sharp frame from non-OIS capture   |
| `nonois_blur`  | blurred frame from non-OIS capture |

Multiple scenes may reuse the same sharp frame but contain different blur frames corresponding to different motion events.

After alignment and preprocessing, images are exported as standardized JPEG images.

---

# Key Features

The dataset generation pipeline provides:

* automatic sharp and blur frame detection
* support for RAW `.dng` frames extracted from MotionCam
* automatic scene generation from motion segments
* geometric, photometric, and color alignment
* bicubic interpolation for resolution standardization
* reproducible dataset construction
* full traceability through logging
* safe restart capability if processing stops

---

# Logging and Traceability

The pipeline records all processing decisions to ensure reproducibility.

Generated logs include:

```
logs/
├── laplacian/
│   ├── capture_001_ois.csv
│   └── capture_001_nonois.csv
│
├── scene_selection_log.csv
├── sharp_usage_log.csv
├── geo_log.csv
├── photo_log.csv
├── color_log.csv
└── interpolation_log.csv
```

These logs record:

* Laplacian sharpness scores per frame
* which frames were selected for each scene
* which scenes reuse the same sharp image
* geometric alignment validation results
* photometric normalization validation results
* color normalization validation results
* interpolation processing results

This logging system enables full traceability and reproducibility of the dataset construction process.

---

# Dataset Continuation

The system supports incremental dataset expansion.

A state file keeps track of processed captures and the next scene ID:

```
dataset/dataset_state.json
```

This allows the pipeline to be executed multiple times without overwriting existing scenes.

---

# Alignment Pipeline

After dataset scenes are generated, they are processed by the alignment pipeline which performs:

1. geometric alignment
2. photometric normalization
3. color normalization

These steps ensure that paired images are spatially aligned and visually consistent.

---

# Interpolation Stage

Following alignment, all images are standardized to a fixed resolution using bicubic interpolation.

The interpolation stage performs:

1. center cropping to obtain a square region
2. bicubic interpolation resizing to **256 × 256 pixels**

This ensures consistent spatial dimensions across the dataset and reduces computational cost for model training.

The resulting dataset becomes the **final benchmark dataset** used for motion-deblurring experiments.