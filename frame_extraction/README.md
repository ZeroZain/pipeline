# Automated Frame Extraction Pipeline

*(OIS vs NON-OIS Motion Blur Dataset Construction)*

This guide explains how to properly prepare your folders, name your videos, install requirements, and run the automated frame extraction script.

---

# 1. Overview

This pipeline will:

* Automatically scan OIS and NON_OIS video folders
* Match video pairs using numeric filenames
* Extract frames from each video
* Compute Laplacian sharpness
* Classify frames as **sharp** or **blur**
* Save frames into a structured dataset

The process supports scalable and reproducible dataset construction for motion deblurring benchmarking.

---

# 2. Prerequisites (DO FIRST)

## 2.1 Install Python

Make sure Python **3.9–3.11** is installed.

Check installation:

```
python --version
```

---

## 2.2 Install Required Libraries

Open terminal / command prompt and run:

```
pip install opencv-python numpy
```

These are the only required dependencies.

---

# 3. Required Input Folder Structure

You must prepare your video dataset exactly like this:

```
dataset_videos/
├── OIS/
│   ├── 1.mp4
│   ├── 2.mp4
│   ├── 3.mp4
│   └── ...
│
└── NON_OIS/
    ├── 1.mp4
    ├── 2.mp4
    ├── 3.mp4
    └── ...
```

---

## IMPORTANT Naming Rules

### Rule 1 — Numeric filenames only

Videos must be named using matching numbers:

* `1.mp4` pairs with `1.mp4`
* `2.mp4` pairs with `2.mp4`
* etc.

✅ Correct:

```
OIS/1.mp4
NON_OIS/1.mp4
```

❌ Incorrect:

```
OIS/video1.mp4
NON_OIS/ois_1.mp4
```

---

### Rule 2 — Both folders must contain the pair

The system processes **only matching pairs**.

Example:

If you have:

```
OIS:      1.mp4, 2.mp4, 3.mp4
NON_OIS:  1.mp4, 3.mp4
```

The pipeline will process:

* Pair 1
* Pair 3

Pair 2 will be skipped automatically.

---

### Rule 3 — Supported video formats

Supported extensions:

* `.mp4`
* `.mov`
* `.avi`

---

# 4. Place the Script

Put your Python script in the **same parent directory** as `dataset_videos`.

Example:

```
project_folder/
├── dataset_videos/
├── frame_extractor.py
```

---

# 5. Configure the Script (If Needed)

Inside the script you will see:

```
VIDEO_ROOT = "dataset_videos"
OUTPUT_ROOT = "dataset_frames"
```

Only change these if your folder names are different.

---

## Optional Parameter Tuning

You may adjust:

```
FRAME_INTERVAL = 1
SHARPNESS_THRESHOLD = 100
JPG_QUALITY = 95
```

### Recommended starting values

* FRAME_INTERVAL → 1
* SHARPNESS_THRESHOLD → 100
* JPG_QUALITY → 95

---

# 6. Run the Script

Open terminal inside your project folder and run:

```
python frame_extractor.py
```

---

# 7. Expected Output Structure

After running, the script will automatically create:

```
dataset_frames/
├── OIS/
│   ├── 1/
│   │   ├── sharp/
│   │   └── blur/
│   ├── 2/
│   └── ...
│
└── NON_OIS/
    ├── 1/
    │   ├── sharp/
    │   └── blur/
    ├── 2/
    └── ...
```

---

## Frame Filename Format

Each extracted frame will look like:

```
pair1_frame_000123_t4.100_lap132.5.jpg
```

Meaning:

* pair1 → video pair ID
* frame_000123 → frame index
* t4.100 → timestamp in seconds
* lap132.5 → Laplacian sharpness score

This improves traceability and reproducibility.

---

# 8. Verification Checklist

Before proceeding to alignment, verify:

* [ ] Frames exist in both OIS and NON_OIS
* [ ] Sharp and blur folders are populated
* [ ] Filenames contain timestamp and Laplacian
* [ ] Frame counts are reasonable
* [ ] No unexpected empty folders

---

# 9. Troubleshooting

## No pairs found

Check:

* Folder names are exactly `OIS` and `NON_OIS`
* Filenames match numerically
* Video extensions are supported

---

## FPS reported as zero

Some phones report bad metadata. The script already includes a safe fallback.

---

## Too many blurry frames

Adjust:

```
SHARPNESS_THRESHOLD
```

Try values:

* 80 (more sensitive)
* 120 (stricter)
* 150 (very strict)

---

# 10. Next Step in the Pipeline

After successful frame extraction, the next Phase 1 stage is:

**Geometric alignment → Photometric alignment → Color alignment**

Do not resize yet.

---

# 11. Recommended Citation Statement (Optional for Thesis)

You may describe the process as:

> An automated Python-based pipeline using OpenCV was developed to extract and classify frames from paired OIS and non-OIS videos based on Laplacian sharpness, enabling structured and reproducible dataset construction.

---

**End of README**
