# Quick Start

Follow these steps to generate dataset scenes from decoded MotionCam frames.

### 1. Prepare decoded frames

Place extracted frames inside the `decoded_frames/` folder using the following structure:

```text
decoded_frames/
├── capture_001/
│   ├── ois/
│   │   ├── frame_000001.png
│   │   └── ...
│   │
│   └── nonois/
│       ├── frame_000001.png
│       └── ...
```

Each `capture_xxx` folder must contain both `ois` and `nonois` frames from the same recording.

---

### 2. Run the dataset builder

From the **project root folder**, run:

```bash
python scripts/laplacian/scene_builder.py
```

---

### 3. Dataset will be created automatically

The script will generate:

```text
dataset/
├── scene_001/
├── scene_002/
```

Each scene contains:

```text
ois_sharp.jpg
ois_blur.jpg
nonois_sharp.jpg
nonois_blur.jpg
```

---

### 4. Logs are saved automatically

```text
logs/
├── laplacian/
└── scene_selection_log.csv
```

These logs record:

* Laplacian sharpness scores per frame
* which frames were selected for each scene

---

### 5. The script can be run multiple times

The pipeline keeps track of processed captures using:

```text
dataset/dataset_state.json
```

So new scenes will continue numbering automatically:

```text
scene_001
scene_002
scene_003
...
```

---

### 6. Next step

After dataset generation, run the alignment pipeline:

```bash
python scripts/alignment/alignment_pipeline.py
```







# Dataset Building Pipeline (Laplacian Frame Selection)

This module builds the dataset used for the alignment and deblurring pipeline.
It analyzes decoded video frames, computes Laplacian sharpness scores, detects blur segments, and constructs scene folders.

The output dataset will match the format required by the alignment pipeline.

---

# 1. Pipeline Overview

The dataset generation pipeline works as follows:

```text
MotionCam video
      ↓
MotionCam decoder
      ↓
decoded_frames/
      ↓
Laplacian scoring
      ↓
sharp frame + blur frame detection
      ↓
dataset/scene_x folders
```

Each blur event in a video becomes one dataset scene.

---

# 2. Required Folder Structure

Before running the script, decoded frames must be organized as follows:

```text
project_root/
│
├── decoded_frames/
│   ├── capture_001/
│   │   ├── ois/
│   │   │   ├── frame_000001.png
│   │   │   ├── frame_000002.png
│   │   │   └── ...
│   │   │
│   │   └── nonois/
│   │       ├── frame_000001.png
│   │       ├── frame_000002.png
│   │       └── ...
│   │
│   ├── capture_002/
│   │   ├── ois/
│   │   └── nonois/
│   │
│   └── ...
```

Important notes:

* Each `capture_xxx` folder represents one **video pair**.
* Each capture must contain **two folders**:

  * `ois`
  * `nonois`
* Frame filenames must remain **in chronological order**.

Example:

```text
frame_000001.png
frame_000002.png
frame_000003.png
```

---

# 3. Output Dataset Structure

After running the script, the dataset will be generated automatically:

```text
dataset/
├── scene_001/
│   ├── ois_sharp.jpg
│   ├── ois_blur.jpg
│   ├── nonois_sharp.jpg
│   └── nonois_blur.jpg
│
├── scene_002/
│   ├── ois_sharp.jpg
│   ├── ois_blur.jpg
│   ├── nonois_sharp.jpg
│   └── nonois_blur.jpg
```

Each scene contains four images:

| File               | Description                    |
| ------------------ | ------------------------------ |
| `ois_sharp.jpg`    | sharp frame from OIS video     |
| `ois_blur.jpg`     | blur frame from OIS video      |
| `nonois_sharp.jpg` | sharp frame from non-OIS video |
| `nonois_blur.jpg`  | blur frame from non-OIS video  |

One blur moment produces **one scene**.

---

# 4. Logs Generated

The script automatically creates logs for traceability.

```text
logs/
├── laplacian/
│   ├── capture_001_ois.csv
│   ├── capture_001_nonois.csv
│   └── ...
│
└── scene_selection_log.csv
```

---

## Laplacian Score Log

Example:

```text
logs/laplacian/capture_001_ois.csv
```

Content:

```text
frame,score
frame_000001.png,120
frame_000002.png,135
frame_000003.png,410
frame_000004.png,395
```

This records the **sharpness score for every frame**.

---

## Scene Selection Log

Example:

```text
logs/scene_selection_log.csv
```

Content:

```text
scene,capture,ois_sharp,ois_blur,nonois_sharp,nonois_blur
scene_001,capture_001,frame_000030.png,frame_000100.png,frame_000028.png,frame_000108.png
scene_002,capture_001,frame_000030.png,frame_000180.png,frame_000028.png,frame_000188.png
```

This allows tracing each dataset image back to its original frame.

---

# 5. Dataset State File

The script maintains a state file:

```text
dataset/dataset_state.json
```

Example:

```json
{
  "next_scene_id": 11,
  "processed_captures": [
    "capture_001",
    "capture_002"
  ]
}
```

This allows the script to:

* continue scene numbering
* skip already processed captures
* resume safely if the script stops

---

# 6. Running the Script

Run the dataset builder from the **project root**:

```bash
python scripts/laplacian/scene_builder.py
```

The script will:

1. read decoded frames
2. compute Laplacian scores
3. detect blur segments
4. create scene folders
5. log selected frames

---

# 7. Running the Script Multiple Times

The script can be run multiple times safely.

Example workflow:

First run:

```text
capture_001
capture_002
```

Generated scenes:

```text
scene_001
scene_002
scene_003
scene_004
```

Later, new captures are added:

```text
capture_003
capture_004
```

Running the script again will continue numbering:

```text
scene_005
scene_006
scene_007
scene_008
```

Existing scenes will not be overwritten.

---

# 8. Blur Detection Logic

The script detects blur events using Laplacian sharpness scores.

Typical capture pattern:

```text
steady → shake → steady → shake
```

Detected segments:

```text
sharp segment
blur segment 1
sharp segment
blur segment 2
```

Each blur segment generates one scene.

If only one blur event exists, only one scene is created.

---

# 9. Important Notes

1. Do not modify filenames inside `decoded_frames/`.
2. Ensure frames are sequentially numbered.
3. OIS and Non-OIS captures must correspond to the same scene.
4. Do not manually edit `dataset_state.json`.

---

# 10. Next Pipeline Step

After the dataset is generated, run the alignment pipeline:

```bash
python scripts/alignment/alignment_pipeline.py
```

This will produce the aligned dataset used for benchmarking.