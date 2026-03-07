# Quick Start

Follow these steps to generate dataset scenes from decoded MotionCam frames.

### 1. Install Dependencies

The Laplacian dataset builder requires the following Python packages:

```bash
pip install opencv-python numpy rawpy
```

These packages are used for:

* OpenCV → image processing
* NumPy → numerical operations
* rawpy → decoding RAW `.dng` frames

---

### 2. Prepare decoded frames

Place extracted frames inside the `decoded_frames/` folder using the following structure:

```text
decoded_frames/
├── capture_001/
│   ├── ois/
│   │   ├── frame_000001.dng
│   │   └── ...
│   │
│   └── nonois/
│       ├── frame_000001.dng
│       └── ...
```

Each `capture_xxx` folder must contain both `ois` and `nonois` frames from the same recording.

Frame numbering must remain in chronological order.

---

### 3. Run the dataset builder

From the **project root folder**, run:

```bash
python scripts/laplacian/scene_builder.py
```

---

### 4. Dataset will be created automatically

The script will generate dataset scenes:

```text
dataset/
├── scene_001/
├── scene_002/
```

Each scene contains:

```text
ois_sharp.dng
ois_blur.dng
nonois_sharp.dng
nonois_blur.dng
```

---

### 5. Logs are saved automatically

```text
logs/
├── laplacian/
└── scene_selection_log.csv
```

These logs record:

* Laplacian sharpness scores per frame
* which frames were selected for each scene

---

### 6. The script can be run multiple times

The pipeline keeps track of processed captures using:

```text
dataset/dataset_state.json
```

This ensures that scene numbering continues automatically:

```text
scene_001
scene_002
scene_003
...
```

Previously processed captures will not be processed again.

---

### 7. Next step

After dataset generation, run the alignment pipeline:

```bash
python scripts/alignment/alignment_pipeline.py
```

---

# Dataset Building Pipeline (Laplacian Frame Selection)

This module builds the dataset used for the alignment and motion-deblurring pipeline.

It analyzes decoded video frames, computes Laplacian sharpness scores, detects blur segments, and constructs dataset scenes automatically.

The output dataset is compatible with the alignment pipeline used in the next stage.

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
RAW decoding (.dng → RGB)
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
│   │   │   ├── frame_000001.dng
│   │   │   ├── frame_000002.dng
│   │   │   └── ...
│   │   │
│   │   └── nonois/
│   │       ├── frame_000001.dng
│   │       ├── frame_000002.dng
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
frame_000001.dng
frame_000002.dng
frame_000003.dng
```

---

# 3. Output Dataset Structure

After running the script, the dataset will be generated automatically:

```text
dataset/
├── scene_001/
│   ├── ois_sharp.dng
│   ├── ois_blur.dng
│   ├── nonois_sharp.dng
│   └── nonois_blur.dng
│
├── scene_002/
│   ├── ois_sharp.dng
│   ├── ois_blur.dng
│   ├── nonois_sharp.dng
│   └── nonois_blur.dng
```

Each scene contains four images:

| File               | Description                    |
| ------------------ | ------------------------------ |
| `ois_sharp.dng`    | sharp frame from OIS video     |
| `ois_blur.dng`     | blur frame from OIS video      |
| `nonois_sharp.dng` | sharp frame from non-OIS video |
| `nonois_blur.dng`  | blur frame from non-OIS video  |

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
frame_000001.dng,120
frame_000002.dng,135
frame_000003.dng,410
frame_000004.dng,395
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
scene_001,capture_001,frame_000003.dng,frame_000006.dng,frame_000003.dng,frame_000006.dng
scene_002,capture_001,frame_000003.dng,frame_000015.dng,frame_000003.dng,frame_000015.dng
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