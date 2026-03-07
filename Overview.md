# Motion Deblurring Dataset Generation Pipeline

This document describes the complete pipeline used to generate the dataset for motion deblurring benchmarking.
The pipeline converts MotionCam captures into structured scene folders that can be used directly by the alignment pipeline.

The system is designed to:

* process multiple video capture pairs
* detect sharp and blurred frames automatically
* build dataset scenes in the required format
* resume safely if the pipeline is stopped
* log all processing decisions for traceability

---

# 1. Full Project Structure

The project should follow this folder layout.

```
project_root/
│
├── captures/
│   ├── capture_001_ois.mcraw
│   ├── capture_001_nonois.mcraw
│   ├── capture_002_ois.mcraw
│   └── capture_002_nonois.mcraw
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
│
├── dataset/
│   ├── scene_001/
│   │   ├── ois_sharp.jpg
│   │   ├── ois_blur.jpg
│   │   ├── nonois_sharp.jpg
│   │   └── nonois_blur.jpg
│   │
│   ├── scene_002/
│   │   ├── ois_sharp.jpg
│   │   ├── ois_blur.jpg
│   │   ├── nonois_sharp.jpg
│   │   └── nonois_blur.jpg
│   │
│   └── dataset_state.json
│
├── logs/
│   ├── laplacian/
│   │   ├── capture_001_ois.csv
│   │   ├── capture_001_nonois.csv
│   │   └── ...
│   │
│   ├── scene_selection_log.csv
│   └── sharp_usage_log.csv
│
├── aligned/
│
└── scripts/
    ├── laplacian_scorer.py
    ├── scene_builder.py
    └── run_pipeline.py
```

---

# 2. Pipeline Stages

The full dataset pipeline follows this sequence:

```
MotionCam video
        ↓
MotionCam decoder
        ↓
decoded_frames/
        ↓
Laplacian scoring
        ↓
Frame selection
        ↓
Scene builder
        ↓
dataset/scene_x
        ↓
Alignment pipeline
        ↓
aligned/
```

---

# 3. Stage 1 — MotionCam Decoder

The MotionCam decoder extracts frames from each video capture.

Example output:

```
decoded_frames/capture_001/ois/frame_000001.png
decoded_frames/capture_001/nonois/frame_000001.png
```

Frame numbering represents temporal order and must not be modified.

---

# 4. Stage 2 — Laplacian Scoring

Each frame is evaluated using the Variance of Laplacian method to measure sharpness.

Example log file:

```
logs/laplacian/capture_001_ois.csv
```

Example contents:

```
frame,score
frame_000001.png,120
frame_000002.png,135
frame_000003.png,410
frame_000004.png,395
frame_000005.png,210
frame_000006.png,80
frame_000007.png,60
```

This log provides the sharpness timeline for the entire video.

---

# 5. Stage 3 — Frame Selection

From the Laplacian scores the script determines:

* the sharp frame (highest score during steady segment)
* one or more blur frames (low score during shaking segments)

Example detection:

```
sharp_frame = frame_000003
blur_frames = [frame_000006, frame_000015]
```

Each blur frame corresponds to a blur moment captured during the recording.

---

# 6. Stage 4 — Scene Builder

Each blur frame generates a new dataset scene.

Example output:

```
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

Scenes may reuse the same sharp frame but contain different blur frames.

---

# 7. Stage 5 — Dataset State File

A dataset state file allows the pipeline to resume safely.

Location:

```
dataset/dataset_state.json
```

Example:

```
{
  "next_scene_id": 3,
  "total_scenes": 2,
  "processed_captures": [
    "capture_001"
  ]
}
```

When the pipeline runs again it reads this file to determine the next scene number.

---

# 8. Stage 6 — Scene Selection Log

The scene selection log records which frames produced each dataset scene.

File:

```
logs/scene_selection_log.csv
```

Example:

```
scene,capture,ois_sharp,ois_blur,nonois_sharp,nonois_blur
scene_001,capture_001,frame_000003.png,frame_000006.png,frame_000002.png,frame_000007.png
scene_002,capture_001,frame_000003.png,frame_000015.png,frame_000002.png,frame_000017.png
```

This ensures every dataset image can be traced back to the original video frame.

---

# 9. Stage 7 — Alignment Pipeline

Once dataset scenes are generated, the alignment pipeline processes them.

Input:

```
dataset/
```

Output:

```
aligned/
├── gt_ois/
│   ├── geo/
│   ├── photo/
│   └── color/
```

The alignment pipeline performs:

1. Geometric alignment
2. Photometric normalization
3. Color normalization

---

# 10. Running the Pipeline Multiple Times

The system supports multiple executions without overwriting data.

Example first run:

```
capture_001
capture_002
```

Scenes created:

```
scene_001
scene_002
scene_003
scene_004
```

Later additional captures are decoded:

```
capture_003
capture_004
```

The pipeline reads the state file and continues numbering:

```
scene_005
scene_006
scene_007
scene_008
```

---

# 11. Key Advantages of This Design

This dataset pipeline provides:

* support for multiple capture sessions
* automatic scene numbering
* full traceability through logs
* safe restart capability
* compatibility with the alignment pipeline
* reproducible dataset generation

This structure ensures the dataset can be expanded incrementally while maintaining consistent organization.
