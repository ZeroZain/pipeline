# Processing Pipeline Position

This stage runs **after the alignment pipeline**.

Complete preprocessing flow:

```
Raw Image (.dng / .jpg / .png)
        ↓
RAW decoding (if .dng)
        ↓
Geometric Alignment
        ↓
Photometric Alignment
        ↓
Color Alignment
        ↓
Bicubic Interpolation (256 × 256)
        ↓
Final Benchmark Dataset
```

This ensures that interpolation is applied only to **fully aligned image pairs**.

---

# 1. Project Folder Structure

The interpolation script expects the **output of the alignment pipeline**.

Example project structure:

```
your_project/
│
├── pipeline_runner.py
├── interpolation_stage.py
│
├── aligned/
│   └── gt_ois/
│       └── color/
│           ├── scene_001/
│           │   ├── ois_sharp.jpg
│           │   ├── ois_blur.jpg
│           │   ├── nonois_sharp.jpg
│           │   └── nonois_blur.jpg
│           │
│           ├── scene_002/
│           │   ├── ois_sharp.jpg
│           │   ├── ois_blur.jpg
│           │   ├── nonois_sharp.jpg
│           │   └── nonois_blur.jpg
│
└── dataset_256/   (auto-created)
```

The script scans all scenes located inside:

```
aligned/gt_ois/color/
```

or

```
aligned/gt_nonois/color/
```

depending on the selected ground truth configuration.

---

# 2. Required Libraries

Install required libraries:

```
pip install opencv-python numpy
```

Library purposes:

| Library | Purpose                         |
| ------- | ------------------------------- |
| OpenCV  | image loading and interpolation |
| NumPy   | numerical operations            |

---

# 3. Selecting Ground Truth Source

Inside the script, locate:

```
GT_SOURCE = "ois"
```

Options:

```
"ois"
```

or

```
"nonois"
```

Meaning:

| Value      | Dataset Used                          |
| ---------- | ------------------------------------- |
| `"ois"`    | reads from `aligned/gt_ois/color/`    |
| `"nonois"` | reads from `aligned/gt_nonois/color/` |

The output dataset will be stored accordingly.

---

# 4. How to Run the Interpolation Stage

From the project root directory run:

```
python interpolation_stage.py
```

The script automatically:

1. scans all scenes in the aligned dataset
2. loads each image
3. center-crops images to square shape
4. resizes images to **256 × 256**
5. saves interpolated images
6. logs processing results

No manual scene selection is required.

---

# 5. Center Crop Procedure

Images may have different aspect ratios depending on the capture device.

Before resizing, each image is **center-cropped to a square region**.

Example:

```
4032 × 3024
      ↓
3024 × 3024 (center crop)
      ↓
256 × 256 resize
```

This step prevents geometric distortion that would occur if images were directly resized.

---

# 6. Bicubic Interpolation

After cropping, images are resized using **bicubic interpolation**.

Interpolation method:

```
cv2.INTER_CUBIC
```

Bicubic interpolation uses a **4 × 4 pixel neighborhood** to estimate new pixel values.

Advantages:

* smoother results compared to bilinear interpolation
* reduced aliasing artifacts
* better preservation of image structures

This makes it suitable for **vision datasets and deep learning preprocessing**.

---

# 7. Output Dataset Structure

After processing, the dataset is written to:

```
dataset_256/
```

Example:

```
dataset_256/
└── gt_ois/
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

All images are saved as:

```
256 × 256 resolution
JPEG (quality = 95)
```

The scene structure remains identical to the aligned dataset.

---

# 8. Log File

The script automatically generates:

```
logs/interpolation_log.csv
```

Columns:

```
scene
image
original_width
original_height
status
```

Example entry:

```
scene_001,ois_blur.jpg,4032,3024,SUCCESS
```

Possible status values:

| Status        | Meaning                           |
| ------------- | --------------------------------- |
| SUCCESS       | image successfully resized        |
| SKIPPED_SMALL | image resolution smaller than 256 |

---

# 9. Full Dataset Preparation Flow

The complete preprocessing workflow becomes:

```
Scene Generation (Laplacian Detection)
        ↓
Geometric Alignment
        ↓
Photometric Alignment
        ↓
Color Alignment
        ↓
Bicubic Interpolation (256 × 256)
        ↓
Final Benchmark Dataset
```

Each stage ensures that the dataset is:

* spatially aligned
* photometrically normalized
* color balanced
* resolution standardized

---

# 10. Purpose of Interpolation Stage

The interpolation stage ensures:

1. consistent input resolution for deep learning models
2. reduced computational cost during training
3. reproducible benchmarking conditions
4. standardized dataset dimensions
