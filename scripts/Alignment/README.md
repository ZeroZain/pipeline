---

# Motion Deblurring Preprocessing Pipeline

## Geometric → Photometric → Color Alignment

This project implements a 3-phase preprocessing pipeline for paired image alignment before motion deblurring benchmarking.

The pipeline runs sequentially:

```
Raw Image
   ↓
Geometric Alignment
   ↓
Photometric Alignment
   ↓
Color Alignment
   ↓
Final Clean Pair
```

Each stage:

* Has its own validation
* Saves outputs in separate folders
* Logs results independently

---

# 1. Project Folder Structure

Your project must look like this:

```
your_project/
│
├── pipeline_runner.py
│
├── dataset/
│   ├── scene_001/
│   │   ├── ois_sharp.jpg
│   │   ├── ois_blur.jpg
│   │   ├── nonois_sharp.jpg
│   │   ├── nonois_blur.jpg
│   │
│   ├── scene_002/
│   │   ├── ois_sharp.jpg
│   │   ├── ois_blur.jpg
│   │   ├── nonois_sharp.jpg
│   │   ├── nonois_blur.jpg
│   │
│   └── ...
│
└── aligned/        (auto-created)
```

Important rules:

1. Each scene must be inside its own folder.
2. Filenames must match exactly:

   * `ois_sharp.jpg`
   * `ois_blur.jpg`
   * `nonois_sharp.jpg`
   * `nonois_blur.jpg`
3. Do not rename files.
4. Do not mix scenes.

---

# 2. Install Required Libraries

Run this once:

```
pip install opencv-python numpy scikit-image
```

---

# 3. Selecting the Ground Truth

Open `pipeline_runner.py`.

Find this line:

```
GT_SOURCE = "ois"
```

You can set:

* `"ois"` → uses `ois_sharp.jpg` as reference
* `"nonois"` → uses `nonois_sharp.jpg` as reference

Only one can be active per run.

---

# 4. How to Run the Pipeline

From the project root directory:

```
python pipeline_runner.py
```

The script will automatically:

1. Loop through all scenes inside `dataset/`
2. Select the ground truth image
3. Perform geometric alignment
4. Perform photometric alignment
5. Perform color alignment
6. Save outputs
7. Log validation metrics

No manual scene selection is required.

---

# 5. Explanation of Each Stage

## 5.1 Geometric Alignment

Purpose:

* Align spatial position of image to the ground truth.

Method:

* ORB feature matching
* RANSAC affine transformation
* Optical flow residual check

Validation:

* Inlier ratio must meet threshold
* Residual optical flow must be below threshold

Output folder:

```
aligned/gt_ois/geo/scene_x/
```

---

## 5.2 Photometric Alignment

Purpose:

* Normalize brightness and contrast.

Method:

* Linear intensity scaling (gain and bias)

Validation:

* Mean intensity difference must decrease after correction

Output folder:

```
aligned/gt_ois/photo/scene_x/
```

This stage uses images from:

```
geo/
```

---

## 5.3 Color Alignment

Purpose:

* Normalize chromatic bias between images.

Method:

* LAB color distribution matching

Validation:

* Delta E (color distance in LAB space) must decrease

Output folder:

```
aligned/gt_ois/color/scene_x/
```

This is the final clean dataset.

This stage uses images from:

```
photo/
```

---

# 6. Output Folder Structure

After running with:

```
GT_SOURCE = "ois"
```

You will get:

```
aligned/
│
└── gt_ois/
    │
    ├── geo/
    │   ├── scene_001/
    │   └── scene_002/
    │
    ├── photo/
    │   ├── scene_001/
    │   └── scene_002/
    │
    └── color/
        ├── scene_001/
        └── scene_002/
```

If you switch to:

```
GT_SOURCE = "nonois"
```

You will also get:

```
aligned/gt_nonois/
```

Outputs are stored separately to prevent overwriting.

---

# 7. Logs Folder

The script automatically creates:

```
logs/
│
├── geo_log.csv
├── photo_log.csv
└── color_log.csv
```

---

## 7.1 geo_log.csv

Columns:

* scene
* image
* inlier_ratio
* mean_flow
* valid

---

## 7.2 photo_log.csv

Columns:

* scene
* image
* mean_before
* mean_after
* valid

---

## 7.3 color_log.csv

Columns:

* scene
* image
* deltaE_before
* deltaE_after
* valid

If validation fails:

* The image does not proceed to the next stage.
* It is recorded in the corresponding log.

---

# 8. Running Both Experimental Modes

To process both ground-truth options:

Step 1:

Set:

```
GT_SOURCE = "ois"
```

Run the script.

Step 2:

Set:

```
GT_SOURCE = "nonois"
```

Run the script again.

Both outputs will be preserved:

```
aligned/gt_ois/
aligned/gt_nonois/
```

---

# 9. Full Processing Flow

For each image:

```
Raw Image
   ↓
Geometric Alignment
   ↓
Photometric Alignment
   ↓
Color Alignment
   ↓
Final Output
```

Each stage:

* Uses the previous stage’s output
* Has independent validation
* Has independent logging

---

# 10. Important Notes

1. Do not interrupt the script midway.
2. Do not manually modify intermediate outputs.
3. Always check logs after running.
4. If many images fail validation, review threshold settings.

---

# 11. What This Pipeline Ensures

1. Spatial consistency
2. Brightness normalization
3. Color normalization
4. Independent validation per stage
5. Reproducible preprocessing
6. Clean dataset for transformer-based deblurring benchmarking

---