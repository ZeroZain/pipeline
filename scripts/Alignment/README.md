# Motion Deblurring Preprocessing Pipeline

## Geometric → Photometric → Color Alignment

This project implements a **3-phase preprocessing pipeline** for paired image alignment before motion deblurring benchmarking.

The pipeline runs sequentially:

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
Final Clean Pair (.jpg)
```

Each stage:

* has its own validation
* logs results independently
* uses the previous stage output

---

# 1. Project Folder Structure

The alignment script reads from the dataset folder which is organized by capture method:

```
workspace/
  data/
    dataset/
      HandShake Method/
        scene_001/
          ois_sharp.dng
          ois_blur.dng
          nonois_sharp.dng
          nonois_blur.dng
      Sliding Method/
        scene_002/
          ...
      Vibration Method/
        scene_003/
          ...
```

Important rules:

1. Each scene must be inside its own folder.
2. Filenames must match exactly:

```
ois_sharp
ois_blur
nonois_sharp
nonois_blur
```

File extensions may be:

```
.dng
.jpg
.png
```

3. Do not rename files.
4. Do not mix scenes.

---

# 2. Install Required Libraries

Run this once:

```
pip install opencv-python numpy scikit-image rawpy
```

Library purposes:

| Library      | Purpose              |
| ------------ | -------------------- |
| OpenCV       | image processing     |
| NumPy        | numerical operations |
| scikit-image | SSIM computation     |
| rawpy        | RAW `.dng` decoding  |

---

# 3. Selecting the Ground Truth

Open `align4.py`.

Find this line:

```
GT_SOURCE = "ois"
```

You can set:

```
"ois"
```

or

```
"nonois"
```

Options:

| Value      | Ground Truth                     |
| ---------- | -------------------------------- |
| `"ois"`    | uses `ois_sharp` as reference    |
| `"nonois"` | uses `nonois_sharp` as reference |

Only one can be active per run.

---

# 4. How to Run the Pipeline

From the project root directory:

```
python scripts/Alignment/align4.py
```

The script automatically:

1. scans all scenes inside `dataset/`
2. loads RAW images if necessary
3. selects the ground truth image
4. performs geometric alignment
5. performs photometric normalization
6. performs color normalization
7. computes validation metrics
8. saves aligned images

No manual scene selection is required.

To also run interpolation after alignment:

```
python scripts/Alignment/align4.py --run-next
```

---

# 5. Explanation of Each Stage

## 5.1 Geometric Alignment

Purpose:

Align spatial position of the image to the ground truth.

Method:

* SIFT feature detection
* feature matching
* RANSAC homography / partial affine / ECC affine
* residual optical flow validation
* chain alignment via non-OIS anchor

Validation checks:

* feature inlier ratio
* residual optical flow magnitude (ROI p90)

---

## 5.2 Photometric Alignment

Purpose:

Normalize brightness and contrast.

Method:

Linear intensity normalization:

```
channel-wise mean shift
```

Validation:

Mean intensity difference must decrease.

---

## 5.3 Color Alignment

Purpose:

Normalize chromatic differences between images.

Method:

LAB color distribution matching.

Validation:

Delta E (LAB color distance) must decrease.

Additional metrics computed:

* SSIM

Output folder:

```
aligned/gt_ois/color/
```

This becomes the **final output** of the alignment stage.

---

# 6. Final Output Folder Structure

After processing with:

```
GT_SOURCE = "ois"
```

The final dataset becomes:

```
workspace/
  data/
    aligned/
      gt_ois/
        color/
          HandShake Method/
            scene_001/
              ois_sharp.jpg
              ois_blur.jpg
              nonois_sharp.jpg
              nonois_blur.jpg
          Sliding Method/
            scene_002/
              ...
          Vibration Method/
            scene_003/
              ...
```

All images are saved as:

```
JPEG
```

---

# 7. Logs Folder

The script automatically creates:

```
workspace/
  logs/
    geo_log.csv
    photo_log.csv
    color_log.csv
    scene_fail_log.csv
```

---

## 7.1 geo_log.csv

Columns:

```
scene, image, inlier_ratio, mean_flow, flow_p90, flow_roi_p90, flow_metric, model, valid, fallback
```

---

## 7.2 photo_log.csv

Columns:

```
scene, image, mean_before, mean_after, valid
```

---

## 7.3 color_log.csv

Columns:

```
scene, image, deltaE_before, deltaE_after, ssim, used_color, overall_pass
```

---

## 7.4 scene_fail_log.csv

Columns:

```
scene, failed_images, failed_count, failed_phases
```

If validation fails:

* the failure is recorded in the corresponding log
* the scene is flagged for review in the dashboard

---

# 8. Running Both Experimental Modes

To test both ground truth strategies:

Step 1:

```
GT_SOURCE = "ois"
```

Run pipeline.

Step 2:

```
GT_SOURCE = "nonois"
```

Run pipeline again.

Results will be stored separately:

```
aligned/gt_ois/
aligned/gt_nonois/
```

This prevents overwriting.

---

# 9. Full Processing Flow

For each image pair:

```
Raw Image
   ↓
RAW decoding (.dng)
   ↓
Geometric Alignment
   ↓
Photometric Alignment
   ↓
Color Alignment
   ↓
Final Aligned Pair (.jpg)
```

Each stage:

* uses the previous stage output
* has independent validation
* has independent logging

---

# 10. Important Notes

1. Do not interrupt the script while processing scenes.
2. Do not manually modify intermediate outputs.
3. Always inspect logs after execution.
4. If many images fail validation, review threshold values.

---

# 11. What This Pipeline Ensures

This preprocessing pipeline guarantees:

1. spatial consistency
2. brightness normalization
3. color normalization
4. independent validation per stage
5. reproducible preprocessing
6. clean dataset for transformer-based motion deblurring benchmarking