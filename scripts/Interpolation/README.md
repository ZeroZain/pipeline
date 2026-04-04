# Processing Pipeline Position

This stage runs **after the alignment pipeline**.

Complete preprocessing flow:

```
Raw Image (.dng / .jpg / .png)
        ↓
RAW decoding (if .dng)
        ↓
Geometric Alignment (Homography/Affine)
        ↓
Photometric & Color Alignment (LAB Global Mean)
        ↓
Bicubic Interpolation & Inward Crop (256 × 256)
        ↓
Final Benchmark Dataset

```

This ensures that interpolation is applied only to **fully aligned and color-corrected image pairs**.

---

# 1. Project Folder Structure

The interpolation script expects the **output of the alignment pipeline**, organized by capture method:

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

    dataset_256/                 (auto-created)
      gt_ois/
        HandShake Method/
          scene_001/
            ...
        Sliding Method/
          ...
        Vibration Method/
          ...
```

---

# 2. Required Libraries

Install required libraries via terminal:

```bash
pip install opencv-python numpy tqdm

```

| Library | Purpose |
| --- | --- |
| OpenCV | Image loading, resizing, and saving |
| NumPy | Matrix operations |
| **tqdm** | **Visual progress bar tracking** |

---

# 3. Selecting Ground Truth Source

Inside `n256.py`, locate:

```python
GT_SOURCE = "ois"

```

| Value | Dataset Used |
| --- | --- |
| `"ois"` | reads from `aligned/gt_ois/color/` |
| `"nonois"` | reads from `aligned/gt_nonois/color/` |

---

# 4. Inward Center Crop (Black Artifact Removal)

Geometric alignment often causes black "void" areas at the edges due to image rotation and warping.

To ensure a clean dataset, the script performs an **Inward-Zoom Crop** before resizing.

* **Default Factor:** `0.8` (Takes the center 80% of the image).
* **Result:** Removes homography-induced border artifacts.

---

# 5. Smart Resume Logic (Skip Existing)

The script features **Resume Logic** to save time during large-scale processing:

1. It checks the `dataset_256` folder before processing a file.
2. If the processed `.jpg` already exists, it **skips** that image.
3. This allows you to stop and restart the script or add new scenes without re-processing old ones.

> **Note:** If you change the `crop_factor` or `TARGET_SIZE`, delete the `dataset_256` folder to force the script to re-generate the images.

---

# 6. Bicubic Interpolation

After cropping, images are resized to **256 × 256** using `cv2.INTER_CUBIC`.

**Advantages for Thesis Research:**

* Uses a **4 × 4 pixel neighborhood** for smoother gradients.
* Preserves high-frequency details better than Bilinear interpolation.
* Industry standard for Super-Resolution and Deblurring datasets (e.g., GOPRO, REDS).

---

# 7. Progress Monitoring

The script uses a dynamic progress bar (`tqdm`) to monitor the status:

* **Percentage:** Overall progress of all scenes.
* **Speed:** Scenes processed per second.
* **ETA:** Estimated time remaining until dataset completion.

---

# 8. Log File Management

Logs are saved to `workspace/logs/interpolation_log.csv`.

**Update Behavior:** The script now **appends** to the log. This ensures that if you process half of your dataset today and the other half tomorrow, the log file will contain the history of both sessions.

| Status | Meaning |
| --- | --- |
| SUCCESS | Image successfully cropped and resized. |
| SKIPPED_SMALL | Image resolution was smaller than 256px. |

---

# 9. Purpose of Interpolation Stage

The interpolation stage ensures:

1. **Border Removal:** Eliminates black pixels from alignment warping.
2. **Consistency:** Standard 256x256 resolution for **MLWNet/SwinIR** models.
3. **Efficiency:** Reduces memory usage during GPU training.
4. **Validity:** Ensures the AI learns from valid image content, not border artifacts.
