# Overview

This pipeline supports the dataset preparation workflow for the thesis on motion deblurring using paired OIS and non-OIS captures from MotionCam.

Its purpose is to transform raw decoded capture sequences into a structured benchmark dataset where each scene contains:

* a sharp reference frame
* a motion-blurred frame
* samples from both OIS and non-OIS recordings

The overall objective is to produce image pairs that are suitable for controlled comparison, preprocessing, and downstream deblurring experiments.

---

# Pipeline Workflow (Aligned with Proposed Methodology)

The dataset is prepared through a highly structured post-processing pipeline designed to minimize inconsistencies arising from device-specific imaging characteristics while preserving motion blur as the primary source of degradation. 

The pipeline strictly follows the sequential stages defined in the thesis:

```text
1. Data Acquisition (Real-world Scenes & Dual Smartphone Setup)
        -> Handled manually + CaptureSync (pairing and organization)
        
2. Frame Extraction from OIS and Non-OIS Videos
        -> Handled by Watcher (download, extract, clean to RAW DNG)
        
3. Sharp and Blur Candidate Selection & Reference Frame Selection
        -> Handled by Laplacian (objective assessment, scene construction & pair locking)
        
4. Geometric Alignment
        -> Handled by Alignment pipeline (feature-based registration)
        
5. Photometric Alignment
        -> Handled by Alignment pipeline (intensity statistics matching)
        
6. Color Alignment
        -> Handled by Alignment pipeline (chromatic normalization)
        
7. Interpolation (Downsampling)
        -> Handled by Interpolation (Bicubic resizing to 512x512)
        
8. Data Cleaning & Multi-Stage Validation
        -> Handled by Dashboard (Quality scoring, split assignment, export)
```

The pipeline spans two machines. On the **capture device**, `CaptureSync` organizes the raw OIS/non-OIS captures. On the **processing machine**, the `Watcher` monitors and extracts the frames. From there, objective sharpness measures are applied and reference frames are selected (`Laplacian`), and the pairs are spatially and photometrically normalized (`Alignment`). Finally, images are resized (`Interpolation`), and the entire dataset is visually inspected and exported via the interactive `Dashboard`.

---

# Installation & Requirements

To run the pipeline, the following Python libraries must be installed. It is recommended to use a virtual environment:

```bash
pip install streamlit opencv-python pandas numpy rawpy scikit-image scipy matplotlib
```

---

# Script Details & Thesis Mapping

### 1. CaptureSync (Pre-Pipeline Organization)
CaptureSync is the entry point running on the capture device. It organizes raw video frame folders from two source devices (OIS and non-OIS) into a structured dataset on Google Drive.
* **Role:** Ensures data is paired temporally (within ±2 seconds tolerance) and classified by motion method (HandShake, Sliding, Vibration). 
* **How to Run:** `python scripts/CaptureSync/capture_sync.py`

### 2. Watcher (Frame Extraction)
* **Thesis Stage:** Frame Extraction from OIS and Non-OIS Videos
* **Role:** Monitors Google Drive, downloads captures, extracts ZIP archives, and removes all non-DNG files to produce the native RAW decoded frame collection. It preserves the exact frame rate and continuous temporal characteristics.
* **How to Run:** `python scripts/Watcher/watcher.py`

### 3. Laplacian (Candidate & Reference Selection)
* **Thesis Stage:** Sharp and Blur Candidate Selection & Reference Frame Selection
* **Role:** Evaluates objective sharpness across extracted frames using the Variance of Laplacian to ensure consistent and repeatable identification of high-frequency content. It then selects a single stable sharp reference frame and a corresponding motion-blurred frame for each scene, ensuring pairs are temporally adjacent before applying correction steps.
* **How to Run:** `python scripts/Laplacian/a2lap.py`

### 4. Alignment (Geometric, Photometric, Color)
* **Thesis Stage:** Geometric Alignment, Photometric Alignment, Color Alignment
* **Role:** A three-stage correction script. It first compensates for spatial misalignment (translation, rotation). Next, it matches intensity statistics to reduce brightness mismatch. Finally, it normalizes color distributions to prevent chromatic inconsistencies from influencing restoration metrics.
* **How to Run:** `python scripts/Alignment/align4.py`

### 5. Interpolation (Downsampling)
* **Thesis Stage:** Interpolation (Downsampling)
* **Role:** Applies inward center cropping to remove invalid borders and uses Bicubic interpolation to resize all images to exactly **512 × 512 pixels**. This standardizes image resolution across all paired samples.
* **How to Run:** `python scripts/Interpolation/interpolate.py`

### 6. Dashboard (Data Cleaning & Validation)
* **Thesis Stage:** Data Cleaning & Multi-Stage Validation
* **Role:** A Streamlit interactive workstation used to execute the final Image Pair Cleaning and Multi-Stage Validation framework. It provides:
  * **Objective Metrics:** Flow ROI p90 (geometric check), Min SSIM, Delta E.
  * **Manual Spot Check:** Fallback for algorithmic failure, allowing researchers to re-select frames or flag broken scenes.
  * **Dataset Structuring & Pair Locking:** Assigns scenes to Training/Validation/Testing splits and exports the finalized structured folders.
* **How to Run:** `streamlit run tools/dashboard/app.py`

---

# Directory Structure

To keep the repository root clean, generated files are grouped under a single `workspace/` parent folder.

```text
pipeline/
  scripts/
    CaptureSync/
    Watcher/
    Laplacian/
    Alignment/
    Interpolation/
  tools/
    dashboard/
  workspace/
    data/
    logs/
  Overview.md
```

---

# Logging and Traceability

The pipeline records processing decisions to support reproducibility and dataset auditing (aligning with the strict validation tables in the methodology).

```text
workspace/
  logs/
    laplacian/              <- Sharpness scoring validation
    scene_selection_log.csv <- Pair formation integrity
    geo_log.csv             <- Geometric spatial correspondence
    photo_log.csv           <- Intensity mapping validation
    color_log.csv           <- Chromatic similarity check
    manual_review.csv       <- Manual spot check & override logs
    dataset_export_log.csv  <- Final pair locking and dataset structuring
```

This traceability is essential for the thesis documentation because each generated sample can be traced back to its original capture and validation metric.
