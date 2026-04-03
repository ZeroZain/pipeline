# CaptureSync Pipeline

## Overview

CaptureSync is a Python script that organizes raw video frame folders from two sources (OIS and Non-OIS) into a structured dataset for machine learning workflows.

The script:

* Matches folders based on timestamp (with time tolerance)
* Groups them into `capture_xxx`
* Classifies them into:

  * handshake
  * sliding
  * vibration
* Saves output directly to a Google Drive synced folder
* Handles invalid data and logs all operations

---

## Folder Structure

Ensure the following structure exists before running the script:

```text
workspace/
  data/
    encoded_frames/
      ├── incoming/
      │    ├── ois/
      │    └── nonois/
      │
      ├── failed/
      └── logs/
```

Note:

* Files are moved directly from `incoming/` to Google Drive
* `failed/` stores unmatched or invalid folders

---

## Input Requirements

### Folder Naming Format

Each folder must follow this pattern:

```text
YYMMDD_HHMMSS_VIDEO_XXmm
```

### Examples

```text
260401_111211_VIDEO_25mm   (nonois)
260401_111211_VIDEO_26mm   (ois)
```

### Notes

* `25mm` corresponds to nonois
* `26mm` corresponds to ois
* Exact timestamp match is NOT required (see pairing logic below)

---

## Pairing Logic (Important)

The script uses **time-tolerant matching** instead of exact matching.

### How it works:

* Converts folder timestamps into datetime
* Matches OIS and Non-OIS folders based on **closest time**
* Accepts matches within:

```text
±2 seconds tolerance
```

### Example:

```text
OIS     → 260403_201606
NONOIS  → 260403_201607
```

Difference = 1 second → Valid pair

---

### Unmatched Cases

Folders are moved to `failed/` if:

* No match found within tolerance
* Invalid naming format
* Duplicate timestamps

---

## Google Drive Setup

This pipeline uses Google Drive Desktop. The script writes files locally, and Google Drive syncs them automatically.

### 1. Install Google Drive for Desktop

* Install Google Drive for Desktop
* Sign in with your Google account
* A new drive will appear:

```text
G:\My Drive\
```

---

### 2. Choose Sync Mode

Select:

* Mirror files (recommended)

Avoid:

* Stream files

---

### 3. Create Target Folder Structure

```text
My Drive/
  Thesis or Crisis/
    Videos/
      Dataset Capture/
        ├── Training Set/
        │     ├── HandShake Method/
        │     ├── Sliding Method/
        │     └── Vibration Method/
        │
        ├── Validation Set/
        │     ├── HandShake Method/
        │     ├── Sliding Method/
        │     └── Vibration Method/
        │
        └── Testing Set/
              ├── HandShake Method/
              ├── Sliding Method/
              └── Vibration Method/
```

---

### 4. Set Drive Path in Script

```python
DRIVE_BASE = r"G:\My Drive\Thesis or Crisis\Videos\Dataset Capture"
```

---

## Workflows

This pipeline supports both manual and automated workflows.

---

## Manual Workflow

This is the original process before automation.

### 1. Create Folder Structure

```text
HandShake Method/
  capture_001/
    ├── ois/
    └── nonois/

Sliding Method/
  capture_001/
    ├── ois/
    └── nonois/

Vibration Method/
  capture_001/
    ├── ois/
    └── nonois/
```

---

### 2. Validate Raw Files

* Check `.mcraw` files are complete
* Remove corrupted or unsupported videos
* Ensure both devices have matching recordings

---

### 3. Manual Export (MotionCam Pro)

* Export each video manually
* Match OIS and Non-OIS manually
* Save into correct capture folders

---

### 4. Upload via FolderSync

* Sync folders from both devices
* Do not sync simultaneously to avoid duplication

---

## Automated Workflow (CaptureSync)

### 1. Add Input Data

```text
encoded_frames/incoming/ois/
encoded_frames/incoming/nonois/
```

---

### 2. Activate Virtual Environment

```bash
.venv\Scripts\Activate
```

---

### 3. Run the Script

```bash
python scripts/CaptureSync/capture_sync.py
```

---

### 4. Select Dataset

```text
Dataset (training/validation/testing):
```

Mapped to:

* training → Training Set
* validation → Validation Set
* testing → Testing Set

---

### 5. Automatic Processing

The script will:

* Validate folder names
* Match pairs using time tolerance
* Assign method (handshake → sliding → vibration)
* Create `capture_xxx` folders
* Move folders into Google Drive structure
* Sync automatically

---

## Processing Logic

```text
{Drive}/{Dataset Set}/{Method Name}/capture_xxx/
```

Structure:

```text
capture_xxx/
  ├── ois/
  │    └── YYMMDD_HHMMSS_VIDEO_25mm/
  │         └── .zip
  └── nonois/
       └── YYMMDD_HHMMSS_VIDEO_26mm/
            └── .zip
```

---

## Important Behavior

* Files are **moved**, not copied
* `incoming/` becomes empty after processing
* Files remain in Google Drive folders

---

## Error Handling

The script handles:

* Invalid folder names → `failed/`
* No match within tolerance → `failed/`
* Duplicate timestamps → `failed/`

---

## Logs

```text
logs/run_YYYYMMDD_HHMMSS.txt
```

Contains:

* matched pairs
* skipped entries
* errors

---

## Output Example

```text
Training Set/
  HandShake Method/
    capture_010/
      ├── ois/
      │    └── 260403_201606_VIDEO_26mm/
      │         └── file.zip
      └── nonois/
           └── 260403_201607_VIDEO_25mm/
                └── file.zip
```

---

## Best Practices

* Verify folder names before running
* Do not modify output folders manually
* Check `failed/` regularly
* Ensure Google Drive is syncing

---

## Troubleshooting

### Too many files in `failed/`

* Increase tolerance (e.g., 3 seconds)

### Empty folders in Google Drive

* Ensure files are not moved after creation

### Files not syncing

* Ensure Google Drive Desktop is running
* Check internet connection

---

## Notes

This pipeline improves reliability by using time-tolerant pairing, reducing manual matching errors and handling real-world capture delays between devices.
