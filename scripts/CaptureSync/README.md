# CaptureSync Pipeline

## Overview

CaptureSync is a Python script that organizes raw video frame folders from two sources (OIS and Non-OIS) into a structured dataset for machine learning workflows.

The script:

* Matches folders based on timestamp
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
* The timestamp must match between both folders

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

Notes:

* Each scene uses 2 devices (OIS and Non-OIS)
* Each scene is repeated across 3 methods

---

### 2. Validate Raw Files

* Check `.mcraw` files are complete
* Remove corrupted or unsupported videos
* Ensure both devices have matching recordings

---

### 3. Manual Export (MotionCam Pro)

For each video:

* Set render folder:

  ```text
  exports/{Method}/{capture_xxx}/
  ```

* Export settings:

  * Format: DNG
  * Turn OFF: bake in vignette
  * Turn ON: delete after render
  * Ensure ZIP compression is enabled

* Export each video manually

Notes:

* Manual pairing required due to filename differences

---

### 4. Upload via FolderSync

**OIS device:**

1. HandShake Method
2. Sliding Method
3. Vibration Method

**Non-OIS device:**

1. Vibration Method
2. HandShake Method
3. Sliding Method

Important:

* Do NOT sync both devices at the same time
* Prevents duplicate folders in Google Drive

---

## Automated Workflow (CaptureSync)

This is the improved workflow using the script.

### 1. Add Input Data

```text
encoded_frames/incoming/ois/
encoded_frames/incoming/nonois/
```

---

### 2. Activate Virtual Environment

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate
```

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate.bat
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
* Match OIS and Non-OIS pairs
* Assign method (handshake → sliding → vibration)
* Create `capture_xxx` folders
* Move folders into Google Drive structure
* Sync automatically via Google Drive

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
* After processing, `incoming/` will be empty
* Files remain inside Google Drive folders

---

## Error Handling

The script handles:

* Invalid folder names → `failed/`
* Missing pairs → `failed/`
* Duplicate timestamps → `failed/`

---

## Logs

```text
logs/run_YYYYMMDD_HHMMSS.txt
```

Contains:

* processed captures
* skipped entries
* error details

---

## Output Example

```text
Training Set/
  HandShake Method/
    capture_010/
      ├── ois/
      │    └── 260401_111211_VIDEO_25mm/
      │         └── file.zip
      └── nonois/
           └── 260401_111211_VIDEO_26mm/
                └── file.zip
```

---

## Best Practices

* Verify folder names before running
* Do not modify output folders manually
* Check `failed/` after each run
* Ensure Google Drive is syncing

---

## Troubleshooting

### Empty folders in Google Drive

* Ensure files are not being moved after creation
* Check local folders before sync

### No folders processed

* Check naming format
* Ensure both `ois` and `nonois` folders contain data

### Files not syncing

* Ensure Google Drive Desktop is running
* Check internet connection

---

## Notes

This pipeline reduces manual workload by automating dataset organization and ensuring consistent structure.
It follows a local-first approach where Google Drive handles synchronization automatically.
