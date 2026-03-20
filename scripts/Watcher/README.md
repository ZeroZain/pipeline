# Watcher Pipeline (Detection and Processing)

## 1. Overview

The Watcher Pipeline is responsible for automatically detecting and processing `capture_xxx` folders from Google Drive.

It serves as the entry point and processing engine of the system, handling the full workflow from detection to final output.

---

## 2. Functionality

The pipeline performs the following steps:

1. Monitors a Google Drive–synced folder on the local machine
2. Detects new and existing `capture_xxx` folders
3. Waits until the upload is fully complete
4. Copies data into a local staging directory
5. Extracts ZIP files from `ois` and `nonois` folders
6. Removes all non-`.dng` files (e.g., `.zip`, `.wav`, `.ini`)
7. Moves the processed data into the final output directory

---

## 3. Pipeline Flow

```text
Google Drive (upload capture_xxx)
        ↓
Local Sync Folder (G:)
        ↓
Detection (Watcher)
        ↓
Upload Validation (wait until complete)
        ↓
Copy → Staging (local)
        ↓
Extract and Clean Files
        ↓
Move → decoded_frames
```

---

## 4. Directory Structure

### 4.1 Watch Folder (Google Drive)

```text
G:\My Drive\Thesis or Crisis\Videos\automationInput
```

This folder must:

* Exist locally
* Be synced using Google Drive for Desktop
* Contain real folders (not shortcuts)

---

### 4.2 Staging Folder (Temporary Processing)

```text
C:\Users\Windows 11\Documents\GitHub\pipeline\staging
```

Used for:

* Safe local processing
* Temporary storage during extraction

Note:

* This folder will appear empty after processing because files are moved to the output directory

---

### 4.3 Output Folder

```text
C:\Users\Windows 11\Documents\GitHub\pipeline\decoded_frames
```

Contains the final processed dataset with only `.dng` files.

---

## 5. Setup and Execution

### 5.1 Open Terminal

```bash
cd pipeline
```

---

### 5.2 Activate Virtual Environment

```bash
.\.venv\Scripts\activate
```

---

### 5.3 Run the Watcher

```bash
python scripts/Watcher/watcher.py
```

---

## 6. Expected Behavior

### 6.1 On Startup

```text
[INIT] Checking existing folders...
Watching for new capture folders...
```

---

### 6.2 During Processing

```text
[FOUND EXISTING] capture_001
[NEW FOLDER DETECTED] capture_002
[WAITING] Ensuring full capture upload is complete...
[READY] Capture fully uploaded and stable
[COPYING]
[PROCESSING]
[EXTRACTING]
[CLEANING]
[DONE]
[COMPLETE] capture_xxx processed
```

---

## 7. Testing Procedure

1. Run the watcher script
2. Upload a new folder to Google Drive:

```text
capture_003/
   ├── ois/
   │     file.zip
   └── nonois/
         file.zip
```

3. Wait until processing completes

---

## 8. Expected Output

```text
decoded_frames/
   capture_003/
      ├── ois/
      │     *.dng
      └── nonois/
            *.dng
```

Only `.dng` files should remain in both subfolders.

---

## 9. Important Behavior

* Processing is sequential (one capture at a time)
* Both existing and newly added folders are supported
* Upload completion is validated before processing
* Staging ensures safe and consistent processing
* Non-essential files are automatically removed

---

## 10. Common Issues

### 10.1 Incomplete Output

Cause:

* Folder detected before upload finished

Solution:

* Ensure upload completes fully before processing begins
* The system now includes automatic waiting logic

---

### 10.2 Google Drive Issues

* Files must be fully synced locally
* Avoid using shortcuts
* Use real folders only

---

### 10.3 Module Not Found

Install required dependency:

```bash
pip install watchdog
```

---

## 11. Design Rationale

### 11.1 Use of Staging

```text
Drive → Staging → Processing → Output
```

Advantages:

* Prevents partial file processing
* Improves performance (local disk access)
* Keeps source data unchanged

---

## 12. Notes for Team

* Always run the watcher before uploading new data
* Do not upload shortcut folders
* Do not manually modify the staging directory
* All outputs are automatically generated in `decoded_frames`

---

## 13. Future Improvements

* Logging system
* Performance optimization
* Parallel processing
* Data validation (frame count, completeness)

---

## 14. Summary

This module is a complete automated pipeline that handles:

* Detection
* Upload validation
* Staging
* Extraction
* Cleanup
* Output generation

It should be treated as a full processing engine rather than a simple watcher.
