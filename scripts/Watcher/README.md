# Watcher Pipeline (Detection and Processing)

## 1. Overview

The Watcher Pipeline automatically detects and processes `capture_xxx` folders from Google Drive.

It serves as the entry point and processing engine of the system, handling the full workflow from detection to final dataset generation. The pipeline is designed to be robust against incomplete uploads and flexible in handling different input folder structures.

---

## 2. Functionality

The pipeline performs the following steps:

1. Monitors a Google Drive–synced folder on the local machine
2. Detects new and existing `capture_xxx` folders
3. Waits until the upload is fully complete
4. Copies data into a local staging directory
5. Recursively locates ZIP files inside `ois` and `nonois` folders
6. Extracts ZIP files and flattens nested structures
7. Removes all non-`.dng` files (e.g., `.zip`, `.wav`, `.ini`)
8. Moves the processed data into the final output directory

---

## 3. Supported Input Structures

The pipeline supports both flat and nested structures.

### 3.1 Standard Structure

```text
capture_xxx/
   ├── ois/
   │     file.zip
   └── nonois/
         file.zip
```

---

### 3.2 Nested Structure (Supported)

```text
capture_xxx/
   ├── ois/
   │     some_folder/
   │         file.zip
   └── nonois/
         another_folder/
             file.zip
```

---

### 3.3 Mixed Structure

```text
capture_xxx/
   ├── ois/
   │     file.zip
   │     folder/
   │         file.zip
   └── nonois/
         file.zip
```

All ZIP files are automatically detected regardless of depth and processed into a consistent output format.

---

## 4. Pipeline Flow

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
Recursive ZIP Detection
        ↓
Extract and Flatten
        ↓
Clean Non-DNG Files
        ↓
Move → decoded_frames
```

---

## 5. Directory Structure

### 5.1 Watch Folder (Google Drive)

```text
G:\My Drive\Thesis or Crisis\Videos\automationInput
```

Requirements:

* Must exist locally
* Must be synced using Google Drive for Desktop
* Must contain real folders (not shortcuts)

---

### 5.2 Staging Folder (Temporary Processing)

```text
C:\Users\Windows 11\Documents\GitHub\pipeline\staging
```

Used for:

* Safe local processing
* Temporary extraction and transformation

Note:

* This folder may appear empty after processing because files are moved to the output directory

---

### 5.3 Output Folder

```text
C:\Users\Windows 11\Documents\GitHub\pipeline\decoded_frames
```

Contains the final processed dataset with only `.dng` files.

---

## 6. Setup and Execution

### 6.1 Open Terminal

```bash
cd pipeline
```

---

### 6.2 Activate Virtual Environment

```bash
.\.venv\Scripts\activate
```

---

### 6.3 Run the Watcher

```bash
python scripts/Watcher/watcher.py
```

---

## 7. Expected Behavior

### 7.1 On Startup

```text
[INIT] Checking existing folders...
Watching for new capture folders...
```

---

### 7.2 During Processing

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

## 8. Testing Procedure

1. Run the watcher script
2. Upload a folder to Google Drive:

```text
capture_003/
   ├── ois/
   │     folder/
   │         file.zip
   └── nonois/
         file.zip
```

3. Wait until processing completes

---

## 9. Expected Output

```text
decoded_frames/
   capture_003/
      ├── ois/
      │     *.dng
      └── nonois/
            *.dng
```

All nested structures are flattened, and only `.dng` files remain.

---

## 10. Important Behavior

* Processing is sequential (one capture at a time)
* Both existing and newly added folders are supported
* Upload completion is validated before processing
* Nested ZIP structures are automatically handled
* Staging ensures safe and consistent processing
* Non-essential files are automatically removed

---

## 11. Common Issues

### 11.1 Incomplete Output

Cause:

* Folder detected before upload finished

Solution:

* The system includes upload validation logic
* Ensure stable internet connection during upload

---

### 11.2 Google Drive Issues

* Files must be fully synced locally
* Avoid using shortcuts
* Use real folders only

---

### 11.3 Module Not Found

Install dependency:

```bash
pip install watchdog
```

---

## 12. Design Rationale

### 12.1 Use of Staging

```text
Drive → Staging → Processing → Output
```

Advantages:

* Prevents partial file processing
* Improves performance (local disk access)
* Keeps source data unchanged

---

### 12.2 Flexible Input Handling

The system uses recursive file discovery to handle inconsistent folder structures. This ensures compatibility with real-world data where file organization may vary.

---

## 13. Notes for Team

* Always run the watcher before uploading new data
* Do not upload shortcut folders
* Do not manually modify the staging directory
* Output will automatically appear in `decoded_frames`

---

## 14. Future Improvements

* Logging system
* Performance optimization
* Parallel processing
* Data validation (frame count, completeness)
* Corrupted ZIP detection

---

## 15. Summary

This module implements a complete automated data processing pipeline that handles:

* Detection
* Upload validation
* Flexible input ingestion
* Staging
* Extraction and normalization
* Cleanup
* Output generation

It should be treated as a full processing engine rather than a simple watcher.
