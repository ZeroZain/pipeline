# Watcher Pipeline

## Overview

The watcher scans a Google Drive Desktop sync folder, detects dataset `capture_xxx` folders, waits until uploads are stable, extracts ZIP files, removes non-DNG files, and moves the cleaned result into the local `decoded_frames` dataset.

The watcher scans the Google Drive root for method folders directly:

```text
G:\My Drive\Thesis or Crisis\Videos\Dataset Capture\
  HandShake Method\
  Sliding Method\
  Vibration Method\
```

Inside each method folder, the watcher looks for `capture_xxx` folders.

---

## Expected Input Structure

```text
Dataset Capture/
  HandShake Method/
    capture_001/
      ois/
      nonois/
  Sliding Method/
    capture_001/
      ois/
      nonois/
  Vibration Method/
    capture_001/
      ois/
      nonois/
```

Each `capture_xxx` folder must contain both `ois/` and `nonois/`.
ZIP files may exist directly inside those folders or inside nested subfolders.

Supported examples:

```text
capture_001/
  ois/
    file.zip
  nonois/
    file.zip
```

```text
capture_001/
  ois/
    nested/
      file.zip
  nonois/
    nested/
      file.zip
```

```text
capture_001/
  ois/
    file.zip
    nested/
      file.zip
  nonois/
    file.zip
```

---

## Output Structure

The watcher preserves the method folder when moving processed captures into local output:

```text
decoded_frames/
  HandShake Method/
    capture_001/
      ois/
        *.dng
      nonois/
        *.dng
  Sliding Method/
    capture_001/
      ois/
        *.dng
      nonois/
        *.dng
  Vibration Method/
    capture_001/
      ois/
        *.dng
      nonois/
        *.dng
```

Only `.dng` files remain in the final output.

---

## What The Watcher Does

1. Scans the Google Drive root every `15` seconds.
2. Finds `capture_xxx` folders under `HandShake Method`, `Sliding Method`, and `Vibration Method`.
3. Skips captures that are already present in `decoded_frames`.
4. Skips captures that are incomplete.
5. Queues only captures that have:
   - `ois/`
   - `nonois/`
   - at least one ZIP file in each side
6. Waits until ZIP file sizes are stable before processing.
7. Copies the capture into `staging/`.
8. Extracts ZIP files recursively.
9. Flattens nested extracted folders.
10. Removes non-DNG files.
11. Moves the cleaned capture into `decoded_frames/`.

---

## Reliability Behavior

### Duplicate Protection

The watcher does not process the same capture twice at the same time.

It tracks:
- queued captures
- active captures
- already processed captures in `decoded_frames`

### Incomplete Captures

A `capture_xxx` folder is not queued if any of these are missing:
- `ois/`
- `nonois/`
- ZIP files under `ois/`
- ZIP files under `nonois/`

This prevents wasting time on empty or partially uploaded folders.

### Upload Stability Check

Even after a capture looks structurally ready, the watcher still waits for ZIP files to stop changing size before extraction.

Default timeout:
- `UPLOAD_TIMEOUT = 300` seconds

### Retry Handling

If processing fails, the watcher retries once immediately.

If the capture still fails, the watcher stores a signature of the source ZIP files and will not keep retrying the exact same failed source state on every scan.

If the source files change later in Google Drive, the capture becomes eligible again.

### Safe Staging

The watcher processes data in `staging/` before moving it into `decoded_frames/`.
This reduces the chance of corrupting the source folder or the final output.

---

## Console Status Summary

After each scan, the watcher prints a compact summary like this:

```text
[SCAN] found=24 ready=20 enqueued=2 busy=1 processed=18 incomplete=4 failed_hold=0 queued=2 active=1 failed=0
```

Field meanings:
- `found`: total `capture_xxx` folders discovered in Google Drive
- `ready`: captures that currently have the required structure
- `enqueued`: captures newly added to the queue during this scan
- `busy`: captures already queued or already being processed
- `processed`: captures already present in `decoded_frames`
- `incomplete`: captures missing required folders or ZIP files
- `failed_hold`: captures skipped because they already failed with the same unchanged source files
- `queued`: current queue size after the scan
- `active`: captures currently being processed by workers
- `failed`: number of failed-source signatures currently remembered by the watcher

---

## Key Paths

- Google Drive input root:
  - `G:\My Drive\Thesis or Crisis\Videos\Dataset Capture`
- Local staging folder:
  - `workspace/data/staging`
- Local output folder:
  - `workspace/data/decoded_frames`

---

## Setup

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependency if needed:

```powershell
pip install watchdog
```

Run the watcher:

```powershell
python scripts\Watcher\watcher.py
```

---

## Notes

- The Google Drive root must exist before startup.
- The watcher uses polling plus worker threads; it does not depend on file system events for correctness.
- The current code starts `2` workers.
- ZIP extraction is recursive, but final output is flattened and cleaned.
