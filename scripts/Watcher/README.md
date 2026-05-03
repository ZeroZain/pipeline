# Watcher

## Overview
The Watcher script acts as the bridge between the capture device and the processing machine. It monitors the synchronized Google Drive folder, safely downloading and unpacking the data.

## Thesis Alignment: Frame Extraction from OIS and Non-OIS Videos
This script directly implements the "Frame Extraction" stage of the methodology. 
* It extracts individual frames from the RAW video recordings (`.zip` files).
* It enforces "Decode validity" (from the Video Quality Cleaning checklist) by removing non-DNG files and ensuring frames are preserved at their native recording frame rate without temporal loss.

## How It Works
1. Continuously scans Google Drive for new `capture_xxx` folders.
2. Waits for file stability to ensure uploads are fully complete.
3. Recursively extracts ZIP archives locally into a staging area.
4. Cleans out non-RAW files, preserving only the DNG frames.
5. Moves the validated frame sequences into `workspace/data/decoded_frames/`.

## Setup & Usage

### Dependencies
This script requires the `watchdog` library to monitor folder changes.
```bash
pip install watchdog
```

### Running the Script
Run the script from the project root:
```bash
python scripts/Watcher/watcher.py
```
