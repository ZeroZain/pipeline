# CaptureSync

## Overview
CaptureSync acts as the initial data acquisition and pairing utility, functioning immediately after raw videos are recorded. It runs on the capture device to synchronize and organize the raw OIS and non-OIS recordings into a structured Google Drive folder.

## Thesis Alignment: Data Acquisition Preparation
While not a formal processing stage in the final pipeline, this script ensures the fundamental requirement of "Dual Smartphone Setup" is organized. It pairs folders by timestamp (±2 seconds tolerance) to account for real-world capture delays and classifies them by motion method (HandShake, Sliding, Vibration).

## How It Works
1. Scans `incoming/ois/` and `incoming/nonois/`.
2. Validates folder names.
3. Matches OIS and non-OIS folders based on timestamp proximity.
4. Moves paired data into a shared, versioned structure on Google Drive.
5. Logs operations for traceability.

## Setup & Usage

### Dependencies
This script only relies on standard Python libraries (`os`, `shutil`, `re`, `datetime`). No external packages are required.

### Running the Script
Run the script from the project root:
```bash
python scripts/CaptureSync/capture_sync.py
```
