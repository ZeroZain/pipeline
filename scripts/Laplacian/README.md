# Laplacian

## Overview
The Laplacian script performs mathematical analysis on the decoded RAW frames to evaluate image sharpness across the entire video sequence.

## Thesis Alignment: Sharp and Blur Candidate Selection
This script directly implements the "Candidate Selection" stage, governed strictly by objective sharpness assessment rather than subjective visual inspection.
* It computes the **Variance of Laplacian** for each frame.
* High variance indicates high-frequency content (sharp candidates).
* Low variance indicates suppressed edge strength (motion-blurred candidates).
* The script outputs CSV logs that are later used to ensure repeatable identification of reference frames.

## Outputs
* Generates continuous sharpness metrics logged to `workspace/logs/laplacian/`.

## Setup & Usage

### Dependencies
This script requires several image processing and numerical libraries.
```bash
pip install opencv-python numpy rawpy tqdm
```

### Running the Script
Run the script from the project root:
```bash
python scripts/Laplacian/a2lap.py
```