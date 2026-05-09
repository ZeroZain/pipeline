# Interpolation

## Overview
The Interpolation script normalizes the spatial resolution of all aligned image pairs, preparing them for final structured output.

## Thesis Alignment: Interpolation (Downsampling)
This script directly implements the "Interpolation (Downsampling)" stage of the methodology.
* It applies inward center cropping to remove invalid border regions introduced during Geometric Alignment.
* It applies **Bicubic interpolation** to standardize images.
* The default size is **512 × 512 pixels** (configurable).
* As noted in the thesis, Bicubic interpolation preserves edge structures more effectively than Nearest Neighbor or Bilinear methods, maintaining structural consistency for fair evaluation.

## Outputs
* The final resized images are output to `workspace/data/dataset_<size>/gt_ois/`.

## Setup & Usage

### Dependencies
This script utilizes OpenCV for bicubic interpolation.
```bash
pip install opencv-python numpy tqdm
```

### Running the Script
Run the script from the project root. It will process all aligned scenes.
```bash
python scripts/Interpolation/interpolate.py
```

To use a different size (e.g., 256):
```bash
python scripts/Interpolation/interpolate.py --size 256
```

To run interpolation on a specific scene only:
```bash
python scripts/Interpolation/interpolate.py --scene "HandShake Method/scene_001"
```
