# Alignment Pipeline

## Overview
The Alignment script is a comprehensive three-stage normalization module that acts upon the constructed scenes. It corrects spatial, intensity, and chromatic discrepancies between the paired OIS and non-OIS frames.

## Thesis Alignment: Geometric, Photometric, and Color Alignment
This script strictly follows the three-step alignment procedure defined in the methodology:
1. **Geometric Alignment:** Achieves pixel-level correspondence by compensating for translation, rotation, and small perspective deviations using feature-based registration.
2. **Photometric Alignment:** Reduces intensity differences by mapping exposure response and sensor behavior statistics to match the sharp reference frame.
3. **Color Alignment:** Normalizes color distributions (chromatic shifts) to ensure consistent appearance, preventing color discrepancies from influencing learning-based restoration models.

## Outputs
* The aligned images are saved as JPEG files into `workspace/data/aligned/`.
* Validation metrics for each stage are logged in `geo_log.csv`, `photo_log.csv`, and `color_log.csv`.

## Setup & Usage

### Dependencies
This script relies heavily on OpenCV and scikit-image for feature matching, transformation, and evaluation metrics.
```bash
pip install opencv-python numpy rawpy tqdm scikit-image
```

### Running the Script
Run the script from the project root. It will process all pending scenes.
```bash
python scripts/Alignment/align4.py
```

To run the alignment on a specific scene only (e.g. for debugging or after applying frame overrides via the Dashboard):
```bash
python scripts/Alignment/align4.py --scene "HandShake Method/scene_001"
```