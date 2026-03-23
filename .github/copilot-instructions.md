# Copilot guidance for the pipeline repo

This file captures concise, actionable knowledge for AI coding agents to be productive quickly in this repository.

1) Big picture
- Purpose: constructs paired deblurring scenes from MotionCam frames. Stages: decode → laplacian scoring (scene build) → alignment (geo → photo → color) → interpolation (256×256) → final dataset.
- Key folders: `decoded_frames/` (input), `dataset/` (generated scenes), `aligned/` (alignment outputs), `dataset_256/` (final interpolated dataset), `logs/` (processing trace).

2) Primary entry scripts & how to run (concrete)
- Watcher (auto-ingest from Drive): `python scripts/Watcher/watcher.py`. Requires WATCH_FOLDER to point to a Google Drive local sync folder; staging is `staging/`, output `decoded_frames/`.
- Laplacian / scene construction: `python scripts/Laplacian/dataset_builder.py` (or `scene_builder.py` variant). Produces `dataset/scene_XXX` and writes `logs/laplacian/*.csv` and `logs/scene_selection_log.csv`.
- Alignment pipeline: `python scripts/Alignment/pipeline_runner.py`. Change GT via `GT_SOURCE = "ois"` or `"nonois"` at top of `pipeline_runner.py`.
- Interpolation: `python scripts/Interpolation/interpolation.py`. Reads `aligned/gt_<ois|nonois>/color/` and writes `dataset_256/`.

3) Codebase conventions you must follow
- Scene folder naming and files: every scene folder must contain exactly `ois_sharp`, `ois_blur`, `nonois_sharp`, `nonois_blur` (extension `.dng|.jpg|.png`). Do not rename files; the alignment code expects those exact prefixes and will form output names by `to_jpg()`.
- Logs are authoritative: the pipeline uses files under `logs/` (e.g., `logs/laplacian/*.csv`, `logs/scene_selection_log.csv`, `logs/geo_log.csv`, `logs/interpolation_log.csv`) to decide whether to skip or resume work. Prefer updating logs where appropriate rather than bypassing them.
- Resume/state files: `dataset/dataset_state.json` is used by the Laplacian stage (and other scripts append logs). Use the same keys (`next_scene_id`, `processed_captures`) when modifying state.

4) Important implementation patterns & examples
- Laplacian scoring: uses variance-of-Laplacian (see `scripts/Laplacian/*`). Search window and segment detection are used to align OIS and non-OIS frames—do not assume perfect frame sync.
- Alignment: `scripts/Alignment/pipeline_runner.py` uses SIFT feature detection and `cv2.findHomography` with RANSAC; fallback uses a master_H if per-image matching fails. Photometric alignment is mean-shift per-channel; color alignment is LAB global-mean matching. Validation thresholds are recorded in `logs/geo_log.csv`, `photo_log.csv`, `color_log.csv`.
- Interpolation: center inward-crop (default `crop_factor=0.8`) then `cv2.INTER_CUBIC` resize to 256×256. The script skips already-processed files (resume logic).

5) External deps & gotchas discovered in code
- Uses `rawpy` for DNG decoding and OpenCV for core ops. The alignment script calls `cv2.SIFT_create()` — ensure OpenCV build includes SIFT (install `opencv-contrib-python` if SIFT is missing). Recommended pip installs found in READMEs:
  - `pip install opencv-contrib-python numpy scikit-image rawpy tqdm` (or `opencv-contrib-python` instead of opencv-python when SIFT is required).
- RAW decoding and Laplacian scoring are CPU-heavy; expect long runtime on many captures.

6) Where to change behavior (quick code pointers)
- Change ground-truth selection: top of `scripts/Alignment/pipeline_runner.py` and `scripts/Interpolation/interpolation.py` (`GT_SOURCE = "ois"` / `"nonois"`).
- Adjust geometric thresholds: `GEO_INLIER_THRESHOLD`, `FLOW_THRESHOLD` in `pipeline_runner.py`.
- Adjust Laplacian heuristics: `BLUR_THRESHOLD_RATIO`, `MIN_BLUR_FRAMES`, `SEARCH_WINDOW` in `scripts/Laplacian/dataset_builder.py`.
- Change crop/target size: `crop_factor` and `TARGET_SIZE` in `scripts/Interpolation/interpolation.py`.

7) Useful quick checks and debugging hints
- If a scene isn't produced, inspect `logs/laplacian/` and `logs/scene_selection_log.csv` to see frame scores and selected files (example: `logs/scene_selection_log.csv` maps scene → capture → filenames).
- If alignment fails or SIFT errors occur, check your OpenCV installation; SIFT often requires `opencv-contrib-python`.
- If images are missing in `aligned/gt_*`, confirm `GT_SOURCE` matches the chosen reference and that `dataset/scene_xxx` contains the required `_sharp` files.
- Use the CSV logs (`geo_log.csv`, `photo_log.csv`, `color_log.csv`, `scene_fail_log.csv`) to determine why pairs were rejected (columns like `inlier_ratio`, `mean_flow`, `deltaE_before/after`, `overall_pass`).

8) Minimal environment setup (discovered in READMEs)
- Create & activate venv (Windows PowerShell):

  .\.venv\Scripts\Activate.ps1

- Install deps (examples gathered from READMEs):

  pip install opencv-contrib-python numpy scikit-image rawpy tqdm

9) What to avoid
- Do not arbitrarily rename the four scene files (ois_sharp, ois_blur, nonois_sharp, nonois_blur).
- Do not delete or rewrite the logs unless you intentionally want to reset resume state—these logs are used to skip processed work.

If anything in these instructions is unclear or you want more detail for a specific task (tests, CI, or modifying a concrete stage), tell me which area to expand and I will iterate.
