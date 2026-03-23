import os
import cv2
import csv
import shutil
import numpy as np
import rawpy
from tqdm import tqdm

# ================= CONFIG =================

DECODED_ROOT = "decoded_frames"
DATASET_ROOT = "dataset"
LOG_ROOT = "logs"

LAPLACIAN_LOG_DIR = os.path.join(LOG_ROOT, "laplacian")
SCENE_LOG = os.path.join(LOG_ROOT, "scene_selection_log.csv")
BAD_LOG = os.path.join(LOG_ROOT, "bad_samples.csv")

MIN_BLUR_FRAMES = 10
MAX_SCENES_PER_VIDEO = 2
SEARCH_WINDOW = 3
BLUR_PERCENTILE = 40

# Quality controls
STRICT_MODE = True
MIN_SHARPNESS = 0.1

DEBUG_MODE = True
DEBUG_OUTPUT = "debug_vis"

# ================= SETUP =================

def ensure_dirs():
    """Create required directories."""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(LAPLACIAN_LOG_DIR, exist_ok=True)
    if DEBUG_MODE:
        os.makedirs(DEBUG_OUTPUT, exist_ok=True)

# ================= LOG =================

def append_csv(file, header, row):
    """Append row to CSV."""
    exists = os.path.exists(file)
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

def is_processed_from_logs(capture_name):
    """Skip already processed captures."""
    ois_log = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_ois.csv")
    nonois_log = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_nonois.csv")
    return os.path.exists(ois_log) and os.path.exists(nonois_log)

# ================= IMAGE =================

def read_image(path):
    """Read RAW or standard image."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True,
                                      no_auto_bright=True,
                                      output_bps=8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            return None

    return cv2.imread(path)

# ================= SHARPNESS =================

def laplacian_score(image):
    """Noise-aware sharpness score."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    lap = cv2.Laplacian(denoised, cv2.CV_64F).var()
    noise = np.std(gray.astype(np.float32) - denoised.astype(np.float32))

    return float(lap - 0.5 * noise)

# ================= SCORING =================

def score_frames(folder, capture_name, cam_type):
    """Compute or load frame scores."""
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")

    if os.path.exists(log_file):
        scores = []
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                scores.append((row[0], float(row[1])))
        return scores

    if not os.path.exists(folder):
        return []

    scores = []
    for f in tqdm(sorted(os.listdir(folder)),
                  desc=f"Scoring {capture_name}-{cam_type}", leave=False):
        img = read_image(os.path.join(folder, f))
        if img is None:
            continue
        scores.append((f, laplacian_score(img)))

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score"])
        writer.writerows(scores)

    return scores

# ================= SEGMENT =================

def detect_segments(scores):
    """Detect blur segments using normalized percentile."""
    values = np.array([s for _, s in scores])

    if len(values) == 0:
        return None, []

    # Normalize scores (important improvement)
    values = (values - np.mean(values)) / (np.std(values) + 1e-6)

    sharp_idx = int(np.argmax(values))
    threshold = np.percentile(values, BLUR_PERCENTILE)

    segments, current = [], []

    for i, v in enumerate(values):
        if v <= threshold:
            current.append(i)
        else:
            if len(current) >= MIN_BLUR_FRAMES:
                segments.append(current)
            current = []

    if len(current) >= MIN_BLUR_FRAMES:
        segments.append(current)

    return sharp_idx, segments

# ================= VALIDATION =================

def validate_selection(sharp_score, blur_score):
    """Check sharp vs blur separation."""
    diff = sharp_score - blur_score
    return diff >= 0.1, diff

# ================= DEBUG =================

def visualize_selection(scene_id, imgs, scores):
    """Visualize 4 images with scores."""
    if any(img is None for img in imgs):
        return

    h = min(img.shape[0] for img in imgs)
    imgs = [cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h)) for img in imgs]

    vis = np.hstack(imgs)

    labels = ["OIS SHARP", "OIS BLUR", "NON-OIS SHARP", "NON-OIS BLUR"]

    x = 0
    for i, img in enumerate(imgs):
        text = f"{labels[i]} | {scores[i]:.2f}"
        cv2.putText(vis, text, (x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0) if "SHARP" in labels[i] else (0,0,255), 2)
        x += img.shape[1]

    cv2.imwrite(os.path.join(DEBUG_OUTPUT, f"scene_{scene_id:03d}.jpg"), vis)

# ================= SCENE =================

def build_scene(scene_id, capture_name,
                ois_sharp, ois_blur,
                nonois_sharp, nonois_blur,
                ois_dir, nonois_dir):
    """Copy frames + log."""
    scene_name = f"scene_{scene_id:03d}"
    scene_path = os.path.join(DATASET_ROOT, scene_name)
    os.makedirs(scene_path, exist_ok=True)

    for src, file, label in [
        (ois_dir, ois_sharp, "ois_sharp"),
        (ois_dir, ois_blur, "ois_blur"),
        (nonois_dir, nonois_sharp, "nonois_sharp"),
        (nonois_dir, nonois_blur, "nonois_blur")
    ]:
        ext = os.path.splitext(file)[1]
        shutil.copy(os.path.join(src, file),
                    os.path.join(scene_path, f"{label}{ext}"))

    append_csv(
        SCENE_LOG,
        ["scene", "capture", "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [scene_name, capture_name,
         ois_sharp, ois_blur,
         nonois_sharp, nonois_blur]
    )

# ================= MAIN =================

def process_capture(capture_name, scene_id):
    """Process one capture."""
    if is_processed_from_logs(capture_name):
        tqdm.write(f"Skipping: {capture_name}")
        return scene_id

    capture_path = os.path.join(DECODED_ROOT, capture_name)

    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")

    ois = score_frames(ois_dir, capture_name, "ois")
    nonois = score_frames(nonois_dir, capture_name, "nonois")

    if not ois or not nonois:
        return scene_id

    _, segments = detect_segments(ois)
    if not segments:
        return scene_id

    values = np.array([s for _, s in ois])
    segments.sort(key=lambda seg: np.mean([values[i] for i in seg]))

    for seg in segments[:MAX_SCENES_PER_VIDEO]:

        blur_idx = seg[len(seg)//2]

        # Find sharp frame before blur
        sharp_start = max(0, blur_idx - 15)
        sharp_idx = sharp_start + np.argmax(values[sharp_start:blur_idx])

        sharp_score = values[sharp_idx]
        blur_score = values[blur_idx]

        # Skip weak sharp frames
        if sharp_score < MIN_SHARPNESS:
            continue

        # ================= NON-OIS MATCHING =================

        s_start = max(0, sharp_idx - SEARCH_WINDOW)
        s_end = min(len(nonois), sharp_idx + SEARCH_WINDOW + 1)
        if s_start >= s_end:
            continue

        b_start = max(0, blur_idx - SEARCH_WINDOW)
        b_end = min(len(nonois), blur_idx + SEARCH_WINDOW + 1)
        if b_start >= b_end:
            continue

        best_nonois_sharp = max(nonois[s_start:s_end], key=lambda x: x[1])
        worst_nonois_blur = min(nonois[b_start:b_end], key=lambda x: x[1])

        # ================= VALIDATION =================

        valid, diff = validate_selection(sharp_score, blur_score)

        if not valid:
            append_csv(
                BAD_LOG,
                [
                    "capture",
                    "scene_id",
                    "ois_sharp_frame",
                    "ois_blur_frame",
                    "nonois_sharp_frame",
                    "nonois_blur_frame",
                    "ois_sharp_score",
                    "ois_blur_score",
                    "nonois_sharp_score",
                    "nonois_blur_score",
                    "diff"
                ],
                [
                    capture_name,
                    scene_id,
                    ois[sharp_idx][0],
                    ois[blur_idx][0],
                    best_nonois_sharp[0],
                    worst_nonois_blur[0],
                    sharp_score,
                    blur_score,
                    best_nonois_sharp[1],
                    worst_nonois_blur[1],
                    diff
                ]
            )

            if STRICT_MODE:
                continue

        # ================= DEBUG =================

        if DEBUG_MODE:
            imgs = [
                read_image(os.path.join(ois_dir, ois[sharp_idx][0])),
                read_image(os.path.join(ois_dir, ois[blur_idx][0])),
                read_image(os.path.join(nonois_dir, best_nonois_sharp[0])),
                read_image(os.path.join(nonois_dir, worst_nonois_blur[0]))
            ]

            scores = [
                sharp_score,
                blur_score,
                best_nonois_sharp[1],
                worst_nonois_blur[1]
            ]

            visualize_selection(scene_id, imgs, scores)

        # ================= BUILD =================

        build_scene(
            scene_id,
            capture_name,
            ois[sharp_idx][0],
            ois[blur_idx][0],
            best_nonois_sharp[0],
            worst_nonois_blur[0],
            ois_dir,
            nonois_dir
        )

        scene_id += 1

    return scene_id

# ================= ENTRY =================

def main():
    """Run pipeline."""
    ensure_dirs()

    captures = sorted([
        d for d in os.listdir(DECODED_ROOT)
        if os.path.isdir(os.path.join(DECODED_ROOT, d))
    ])

    scene_id = 1

    for capture in tqdm(captures, desc="Processing"):
        scene_id = process_capture(capture, scene_id)

if __name__ == "__main__":
    main()