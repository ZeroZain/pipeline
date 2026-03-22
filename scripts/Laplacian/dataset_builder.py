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

MIN_BLUR_FRAMES = 10
BLUR_THRESHOLD_RATIO = 0.5
MAX_SCENES_PER_VIDEO = 2
SEARCH_WINDOW = 3

# ==========================================


def ensure_dirs():
    """
    Create required directories.
    """
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(LAPLACIAN_LOG_DIR, exist_ok=True)


# ================= LOG-BASED CHECK =================

def is_processed_from_logs(capture_name):
    """
    A capture is considered processed if both OIS and NON-OIS logs exist.
    """
    ois_log = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_ois.csv")
    nonois_log = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_nonois.csv")

    return os.path.exists(ois_log) and os.path.exists(nonois_log)


# ================= IMAGE =================

def read_image(path):
    """
    Read image from disk (supports DNG and standard formats).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            return None

    return cv2.imread(path)


def laplacian_score(image):
    """
    Compute sharpness using Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ================= FRAME SCORING =================

def score_frames(folder, capture_name, cam_type):
    """
    Load scores from log if available, otherwise compute and save.
    """
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")

    # ✅ LOAD existing log (fast)
    if os.path.exists(log_file):
        scores = []
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                scores.append((row[0], float(row[1])))
        return scores

    # ❌ Otherwise compute
    if not os.path.exists(folder):
        return []

    files = sorted(os.listdir(folder))
    scores = []

    for f in files:
        path = os.path.join(folder, f)
        img = read_image(path)

        if img is None:
            continue

        scores.append((f, laplacian_score(img)))

    # Save log
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score"])
        writer.writerows(scores)

    return scores


# ================= SEGMENT DETECTION =================

def detect_segments(scores):
    """
    Detect blur segments using relative threshold.
    """
    values = np.array([s for _, s in scores])

    if len(values) == 0:
        return None, []

    sharp_idx = int(np.argmax(values))
    sharp_score = values[sharp_idx]
    blur_threshold = sharp_score * BLUR_THRESHOLD_RATIO

    segments = []
    current = []

    for i, v in enumerate(values):
        if v < blur_threshold:
            current.append(i)
        else:
            if len(current) >= MIN_BLUR_FRAMES:
                segments.append(current)
            current = []

    if len(current) >= MIN_BLUR_FRAMES:
        segments.append(current)

    return sharp_idx, segments


# ================= CSV =================

def append_csv(file, header, row):
    """
    Append row to CSV (create with header if not exists).
    """
    exists = os.path.exists(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(header)

        writer.writerow(row)


# ================= SCENE BUILD =================

def build_scene(scene_id, capture_name,
                ois_sharp, ois_blur,
                nonois_sharp, nonois_blur,
                ois_dir, nonois_dir):
    """
    Create dataset scene by copying selected frames.
    """
    scene_name = f"scene_{scene_id:03d}"
    scene_path = os.path.join(DATASET_ROOT, scene_name)

    os.makedirs(scene_path, exist_ok=True)

    for src_dir, filename, label in [
        (ois_dir, ois_sharp, "ois_sharp"),
        (ois_dir, ois_blur, "ois_blur"),
        (nonois_dir, nonois_sharp, "nonois_sharp"),
        (nonois_dir, nonois_blur, "nonois_blur")
    ]:
        ext = os.path.splitext(filename)[1]

        shutil.copy(
            os.path.join(src_dir, filename),
            os.path.join(scene_path, f"{label}{ext}")
        )

    append_csv(
        SCENE_LOG,
        ["scene", "capture", "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [scene_name, capture_name,
         ois_sharp, ois_blur,
         nonois_sharp, nonois_blur]
    )

    return scene_name


# ================= MAIN PROCESS =================

def process_capture(capture_name, scene_id):
    """
    Process a single capture using log-based skipping.
    """
    if is_processed_from_logs(capture_name):
        tqdm.write(f"Skipping (logs exist): {capture_name}")
        return scene_id

    capture_path = os.path.join(DECODED_ROOT, capture_name)

    tqdm.write(f"\n--- Processing {capture_name} ---")

    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")

    ois_results = score_frames(ois_dir, capture_name, "ois")
    nonois_results = score_frames(nonois_dir, capture_name, "nonois")

    if not ois_results or not nonois_results:
        return scene_id

    _, blur_segments = detect_segments(ois_results)

    if not blur_segments:
        tqdm.write("No blur segments found.")
        return scene_id

    values = np.array([s for _, s in ois_results])

    blur_segments.sort(key=lambda seg: np.mean([values[i] for i in seg]))
    top_segments = blur_segments[:MAX_SCENES_PER_VIDEO]

    for segment in top_segments:

        blur_idx_ois = segment[2] if len(segment) > 2 else segment[0]

        sharp_start = max(0, blur_idx_ois - 15)
        sharp_idx_ois = sharp_start + np.argmax(values[sharp_start:blur_idx_ois])
        sharp_f_ois = ois_results[sharp_idx_ois][0]

        s_start = max(0, sharp_idx_ois - SEARCH_WINDOW)
        s_end = min(len(nonois_results), sharp_idx_ois + SEARCH_WINDOW + 1)

        best_nonois_sharp = max(nonois_results[s_start:s_end], key=lambda x: x[1])

        sharp_f_nonois = best_nonois_sharp[0]
        threshold = best_nonois_sharp[1] * BLUR_THRESHOLD_RATIO

        b_start = max(0, blur_idx_ois - SEARCH_WINDOW)
        b_end = min(len(nonois_results), blur_idx_ois + SEARCH_WINDOW + 1)

        candidates = [x for x in nonois_results[b_start:b_end] if x[1] <= threshold]

        if not candidates:
            continue

        worst_blur = min(candidates, key=lambda x: x[1])

        build_scene(
            scene_id,
            capture_name,
            sharp_f_ois,
            ois_results[blur_idx_ois][0],
            sharp_f_nonois,
            worst_blur[0],
            ois_dir,
            nonois_dir
        )

        scene_id += 1

    return scene_id


# ================= ENTRY =================

def main():
    """
    Main pipeline entry using log-based tracking.
    """
    ensure_dirs()

    captures = sorted([
        d for d in os.listdir(DECODED_ROOT)
        if os.path.isdir(os.path.join(DECODED_ROOT, d))
    ])

    processed_count = sum(1 for c in captures if is_processed_from_logs(c))

    tqdm.write(f"Total captures: {len(captures)}")
    tqdm.write(f"Already processed (from logs): {processed_count}")

    scene_id = 1

    for capture in tqdm(captures, desc="Overall Progress"):
        scene_id = process_capture(capture, scene_id)


if __name__ == "__main__":
    main()