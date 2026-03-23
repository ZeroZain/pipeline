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
MAX_SCENES_PER_VIDEO = 2
SEARCH_WINDOW = 3
BLUR_PERCENTILE = 40  # relative blur threshold

# Debug options
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

# ================= LOG CHECK =================

def is_processed_from_logs(capture_name):
    """Check if capture already processed using existing logs."""
    ois_log = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_ois.csv")
    nonois_log = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_nonois.csv")
    return os.path.exists(ois_log) and os.path.exists(nonois_log)

# ================= IMAGE =================

def read_image(path):
    """Read image (supports RAW and standard formats)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            return None

    return cv2.imread(path)

# ================= SHARPNESS =================

def laplacian_score(image):
    """Compute noise-robust sharpness score."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise to suppress noise influence
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)

    # Edge strength
    lap = cv2.Laplacian(denoised, cv2.CV_64F).var()

    # Noise estimate
    noise = np.std(gray.astype(np.float32) - denoised.astype(np.float32))

    return float(lap - 0.5 * noise)

# ================= FRAME SCORING =================

def score_frames(folder, capture_name, cam_type):
    """Compute or load sharpness scores."""
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")

    # Load existing scores
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

    files = sorted(os.listdir(folder))
    scores = []

    for f in tqdm(files, desc=f"Scoring {capture_name}-{cam_type}", leave=False):
        img = read_image(os.path.join(folder, f))
        if img is None:
            continue
        scores.append((f, laplacian_score(img)))

    # Save scores
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score"])
        writer.writerows(scores)

    return scores

# ================= SEGMENT DETECTION =================

def detect_segments(scores):
    """Detect blur segments using percentile threshold."""
    values = np.array([s for _, s in scores])

    if len(values) == 0:
        return None, []

    sharp_idx = int(np.argmax(values))
    threshold = np.percentile(values, BLUR_PERCENTILE)

    segments = []
    current = []

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
    """Check if sharp and blur separation is sufficient."""
    diff = sharp_score - blur_score
    return diff >= 0.1, diff

# ================= DEBUG =================

def visualize_selection(scene_id, sharp_img, blur_img, sharp_score, blur_score):
    """Save side-by-side visualization of selected frames."""
    if sharp_img is None or blur_img is None:
        return

    vis = np.hstack([sharp_img, blur_img])

    cv2.putText(vis, f"SHARP: {sharp_score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(vis, f"BLUR: {blur_score:.3f}",
                (sharp_img.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    out_path = os.path.join(DEBUG_OUTPUT, f"scene_{scene_id:03d}.jpg")
    cv2.imwrite(out_path, vis)

# ================= CSV =================

def append_csv(file, header, row):
    """Append row to CSV file."""
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
    """Copy selected frames into dataset structure."""
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

# ================= MAIN =================

def process_capture(capture_name, scene_id):
    """Process capture: scoring, detection, pairing, validation."""
    if is_processed_from_logs(capture_name):
        tqdm.write(f"Skipping: {capture_name}")
        return scene_id

    capture_path = os.path.join(DECODED_ROOT, capture_name)

    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")

    ois_results = score_frames(ois_dir, capture_name, "ois")
    nonois_results = score_frames(nonois_dir, capture_name, "nonois")

    if not ois_results or not nonois_results:
        return scene_id

    _, blur_segments = detect_segments(ois_results)
    if not blur_segments:
        return scene_id

    values = np.array([s for _, s in ois_results])
    blur_segments.sort(key=lambda seg: np.mean([values[i] for i in seg]))

    for segment in blur_segments[:MAX_SCENES_PER_VIDEO]:

        blur_idx = segment[len(segment)//2]

        sharp_start = max(0, blur_idx - 15)
        sharp_idx = sharp_start + np.argmax(values[sharp_start:blur_idx])

        sharp_file = ois_results[sharp_idx][0]
        blur_file = ois_results[blur_idx][0]

        sharp_score = values[sharp_idx]
        blur_score = values[blur_idx]

        valid, diff = validate_selection(sharp_score, blur_score)
        if not valid:
            tqdm.write(f"⚠️ Weak separation ({diff:.3f}) in {capture_name}")

        # Debug visualization
        if DEBUG_MODE:
            sharp_img = read_image(os.path.join(ois_dir, sharp_file))
            blur_img = read_image(os.path.join(ois_dir, blur_file))
            visualize_selection(scene_id, sharp_img, blur_img, sharp_score, blur_score)

        # Match NON-OIS frames
        s_start = max(0, sharp_idx - SEARCH_WINDOW)
        s_end = min(len(nonois_results), sharp_idx + SEARCH_WINDOW + 1)

        best_nonois_sharp = max(nonois_results[s_start:s_end], key=lambda x: x[1])

        b_start = max(0, blur_idx - SEARCH_WINDOW)
        b_end = min(len(nonois_results), blur_idx + SEARCH_WINDOW + 1)

        worst_nonois_blur = min(nonois_results[b_start:b_end], key=lambda x: x[1])

        build_scene(
            scene_id,
            capture_name,
            sharp_file,
            blur_file,
            best_nonois_sharp[0],
            worst_nonois_blur[0],
            ois_dir,
            nonois_dir
        )

        scene_id += 1

    return scene_id

# ================= ENTRY =================

def main():
    """Run dataset construction pipeline."""
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