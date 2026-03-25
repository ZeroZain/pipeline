import os
import cv2
import csv
import json
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
STATE_FILE = os.path.join(DATASET_ROOT, "dataset_state.json")

MAX_SCENES_PER_VIDEO = 2
BLUR_PERCENTILE = 30
MAX_BLUR_SEARCH = 8
SEARCH_WINDOW = 5

DEBUG_MODE = True
DEBUG_OUTPUT = "debug_vis"

# ================= SETUP =================

def ensure_dirs():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(LAPLACIAN_LOG_DIR, exist_ok=True)
    if DEBUG_MODE:
        os.makedirs(DEBUG_OUTPUT, exist_ok=True)

# ================= STATE =================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"processed_captures": []}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# ================= IMAGE =================

def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            return None
    return cv2.imread(path)

# ================= SHARPNESS =================

def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# ================= FRAME SCORING =================

def score_frames(folder, capture_name, cam_type):
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

    files = sorted(os.listdir(folder))
    scores = []

    for f in tqdm(files, desc=f"Scoring {capture_name}-{cam_type}", leave=False):
        img = read_image(os.path.join(folder, f))
        if img is None:
            continue
        scores.append((f, laplacian_score(img)))

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score"])
        writer.writerows(scores)

    return scores

# ================= SEGMENTS =================

def build_segments(values, threshold):
    segments = []
    current = [0]

    def get_type(v):
        return "sharp" if v > threshold else "blur"

    for i in range(1, len(values)):
        if get_type(values[i]) == get_type(values[i - 1]):
            current.append(i)
        else:
            segments.append(current)
            current = [i]

    segments.append(current)
    return segments

# ================= BLUR SELECTION =================

def select_blur(values, segment):
    candidates = segment[:MAX_BLUR_SEARCH]
    return min(candidates, key=lambda x: values[x])

# ================= DEBUG =================

def visualize_selection(scene_id,
                        ois_sharp, ois_blur,
                        nonois_sharp, nonois_blur,
                        ois_sharp_score, ois_blur_score,
                        nonois_sharp_score, nonois_blur_score):

    if any(img is None for img in [ois_sharp, ois_blur, nonois_sharp, nonois_blur]):
        return

    vis = np.hstack([ois_sharp, ois_blur, nonois_sharp, nonois_blur])
    h, w = ois_sharp.shape[:2]

    cv2.putText(vis, f"OIS SHARP: {ois_sharp_score:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(vis, f"OIS BLUR: {ois_blur_score:.2f}",
                (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.putText(vis, f"NO-OIS SHARP: {nonois_sharp_score:.2f}",
                (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(vis, f"NO-OIS BLUR: {nonois_blur_score:.2f}",
                (3*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

    out_path = os.path.join(DEBUG_OUTPUT, f"scene_{scene_id:03d}.jpg")
    cv2.imwrite(out_path, vis)

# ================= CSV =================

def append_csv(file, header, row):
    exists = os.path.exists(file)
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

# ================= SCENE =================

def build_scene(scene_id, capture_name,
                ois_sharp, ois_blur,
                nonois_sharp, nonois_blur,
                ois_dir, nonois_dir):

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
        shutil.copy(os.path.join(src_dir, filename),
                    os.path.join(scene_path, f"{label}{ext}"))

    append_csv(
        SCENE_LOG,
        ["scene", "capture", "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [scene_name, capture_name,
         ois_sharp, ois_blur,
         nonois_sharp, nonois_blur]
    )

# ================= PROCESS =================

def process_capture(capture_name, scene_id):

    capture_path = os.path.join(DECODED_ROOT, capture_name)
    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")

    ois_results = score_frames(ois_dir, capture_name, "ois")
    nonois_results = score_frames(nonois_dir, capture_name, "nonois")

    if not ois_results or not nonois_results:
        return scene_id

    ois_values = [s for _, s in ois_results]
    nonois_values = [s for _, s in nonois_results]

    threshold_ois = np.percentile(ois_values, BLUR_PERCENTILE)

    ois_segments = build_segments(ois_values, threshold_ois)
    scene_count = 0

    for i in range(len(ois_segments) - 1):

        if scene_count >= MAX_SCENES_PER_VIDEO:
            break

        sharp_seg = ois_segments[i]
        blur_seg = ois_segments[i + 1]

        if ois_values[sharp_seg[0]] <= threshold_ois:
            continue
        if ois_values[blur_seg[0]] > threshold_ois:
            continue

        sharp_idx = max(sharp_seg, key=lambda x: ois_values[x])
        blur_idx = select_blur(ois_values, blur_seg)

        # match non-OIS frames near OIS indices
        sharp_idx_nonois = min(range(len(nonois_values)), key=lambda x: abs(x - sharp_idx))
        blur_idx_nonois = min(range(len(nonois_values)), key=lambda x: abs(x - blur_idx))

        # refine within window
        s_start = max(0, sharp_idx_nonois - SEARCH_WINDOW)
        s_end = min(len(nonois_values), sharp_idx_nonois + SEARCH_WINDOW)

        sharp_idx_nonois = max(range(s_start, s_end), key=lambda x: nonois_values[x])

        b_start = max(0, blur_idx_nonois - SEARCH_WINDOW)
        b_end = min(len(nonois_values), blur_idx_nonois + SEARCH_WINDOW)

        blur_idx_nonois = min(range(b_start, b_end), key=lambda x: nonois_values[x])

        sharp_file = ois_results[sharp_idx][0]
        blur_file = ois_results[blur_idx][0]

        if DEBUG_MODE:
            visualize_selection(
                scene_id,
                read_image(os.path.join(ois_dir, sharp_file)),
                read_image(os.path.join(ois_dir, blur_file)),
                read_image(os.path.join(nonois_dir, nonois_results[sharp_idx_nonois][0])),
                read_image(os.path.join(nonois_dir, nonois_results[blur_idx_nonois][0])),
                ois_values[sharp_idx],
                ois_values[blur_idx],
                nonois_values[sharp_idx_nonois],
                nonois_values[blur_idx_nonois]
            )

        build_scene(
            scene_id,
            capture_name,
            sharp_file,
            blur_file,
            nonois_results[sharp_idx_nonois][0],
            nonois_results[blur_idx_nonois][0],
            ois_dir,
            nonois_dir
        )

        scene_id += 1
        scene_count += 1

    return scene_id

# ================= MAIN =================

def main():
    ensure_dirs()
    state = load_state()

    captures = sorted([
        d for d in os.listdir(DECODED_ROOT)
        if os.path.isdir(os.path.join(DECODED_ROOT, d))
    ])

    scene_id = 1

    for capture in tqdm(captures, desc="Processing"):

        if capture in state["processed_captures"]:
            tqdm.write(f"Skipping (already processed): {capture}")
            continue

        prev_scene_id = scene_id
        scene_id = process_capture(capture, scene_id)

        if scene_id > prev_scene_id:
            state["processed_captures"].append(capture)
            save_state(state)

if __name__ == "__main__":
    main()