import os
import cv2
import json
import csv
import shutil
import numpy as np
import rawpy
from tqdm import tqdm # Import tqdm for progress tracking

# Configuration
DECODED_ROOT = "decoded_frames"
DATASET_ROOT = "dataset"
LOG_ROOT = "logs"

LAPLACIAN_LOG_DIR = os.path.join(LOG_ROOT, "laplacian")
SCENE_LOG = os.path.join(LOG_ROOT, "scene_selection_log.csv")
STATE_FILE = os.path.join(DATASET_ROOT, "dataset_state.json")

# --- REFINED SETTINGS ---
MIN_BLUR_FRAMES = 10       
BLUR_THRESHOLD_RATIO = 0.5 
MAX_SCENES_PER_VIDEO = 2   
SEARCH_WINDOW = 3          

def ensure_dirs():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(LAPLACIAN_LOG_DIR, exist_ok=True)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"next_scene_id": 1, "processed_captures": []}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            return None
    else:
        return cv2.imread(path)

def laplacian_score(image):
    """Calculates image sharpness using the Variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def score_frames(folder, capture_name, cam_type):
    if not os.path.exists(folder): return []
    files = sorted(os.listdir(folder))
    scores = []
    
    # Progress bar for frame scoring
    for f in tqdm(files, desc=f"Scoring {cam_type}", leave=False):
        path = os.path.join(folder, f)
        img = read_image(path)
        if img is None: continue
        scores.append((f, laplacian_score(img)))
    
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "score"])
        for f, s in scores:
            writer.writerow([f, s])
    return scores

def detect_segments(scores):
    values = np.array([s for _, s in scores])
    if len(values) == 0: return None, []
    
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

def append_csv(file, header, row):
    exists = os.path.exists(file)
    with open(file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

def build_scene(scene_id, capture_name, ois_sharp, ois_blur, nonois_sharp, nonois_blur, ois_dir, nonois_dir):
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
        shutil.copy(os.path.join(src_dir, filename), os.path.join(scene_path, f"{label}{ext}"))

    append_csv(
        SCENE_LOG,
        ["scene", "capture", "ois_sharp", "ois_blur", "nonois_sharp", "nonois_blur"],
        [scene_name, capture_name, ois_sharp, ois_blur, nonois_sharp, nonois_blur]
    )
    return scene_name

def process_capture(capture_name, state):

    capture_path = os.path.join(DECODED_ROOT, capture_name)
    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")

    print(f"\n--- Processing {capture_name} ---")

    ois_results = score_frames(ois_dir, capture_name, "ois")
    nonois_results = score_frames(nonois_dir, capture_name, "nonois")

    if not ois_results or not nonois_results:
        return

    values = np.array([s for _, s in ois_results])

    sharp_score = np.max(values)
    blur_threshold = sharp_score * BLUR_THRESHOLD_RATIO

    # Detect blur segments
    blur_segments = []
    current = []

    for i, v in enumerate(values):

        if v < blur_threshold:
            current.append(i)

        else:
            if len(current) >= MIN_BLUR_FRAMES:
                blur_segments.append(current)
            current = []

    if len(current) >= MIN_BLUR_FRAMES:
        blur_segments.append(current)

    if not blur_segments:
        return

    # Sort segments by blur strength
    blur_segments.sort(key=lambda seg: np.mean([values[i] for i in seg]))

    top_segments = blur_segments[:MAX_SCENES_PER_VIDEO]

    scene_list = []

    for segment in top_segments:

        # Pick early blur frame (near motion start)
        blur_idx_ois = segment[2] if len(segment) > 2 else segment[0]

        # Find sharp frame BEFORE blur
        sharp_search_start = max(0, blur_idx_ois - 15)
        sharp_search_end = blur_idx_ois

        sharp_idx_ois = sharp_search_start + np.argmax(
            values[sharp_search_start:sharp_search_end]
        )

        sharp_f_ois = ois_results[sharp_idx_ois][0]

        # Find matching sharp in non-OIS
        s_start = max(0, sharp_idx_ois - SEARCH_WINDOW)
        s_end = min(len(nonois_results), sharp_idx_ois + SEARCH_WINDOW + 1)

        best_nonois_sharp_entry = max(
            nonois_results[s_start:s_end],
            key=lambda x: x[1]
        )

        sharp_f_nonois = best_nonois_sharp_entry[0]

        # Find blur match in non-OIS
        b_start = max(0, blur_idx_ois - SEARCH_WINDOW)
        b_end = min(len(nonois_results), blur_idx_ois + SEARCH_WINDOW + 1)

        worst_nonois_blur_entry = min(
            nonois_results[b_start:b_end],
            key=lambda x: x[1]
        )

        blur_f_ois = ois_results[blur_idx_ois][0]
        blur_f_nonois = worst_nonois_blur_entry[0]

        scene_name = build_scene(
            state["next_scene_id"],
            capture_name,
            sharp_f_ois,
            blur_f_ois,
            sharp_f_nonois,
            blur_f_nonois,
            ois_dir,
            nonois_dir
        )

        scene_list.append(scene_name)
        state["next_scene_id"] += 1

    if scene_list:
        state["processed_captures"].append(capture_name)
        print(f"Done! Created {len(scene_list)} scenes.")

def main():
    ensure_dirs()
    state = load_state()
    captures = sorted([d for d in os.listdir(DECODED_ROOT) if os.path.isdir(os.path.join(DECODED_ROOT, d))])
    
    # Progress bar for overall capture folders
    for capture in tqdm(captures, desc="Overall Progress"):
        if capture in state["processed_captures"]: continue
        process_capture(capture, state)
        save_state(state)

if __name__ == "__main__":
    main()