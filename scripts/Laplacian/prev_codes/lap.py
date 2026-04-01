import os
import cv2
import csv
import json
import shutil
import argparse
import subprocess
import sys
import re
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

MAX_SCENES_PER_VIDEO = 1 
BLUR_PERCENTILE = 30 # When determining the sharpness threshold for a video, use this percentile of the OIS sharpness scores. Adjust based on your data. 30 means using the 30th percentile, 20 means using the 20th percentile (more aggressive), 40 means using the 40th percentile (more conservative), etc.

# controls transition detection and frame pick ranges
SHARP_WINDOW = 10
BLUR_WINDOW = 4
BLUR_TO_SHARP_MAX_RATIO = 0.8 # A blur frame must have a sharpness score no more than this ratio of its paired sharp frame to be considered valid. Adjust based on your data. 0.7 means the blur frame can be at most 70% as sharp as the sharp frame, 0.5 means it can be at most 50% as sharp, etc.

DROP_RATIO = 1.1 # A frame is considered a transition if the sharpness drops by at least this factor compared to the previous frame. Adjust based on your data. 1.1 means a 10% drop, 1.2 means a 20% drop, etc.
STABLE_CHECK = 10 # After detecting a potential transition, check the next STABLE_CHECK frames to ensure they remain blurry (i.e., their sharpness does not rise back above the threshold). Adjust based on how long you expect the blur to last. 3 means checking the next 3 frames, 5 means checking the next 5 frames, etc.

DEBUG_MODE = True
DEBUG_OUTPUT = "debug_vis"
SCENE_PATTERN = re.compile(r"^scene_(\d+)$")

# ================= SETUP =================

def ensure_dirs():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(LAPLACIAN_LOG_DIR, exist_ok=True)
    if DEBUG_MODE:
        os.makedirs(DEBUG_OUTPUT, exist_ok=True)


def to_posix(path):
    return path.replace(os.sep, "/")


def slugify_path(path):
    parts = []

    for part in os.path.normpath(path).split(os.sep):
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_")
        parts.append(clean or "item")

    return "__".join(parts)


def list_captures(root):
    captures = []

    if not os.path.exists(root):
        return captures

    for current_root, dirs, _ in os.walk(root):
        dirs.sort()

        for name in dirs:
            if name.startswith("capture_"):
                full_path = os.path.join(current_root, name)
                rel_path = os.path.relpath(full_path, root)
                captures.append(rel_path)

        dirs[:] = [name for name in dirs if not name.startswith("capture_")]

    return sorted(captures, key=lambda path: path.lower())


def next_scene_id_from_dataset():
    max_scene_id = 0

    if not os.path.exists(DATASET_ROOT):
        return 1

    for _, dirs, _ in os.walk(DATASET_ROOT):
        for name in dirs:
            match = SCENE_PATTERN.match(name)
            if match:
                max_scene_id = max(max_scene_id, int(match.group(1)))

    return max_scene_id + 1


def capture_categories(capture_rel_path):
    return os.path.normpath(capture_rel_path).split(os.sep)[:-1]

# ================= STATE =================

def load_state():
    discovered_next_scene_id = next_scene_id_from_dataset()

    if not os.path.exists(STATE_FILE):
        return {
            "processed_captures": [],
            "next_scene_id": discovered_next_scene_id
        }
    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    state.setdefault("processed_captures", [])
    state["next_scene_id"] = max(state.get("next_scene_id", 1), discovered_next_scene_id)
    return state

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

# ================= TRANSITION DETECTION =================

def find_transitions(values):
    transitions = []

    for i in range(1, len(values) - STABLE_CHECK):

        prev = values[i - 1]
        curr = values[i]

        if prev > curr:
            ratio = prev / (curr + 1e-6)

            if ratio >= DROP_RATIO:

                future = values[i:i + STABLE_CHECK]

                if all(v <= curr * 1.05 for v in future):
                    transitions.append(i)

    return transitions

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
    category_parts = capture_categories(capture_name)
    scene_rel_path = os.path.join(*category_parts, scene_name) if category_parts else scene_name
    scene_path = os.path.join(DATASET_ROOT, scene_rel_path)
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
        ["scene", "split", "method", "capture", "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [
         to_posix(scene_rel_path),
         category_parts[0] if len(category_parts) > 0 else "",
         category_parts[1] if len(category_parts) > 1 else "",
         to_posix(capture_name),
         ois_sharp, ois_blur,
         nonois_sharp, nonois_blur]
    )

# ================= PROCESS =================

def process_capture(capture_name, scene_id):

    capture_path = os.path.join(DECODED_ROOT, capture_name)
    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")
    capture_key = slugify_path(capture_name)

    ois_results = score_frames(ois_dir, capture_key, "ois")
    nonois_results = score_frames(nonois_dir, capture_key, "nonois")

    if not ois_results or not nonois_results:
        return scene_id

    ois_values = [s for _, s in ois_results]
    nonois_values = [s for _, s in nonois_results]

    threshold_ois = np.percentile(ois_values, BLUR_PERCENTILE)
    threshold_nonois = np.percentile(nonois_values, BLUR_PERCENTILE)

    transitions = find_transitions(ois_values)
    nonois_transitions = find_transitions(nonois_values)

    scene_count = 0

    for t in transitions:

        if scene_count >= MAX_SCENES_PER_VIDEO:
            break

        sharp_start = max(0, t - SHARP_WINDOW)
        sharp_end = t

        blur_start = t
        blur_end = min(len(ois_values), t + BLUR_WINDOW)

        sharp_seg = list(range(sharp_start, sharp_end))
        blur_seg = list(range(blur_start, blur_end))

        if len(sharp_seg) < 5 or len(blur_seg) == 0:
            continue

        sharp_idx = max(sharp_seg, key=lambda x: ois_values[x])
        # Blur is chosen only within its own post-drop window.
        blur_idx = min(blur_seg, key=lambda x: ois_values[x])

        # validation
        if ois_values[sharp_idx] <= threshold_ois:
            continue

        # Blur frame must be significantly blurrier than its paired sharp frame.
        if ois_values[blur_idx] > (ois_values[sharp_idx] * BLUR_TO_SHARP_MAX_RATIO):
            continue

        # non-OIS matching: use independent transition detection (same approach as OIS)
        if not nonois_transitions:
            continue

        nonois_t = min(nonois_transitions, key=lambda x: abs(x - t))

        sharp_start_nonois = max(0, nonois_t - SHARP_WINDOW)
        sharp_end_nonois = nonois_t

        blur_start_nonois = nonois_t
        blur_end_nonois = min(len(nonois_values), nonois_t + BLUR_WINDOW)

        sharp_seg_nonois = list(range(sharp_start_nonois, sharp_end_nonois))
        blur_seg_nonois = list(range(blur_start_nonois, blur_end_nonois))

        if len(sharp_seg_nonois) < 5 or len(blur_seg_nonois) == 0:
            continue

        sharp_idx_nonois = max(sharp_seg_nonois, key=lambda x: nonois_values[x])
        blur_idx_nonois = min(blur_seg_nonois, key=lambda x: nonois_values[x])

        if nonois_values[sharp_idx_nonois] <= threshold_nonois:
            continue

        if nonois_values[blur_idx_nonois] > (nonois_values[sharp_idx_nonois] * BLUR_TO_SHARP_MAX_RATIO):
            continue

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


def ask_yes_no(prompt, default=False):
    default_hint = "Y/n" if default else "y/N"

    while True:
        answer = input(f"{prompt} [{default_hint}]: ").strip().lower()

        if not answer:
            return default

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please answer with 'y' or 'n'.")


def run_alignment_pipeline(run_full_pipeline):
    if not run_full_pipeline:
        print("Laplacian stage complete. Alignment and interpolation were not started.")
        return

    align_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Alignment", "align2.py")
    )

    command = [sys.executable, align_script, "--run-next"]

    print("Laplacian stage complete. Starting alignment + interpolation...")
    subprocess.run(command, check=True)

# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(
        description="Select scenes using Laplacian sharpness and optionally continue the full pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["lap", "full"],
        help="lap = run only this script, full = continue with align2.py then 256.py"
    )
    args = parser.parse_args()

    ensure_dirs()
    state = load_state()

    captures = list_captures(DECODED_ROOT)
    processed_captures = set(state["processed_captures"])
    scene_id = state["next_scene_id"]

    for capture in tqdm(captures, desc="Processing"):

        if capture in processed_captures:
            tqdm.write(f"Skipping (already processed): {capture}")
            continue

        prev_scene_id = scene_id
        scene_id = process_capture(capture, scene_id)

        if scene_id > prev_scene_id:
            state["processed_captures"].append(capture)
            processed_captures.add(capture)

        state["next_scene_id"] = scene_id
        save_state(state)

    run_full_pipeline = (
        args.mode == "full"
        if args.mode is not None
        else ask_yes_no(
            "Run the full pipeline after Laplacian? (lap.py -> align2.py -> 256.py)",
            default=False
        )
    )

    run_alignment_pipeline(run_full_pipeline)

if __name__ == "__main__":
    main()