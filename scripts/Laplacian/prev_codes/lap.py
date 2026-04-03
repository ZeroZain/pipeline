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

<<<<<<< Updated upstream
MAX_SCENES_PER_VIDEO = 1 # Maximum number of scenes to extract from each video. Adjust based on how many scenes you want per video and how many videos you have. 1 means extracting 1 scene per video, 2 means extracting 2 scenes per video, etc.
BLUR_PERCENTILE = 30 # When determining the sharpness threshold for a video, use this percentile of the OIS sharpness scores. Adjust based on your data. 30 means using the 30th percentile, 20 means using the 20th percentile (more aggressive), 40 means using the 40th percentile (more conservative), etc.

# controls transition detection and frame pick ranges
<<<<<<<< Updated upstream:scripts/Laplacian/prev_codes/lap.py
SHARP_WINDOW = 10
BLUR_WINDOW = 4
========
SHARP_WINDOW = 10 # When looking for sharp frames, consider this many frames before the detected transition. Adjust based on how long you expect the sharp segment to last and how quickly the sharpness rises. 20 means looking at 20 frames before, 10 means looking at 10 frames, etc.
BLUR_WINDOW = 4 # When looking for sharp frames, consider this many frames before the detected transition. When looking for blur frames, consider this many frames after the detected transition. Adjust based on how long you expect the blur to last and how quickly the sharpness drops. 10 means looking at 10 frames before/after, 5 means looking at 5 frames, etc.
>>>>>>>> Stashed changes:scripts/Laplacian/lap.py
BLUR_TO_SHARP_MAX_RATIO = 0.8 # A blur frame must have a sharpness score no more than this ratio of its paired sharp frame to be considered valid. Adjust based on your data. 0.7 means the blur frame can be at most 70% as sharp as the sharp frame, 0.5 means it can be at most 50% as sharp, etc.

DROP_RATIO = 1.1 # A frame is considered a transition if the sharpness drops by at least this factor compared to the previous frame. Adjust based on your data. 1.1 means a 10% drop, 1.2 means a 20% drop, etc.
STABLE_CHECK = 10 # After detecting a potential transition, check the next STABLE_CHECK frames to ensure they remain blurry (i.e., their sharpness does not rise back above the threshold). Adjust based on how long you expect the blur to last. 3 means checking the next 3 frames, 5 means checking the next 5 frames, etc.
=======
MAX_SCENES_PER_VIDEO = 1 
BLUR_PERCENTILE = 30 # When determining the sharpness threshold for a video, use this percentile of the OIS sharpness scores. Adjust based on your data. 30 means using the 30th percentile, 20 means using the 20th percentile (more aggressive), 40 means using the 40th percentile (more conservative), etc.
SEARCH_WINDOW = 5 # When matching sharp and blurry frames between OIS and non-OIS, only search within this window size around the detected indices. Adjust based on how closely the two cameras are synchronized. 5 means searching 5 frames before and after, 10 means searching 10 frames before and after, etc.

# controls transition detection and frame pick ranges
SHARP_WINDOW = 10
BLUR_WINDOW = 4
BLUR_TO_SHARP_MAX_RATIO = 0.8 # A blur frame must have a sharpness score no more than this ratio of its paired sharp frame to be considered valid. Adjust based on your data. 0.7 means the blur frame can be at most 70% as sharp as the sharp frame, 0.5 means it can be at most 50% as sharp, etc.

DROP_RATIO = 1.1 # A frame is considered a transition if the sharpness drops by at least this factor compared to the previous frame. Adjust based on your data. 1.1 means a 10% drop, 1.2 means a 20% drop, etc.
STABLE_CHECK = 3 # After detecting a potential transition, check the next STABLE_CHECK frames to ensure they remain blurry (i.e., their sharpness does not rise back above the threshold). Adjust based on how long you expect the blur to last. 3 means checking the next 3 frames, 5 means checking the next 5 frames, etc.
>>>>>>> Stashed changes

DEBUG_MODE = True
DEBUG_OUTPUT = "debug_selection"
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
<<<<<<< Updated upstream
                rel_path = os.path.normpath(os.path.relpath(full_path, root))
=======
                rel_path = os.path.relpath(full_path, root)
>>>>>>> Stashed changes
                captures.append(rel_path)

        dirs[:] = [name for name in dirs if not name.startswith("capture_")]

    return sorted(captures, key=lambda path: path.lower())


<<<<<<< Updated upstream
def capture_categories(capture_rel_path):
    return os.path.normpath(capture_rel_path).split(os.sep)[:-1]


def category_key(category_parts):
    return to_posix(os.path.join(*category_parts)) if category_parts else "."


def capture_category_key(capture_rel_path):
    return category_key(capture_categories(capture_rel_path))


def scene_name(scene_id):
    return f"scene_{scene_id:03d}"


def scene_rel_path(capture_name, scene_id):
    category_parts = capture_categories(capture_name)
    name = scene_name(scene_id)
    return os.path.join(*category_parts, name) if category_parts else name


def next_scene_ids_from_dataset():
    next_ids = {}

    if not os.path.exists(DATASET_ROOT):
        return next_ids

    for current_root, dirs, _ in os.walk(DATASET_ROOT):
        dirs.sort()

        scene_ids = []
        child_dirs = []

        for name in dirs:
            match = SCENE_PATTERN.match(name)
            if match:
                scene_ids.append(int(match.group(1)))
            else:
                child_dirs.append(name)

        if scene_ids:
            rel_root = os.path.relpath(current_root, DATASET_ROOT)
            key = "." if rel_root == "." else to_posix(rel_root)
            next_ids[key] = max(scene_ids) + 1

        dirs[:] = child_dirs

    return next_ids
=======
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
>>>>>>> Stashed changes

# ================= STATE =================

def load_state():
<<<<<<< Updated upstream
    discovered_next_scene_ids = next_scene_ids_from_dataset()
=======
    discovered_next_scene_id = next_scene_id_from_dataset()
>>>>>>> Stashed changes

    if not os.path.exists(STATE_FILE):
        return {
            "processed_captures": [],
<<<<<<< Updated upstream
            "next_scene_ids": discovered_next_scene_ids
        }

=======
            "next_scene_id": discovered_next_scene_id
        }
>>>>>>> Stashed changes
    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    state.setdefault("processed_captures", [])
<<<<<<< Updated upstream
    merged_next_scene_ids = dict(discovered_next_scene_ids)

    for key, value in state.get("next_scene_ids", {}).items():
        if isinstance(value, int) and value > 0:
            merged_next_scene_ids[key] = max(merged_next_scene_ids.get(key, 1), value)

    state["next_scene_ids"] = merged_next_scene_ids
    state.pop("next_scene_id", None)
=======
    state["next_scene_id"] = max(state.get("next_scene_id", 1), discovered_next_scene_id)
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
def score_frames(folder, capture_name, cam_type, capture_label=None):
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")
    label = capture_label or capture_name
=======
def score_frames(folder, capture_name, cam_type):
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    for f in tqdm(files, desc=f"Scoring {label}-{cam_type}", leave=False):
=======
    for f in tqdm(files, desc=f"Scoring {capture_name}-{cam_type}", leave=False):
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
def visualize_selection(scene_rel,
=======
def visualize_selection(scene_id,
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    out_path = os.path.join(DEBUG_OUTPUT, f"{slugify_path(scene_rel)}.jpg")
=======
    out_path = os.path.join(DEBUG_OUTPUT, f"scene_{scene_id:03d}.jpg")
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    rel_path = scene_rel_path(capture_name, scene_id)
    scene_path = os.path.join(DATASET_ROOT, rel_path)
=======
    scene_name = f"scene_{scene_id:03d}"
    category_parts = capture_categories(capture_name)
    scene_rel_path = os.path.join(*category_parts, scene_name) if category_parts else scene_name
    scene_path = os.path.join(DATASET_ROOT, scene_rel_path)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        ["scene", "capture", "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [to_posix(rel_path), to_posix(capture_name),
=======
        ["scene", "split", "method", "capture", "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [
         to_posix(scene_rel_path),
         category_parts[0] if len(category_parts) > 0 else "",
         category_parts[1] if len(category_parts) > 1 else "",
         to_posix(capture_name),
>>>>>>> Stashed changes
         ois_sharp, ois_blur,
         nonois_sharp, nonois_blur]
    )

# ================= PROCESS =================

def process_capture(capture_name, scene_id):

    capture_path = os.path.join(DECODED_ROOT, capture_name)
    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")
    capture_key = slugify_path(capture_name)

<<<<<<< Updated upstream
    ois_results = score_frames(ois_dir, capture_key, "ois", capture_name)
    nonois_results = score_frames(nonois_dir, capture_key, "nonois", capture_name)
=======
    ois_results = score_frames(ois_dir, capture_key, "ois")
    nonois_results = score_frames(nonois_dir, capture_key, "nonois")
>>>>>>> Stashed changes

    if not ois_results or not nonois_results:
        return scene_id

    ois_values = [s for _, s in ois_results]
    nonois_values = [s for _, s in nonois_results]

    threshold_ois = np.percentile(ois_values, BLUR_PERCENTILE)
<<<<<<< Updated upstream
    threshold_nonois = np.percentile(nonois_values, BLUR_PERCENTILE)

    transitions = find_transitions(ois_values)
    nonois_transitions = find_transitions(nonois_values)
=======

    transitions = find_transitions(ois_values)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
<<<<<<<< Updated upstream:scripts/Laplacian/prev_codes/lap.py
        # non-OIS matching: use independent transition detection (same approach as OIS)
========
        # non-OIS matching uses non-OIS transition detection (same approach as OIS)
>>>>>>>> Stashed changes:scripts/Laplacian/lap.py
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
        rel_path = scene_rel_path(capture_name, scene_id)

        if DEBUG_MODE:
            visualize_selection(
                rel_path,
=======
        # non-OIS matching
        sharp_idx_nonois = min(range(len(nonois_values)), key=lambda x: abs(x - sharp_idx))
        blur_idx_nonois = min(range(len(nonois_values)), key=lambda x: abs(x - blur_idx))

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
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    next_scene_ids = state["next_scene_ids"]
=======
    scene_id = state["next_scene_id"]
>>>>>>> Stashed changes

    for capture in tqdm(captures, desc="Processing"):

        if capture in processed_captures:
            tqdm.write(f"Skipping (already processed): {capture}")
            continue

<<<<<<< Updated upstream
        category = capture_category_key(capture)
        scene_id = next_scene_ids.get(category, 1)
=======
>>>>>>> Stashed changes
        prev_scene_id = scene_id
        scene_id = process_capture(capture, scene_id)

        if scene_id > prev_scene_id:
            state["processed_captures"].append(capture)
            processed_captures.add(capture)

<<<<<<< Updated upstream
        next_scene_ids[category] = scene_id
=======
        state["next_scene_id"] = scene_id
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    main()
=======
    main()
>>>>>>> Stashed changes
