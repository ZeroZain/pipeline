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
import time
from tqdm import tqdm

# ================= CONFIG =================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")
LOG_ROOT = os.path.join(WORKSPACE_ROOT, "logs")
DEBUG_ROOT = os.path.join(WORKSPACE_ROOT, "debug")

DECODED_ROOT = os.path.join(DATA_ROOT, "decoded_frames")
DATASET_ROOT = os.path.join(DATA_ROOT, "dataset")

LAPLACIAN_LOG_DIR = os.path.join(LOG_ROOT, "laplacian")
SCENE_LOG = os.path.join(LOG_ROOT, "scene_selection_log.csv")
STATE_FILE = os.path.join(DATASET_ROOT, "dataset_state.json")

MAX_SCENES_PER_VIDEO = 2
BLUR_PERCENTILE = 30 # When determining the sharpness threshold for a video, use this percentile of the OIS sharpness scores. Adjust based on your data. 30 means using the 30th percentile, 20 means using the 20th percentile (more aggressive), 40 means using the 40th percentile (more conservative), etc.

# controls transition detection and frame pick ranges
SHARP_WINDOW = 10
BLUR_WINDOW = 4
BLUR_TO_SHARP_MAX_RATIO = 0.85 # OIS blur threshold ratio tuned to recover missed transitions in benchmark logs.
NONOIS_TRANSITION_RATIO = 0.9 # non-OIS often has shallower drop curves; use a looser transition gate.
SHARP_REFERENCE_PERCENTILE = 90 # P90 of the pre-drop window becomes the sharp baseline.
SHARP_PLATEAU_MIN_RATIO = 0.8 # Frames >= this ratio of P90 are considered sharp plateau support.
BLUR_REFERENCE_PERCENTILE = 20 # Select a representative blur frame near this percentile.
STABLE_CHECK = 8 # Confirm blur stays low after transition.
STABLE_MAX_OUTLIERS = 2 # Allow this many brief spike frames in the stable-check window.
STABLE_LOW_PERCENTILE = 80 # Require this percentile of the stable window to remain below blur limit.
OIS_IMMEDIATE_MAX_OUTLIERS = 1 # OIS immediate-drop gate: allow this many spikes in BLUR_WINDOW.
NONOIS_IMMEDIATE_MAX_OUTLIERS = 1 # non-OIS immediate-drop gate: allow this many spikes in BLUR_WINDOW.

# motion consistency checks between OIS and non-OIS pair
ENABLE_MOTION_CHECK = True
REQUIRE_MOTION_MATCH = True
MOTION_MIN_FEATURES = 40
MOTION_MIN_ABS_DX = 0.35
LINEAR_HORIZ_RATIO = 1.8
LINEAR_SIGN_SUPPORT = 0.65
LINEAR_MIN_COSINE = 0.55
NONLINEAR_MIN_COSINE = 0.35
NONLINEAR_LOW_MOTION_RELAX = 0.22
NONOIS_FALLBACK_BLUR_WINDOW = 12
NONOIS_FALLBACK_RATIO = 0.95

DEBUG_MODE = True
DEBUG_OUTPUT = os.path.join(DEBUG_ROOT, "laplacian")
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
    valid_files = [f for f in files if f.lower().endswith(('.dng', '.jpg', '.jpeg', '.png'))]
    if not valid_files:
        return []

    scores = []

    for f in tqdm(valid_files, desc=f"Scoring {capture_name}-{cam_type}", leave=False):
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

def find_transitions(values, blur_ratio=BLUR_TO_SHARP_MAX_RATIO, immediate_max_outliers=0):
    transitions = []

    for i in range(SHARP_WINDOW, len(values) - BLUR_WINDOW):
        sharp_window_values = values[i - SHARP_WINDOW:i]
        sharp_ref = float(np.percentile(sharp_window_values, SHARP_REFERENCE_PERCENTILE))

        sharp_floor = sharp_ref * SHARP_PLATEAU_MIN_RATIO
        sharp_support = sum(v >= sharp_floor for v in sharp_window_values)
        if sharp_support < max(5, int(0.6 * SHARP_WINDOW)):
            continue

        blur_limit = sharp_ref * blur_ratio
        immediate_blur = values[i:i + BLUR_WINDOW]
        # A spike on the first blur frame means the drop has not truly started yet.
        if immediate_blur and immediate_blur[0] > blur_limit:
            continue
        immediate_outliers = sum(v > blur_limit for v in immediate_blur)
        if immediate_outliers > immediate_max_outliers:
            continue

        future_end = min(len(values), i + STABLE_CHECK)
        future = values[i:future_end]
        stable_limit = blur_limit * 1.05
        outliers = sum(v > stable_limit for v in future)
        mostly_low = float(np.percentile(future, STABLE_LOW_PERCENTILE)) <= stable_limit

        if outliers <= STABLE_MAX_OUTLIERS and mostly_low:
            transitions.append(i)

    return transitions


def select_sharp_idx(values, indices):
    if not indices:
        return None, None

    seg_values = [values[i] for i in indices]
    sharp_ref = float(np.percentile(seg_values, SHARP_REFERENCE_PERCENTILE))
    sharp_floor = sharp_ref * SHARP_PLATEAU_MIN_RATIO
    candidates = [i for i in indices if values[i] >= sharp_floor]

    if not candidates:
        return None, sharp_ref

    sharp_idx = min(candidates, key=lambda i: (abs(values[i] - sharp_ref), -values[i]))
    return sharp_idx, sharp_ref


def select_blur_idx(values, indices):
    if not indices:
        return None

    seg_values = [values[i] for i in indices]
    blur_ref = float(np.percentile(seg_values, BLUR_REFERENCE_PERCENTILE))
    return min(indices, key=lambda i: abs(values[i] - blur_ref))


def estimate_motion_profile(sharp_img, blur_img):
    gray0 = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

    pts0 = cv2.goodFeaturesToTrack(
        gray0,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=7
    )

    if pts0 is None or len(pts0) < MOTION_MIN_FEATURES:
        return None

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts0, None)
    if pts1 is None or status is None:
        return None

    valid = status.ravel() == 1
    if np.count_nonzero(valid) < MOTION_MIN_FEATURES:
        return None

    p0 = pts0[valid].reshape(-1, 2)
    p1 = pts1[valid].reshape(-1, 2)
    flow = p1 - p0
    dx = flow[:, 0]
    dy = flow[:, 1]

    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))
    abs_med_dx = abs(med_dx)
    abs_med_dy = abs(med_dy)

    sign_support = float(max(np.mean(dx > 0), np.mean(dx < 0)))
    horiz_ratio = float(abs_med_dx / (abs_med_dy + 1e-6))
    magnitude = float(np.median(np.sqrt(dx * dx + dy * dy)))

    return {
        "median_dx": med_dx,
        "median_dy": med_dy,
        "abs_median_dx": abs_med_dx,
        "horiz_ratio": horiz_ratio,
        "sign_support": sign_support,
        "magnitude": magnitude
    }


def cosine_similarity_2d(ax, ay, bx, by):
    na = np.sqrt(ax * ax + ay * ay)
    nb = np.sqrt(bx * bx + by * by)

    if na < 1e-6 or nb < 1e-6:
        return 0.0

    return float((ax * bx + ay * by) / (na * nb))


def evaluate_motion_pair(ois_sharp_img, ois_blur_img, nonois_sharp_img, nonois_blur_img):
    if not ENABLE_MOTION_CHECK:
        return True, False

    ois_motion = estimate_motion_profile(ois_sharp_img, ois_blur_img)
    nonois_motion = estimate_motion_profile(nonois_sharp_img, nonois_blur_img)

    # If motion cannot be measured reliably, do not reject the pair.
    if ois_motion is None or nonois_motion is None:
        return True, False

    cos_sim = cosine_similarity_2d(
        ois_motion["median_dx"], ois_motion["median_dy"],
        nonois_motion["median_dx"], nonois_motion["median_dy"]
    )

    linear_like_ois = (
        ois_motion["abs_median_dx"] >= MOTION_MIN_ABS_DX and
        ois_motion["horiz_ratio"] >= LINEAR_HORIZ_RATIO and
        ois_motion["sign_support"] >= LINEAR_SIGN_SUPPORT
    )

    linear_like_nonois = (
        nonois_motion["abs_median_dx"] >= MOTION_MIN_ABS_DX and
        nonois_motion["horiz_ratio"] >= LINEAR_HORIZ_RATIO and
        nonois_motion["sign_support"] >= LINEAR_SIGN_SUPPORT
    )
    is_linear_scene = linear_like_ois and linear_like_nonois

    # Apply strict left/right constraints only when both streams are clearly linear.
    if is_linear_scene:
        same_dir = (ois_motion["median_dx"] * nonois_motion["median_dx"]) > 0
        return same_dir and cos_sim >= LINEAR_MIN_COSINE, True

    # Non-linear or mixed-motion case: keep a softer trajectory agreement check.
    if cos_sim >= NONLINEAR_MIN_COSINE:
        return True, False

    low_motion = min(ois_motion["magnitude"], nonois_motion["magnitude"]) < NONLINEAR_LOW_MOTION_RELAX
    return low_motion and cos_sim >= 0.0, False


def fallback_nonois_transition(ois_t, nonois_len):
    # If non-OIS transition detection misses the drop, align by OIS transition index.
    return min(max(0, ois_t), max(0, nonois_len - 1))

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
                ois_drop_frame, nonois_drop_frame,
                ois_drop_frame_actual, nonois_drop_frame_actual,
                nonois_used_fallback,
                is_linear_scene,
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
        ["scene", "method", "capture", "ois_sharp", "ois_blur",
            "nonois_sharp", "nonois_blur", "ois_drop_frame", "nonois_drop_frame",
            "ois_drop_frame_actual", "nonois_drop_frame_actual",
            "nonois_used_fallback", "is_linear_scene"],
        [
         to_posix(scene_rel_path),
         category_parts[0] if len(category_parts) > 0 else "",
         to_posix(capture_name),
         ois_sharp, ois_blur,
            nonois_sharp, nonois_blur,
                ois_drop_frame, nonois_drop_frame,
                ois_drop_frame_actual, nonois_drop_frame_actual,
                "true" if nonois_used_fallback else "false", "true" if is_linear_scene else "false"]
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

    # Track selected frame pairs to avoid duplicates within this capture
    selected_ois_frame_keys = set()
    selected_nonois_frame_keys = set()
    selected_nonois_blur_files = set()

    ois_values = [s for _, s in ois_results]
    nonois_values = [s for _, s in nonois_results]

    threshold_ois = np.percentile(ois_values, BLUR_PERCENTILE)
    threshold_nonois = np.percentile(nonois_values, BLUR_PERCENTILE)

    transitions = find_transitions(
        ois_values,
        BLUR_TO_SHARP_MAX_RATIO,
        OIS_IMMEDIATE_MAX_OUTLIERS
    )
    nonois_transitions = find_transitions(
        nonois_values,
        NONOIS_TRANSITION_RATIO,
        NONOIS_IMMEDIATE_MAX_OUTLIERS
    )
    nonois_actual_t = nonois_transitions[0] if nonois_transitions else None

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

        sharp_idx, sharp_ref_ois = select_sharp_idx(ois_values, sharp_seg)
        blur_idx = select_blur_idx(ois_values, blur_seg)

        if sharp_idx is None or blur_idx is None:
            continue

        # validation
        if ois_values[sharp_idx] <= threshold_ois:
            continue

        # Blur frame must be significantly blurrier than its paired sharp frame.
        if ois_values[blur_idx] > (sharp_ref_ois * BLUR_TO_SHARP_MAX_RATIO):
            continue

        # non-OIS matching: use independent transition detection (same approach as OIS)
        used_nonois_fallback = False
        if not nonois_transitions:
            used_nonois_fallback = True
            nonois_t = fallback_nonois_transition(t, len(nonois_values))
        else:
            nonois_t = min(nonois_transitions, key=lambda x: abs(x - t))

        sharp_start_nonois = max(0, nonois_t - SHARP_WINDOW)
        sharp_end_nonois = nonois_t

        blur_start_nonois = nonois_t
        blur_window_nonois = NONOIS_FALLBACK_BLUR_WINDOW if used_nonois_fallback else BLUR_WINDOW
        blur_end_nonois = min(len(nonois_values), nonois_t + blur_window_nonois)

        sharp_seg_nonois = list(range(sharp_start_nonois, sharp_end_nonois))
        blur_seg_nonois = list(range(blur_start_nonois, blur_end_nonois))

        if len(sharp_seg_nonois) < 5 or len(blur_seg_nonois) == 0:
            continue

        sharp_idx_nonois, sharp_ref_nonois = select_sharp_idx(nonois_values, sharp_seg_nonois)
        blur_idx_nonois = select_blur_idx(nonois_values, blur_seg_nonois)

        if sharp_idx_nonois is None or blur_idx_nonois is None:
            continue

        if nonois_values[sharp_idx_nonois] <= threshold_nonois:
            continue

        ratio_limit_nonois = NONOIS_FALLBACK_RATIO if used_nonois_fallback else NONOIS_TRANSITION_RATIO
        if nonois_values[blur_idx_nonois] > (sharp_ref_nonois * ratio_limit_nonois):
            continue

        sharp_file = ois_results[sharp_idx][0]
        blur_file = ois_results[blur_idx][0]
        nonois_sharp_file = nonois_results[sharp_idx_nonois][0]
        nonois_blur_file = nonois_results[blur_idx_nonois][0]

        ois_sharp_img = read_image(os.path.join(ois_dir, sharp_file))
        ois_blur_img = read_image(os.path.join(ois_dir, blur_file))
        nonois_sharp_img = read_image(os.path.join(nonois_dir, nonois_sharp_file))
        nonois_blur_img = read_image(os.path.join(nonois_dir, nonois_blur_file))

        if any(img is None for img in [ois_sharp_img, ois_blur_img, nonois_sharp_img, nonois_blur_img]):
            continue

        motion_pair_ok, is_linear_scene = evaluate_motion_pair(
            ois_sharp_img,
            ois_blur_img,
            nonois_sharp_img,
            nonois_blur_img
        )

        if REQUIRE_MOTION_MATCH and not motion_pair_ok:
            continue

        if DEBUG_MODE:
            visualize_selection(
                scene_id,
                ois_sharp_img,
                ois_blur_img,
                nonois_sharp_img,
                nonois_blur_img,
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
            nonois_sharp_file,
            nonois_blur_file,
            ois_results[t][0],
            nonois_results[nonois_t][0],
            ois_results[t][0],
            nonois_results[nonois_actual_t][0] if nonois_actual_t is not None else "",
            used_nonois_fallback,
            is_linear_scene,
            ois_dir,
            nonois_dir
        )

        ois_frame_key = (sharp_file, blur_file)
        nonois_frame_key = (nonois_sharp_file, nonois_blur_file)

        selected_ois_frame_keys.add(ois_frame_key)
        selected_nonois_frame_keys.add(nonois_frame_key)
        selected_nonois_blur_files.add(nonois_blur_file)
        scene_id += 1
        scene_count += 1

    # Force-fill pass: if strict transition logic yields fewer scenes than requested,
    # add extra unique scene pairs from high-sharpness regions.
    if scene_count < MAX_SCENES_PER_VIDEO:
        sharp_candidates = sorted(
            range(len(ois_values)),
            key=lambda i: ois_values[i],
            reverse=True,
        )

        for sharp_idx in sharp_candidates:
            if scene_count >= MAX_SCENES_PER_VIDEO:
                break

            if sharp_idx >= len(ois_values) - 1:
                continue

            blur_end = min(len(ois_values), sharp_idx + NONOIS_FALLBACK_BLUR_WINDOW + 1)
            blur_indices = list(range(sharp_idx + 1, blur_end))
            if not blur_indices:
                continue
            blur_idx = min(blur_indices, key=lambda i: ois_values[i])

            if len(ois_values) <= 1 or len(nonois_values) <= 1:
                break

            mapped = int(round((sharp_idx / (len(ois_values) - 1)) * (len(nonois_values) - 1)))
            half_window = max(1, SHARP_WINDOW // 2)
            sharp_start_nonois = max(0, mapped - half_window)
            sharp_end_nonois = min(len(nonois_values), mapped + half_window + 1)
            sharp_seg_nonois = list(range(sharp_start_nonois, sharp_end_nonois))
            if not sharp_seg_nonois:
                continue

            sharp_idx_nonois = max(sharp_seg_nonois, key=lambda i: nonois_values[i])
            if sharp_idx_nonois >= len(nonois_values) - 1:
                continue

            blur_end_nonois = min(
                len(nonois_values),
                sharp_idx_nonois + NONOIS_FALLBACK_BLUR_WINDOW + 1,
            )
            blur_seg_nonois = list(range(sharp_idx_nonois + 1, blur_end_nonois))
            if not blur_seg_nonois:
                continue

            blur_idx_nonois = min(blur_seg_nonois, key=lambda i: nonois_values[i])

            sharp_file = ois_results[sharp_idx][0]
            blur_file = ois_results[blur_idx][0]
            nonois_sharp_file = nonois_results[sharp_idx_nonois][0]
            nonois_blur_file = nonois_results[blur_idx_nonois][0]

            ois_frame_key = (
                sharp_file,
                blur_file,
            )
            nonois_frame_key = (
                nonois_sharp_file,
                nonois_blur_file,
            )
            if (
                ois_frame_key in selected_ois_frame_keys or
                nonois_frame_key in selected_nonois_frame_keys or
                nonois_blur_file in selected_nonois_blur_files
            ):
                continue

            ois_sharp_img = read_image(os.path.join(ois_dir, sharp_file))
            ois_blur_img = read_image(os.path.join(ois_dir, blur_file))
            nonois_sharp_img = read_image(os.path.join(nonois_dir, nonois_sharp_file))
            nonois_blur_img = read_image(os.path.join(nonois_dir, nonois_blur_file))

            if any(img is None for img in [ois_sharp_img, ois_blur_img, nonois_sharp_img, nonois_blur_img]):
                continue

            motion_pair_ok, is_linear_scene = evaluate_motion_pair(
                ois_sharp_img,
                ois_blur_img,
                nonois_sharp_img,
                nonois_blur_img
            )

            if REQUIRE_MOTION_MATCH and not motion_pair_ok:
                continue

            if DEBUG_MODE:
                visualize_selection(
                    scene_id,
                    ois_sharp_img,
                    ois_blur_img,
                    nonois_sharp_img,
                    nonois_blur_img,
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
                nonois_sharp_file,
                nonois_blur_file,
                ois_results[blur_idx][0],
                nonois_results[blur_idx_nonois][0],
                ois_results[blur_idx][0],
                nonois_results[blur_idx_nonois][0],
                True,
                is_linear_scene,
                ois_dir,
                nonois_dir
            )

            selected_ois_frame_keys.add(ois_frame_key)
            selected_nonois_frame_keys.add(nonois_frame_key)
            selected_nonois_blur_files.add(nonois_blur_file)
            scene_id += 1
            scene_count += 1

    if scene_count < MAX_SCENES_PER_VIDEO:
        pass

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
        os.path.join(os.path.dirname(__file__), "..", "Alignment", "align4.py")
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
        help="lap = run only this script, full = continue with align4.py then n256.py"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously and process new folders as they appear."
    )
    args = parser.parse_args()

    ensure_dirs()
    print(f"Laplacian script started (watch mode={'ON' if args.watch else 'OFF'})...")

    master_pbar = None
    
    try:
        while True:
            state = load_state()
            all_captures = list_captures(DECODED_ROOT)
            processed_captures = set(state["processed_captures"])
            scene_id = state["next_scene_id"]

            new_work = [c for c in all_captures if c not in processed_captures]

            # Initialize or update the master progress bar
            if args.watch:
                if master_pbar is None:
                    master_pbar = tqdm(total=len(all_captures), desc="Overall Progress", unit="cap")
                
                # Update total if Watcher added new files
                master_pbar.total = len(all_captures)
                # Ensure the current progress reflects the state file
                master_pbar.n = len(processed_captures)
                master_pbar.refresh()

            if new_work:
                if not args.watch:
                    print(f"\n[SCAN] Found {len(new_work)} captures to process.")
                    # In one-shot mode, use a standard batch bar
                    for capture in tqdm(new_work, desc="Processing"):
                        prev_scene_id = scene_id
                        scene_id = process_capture(capture, scene_id)

                        if scene_id >= prev_scene_id:
                            state["processed_captures"].append(capture)
                        
                        state["next_scene_id"] = scene_id
                        save_state(state)
                else:
                    # In watch mode, process one at a time and update the master bar
                    tqdm.write(f"\n[WATCH] Discovered {len(new_work)} new capture(s).")
                    for capture in new_work:
                        tqdm.write(f"[WATCH] Processing: {capture}")
                        prev_scene_id = scene_id
                        scene_id = process_capture(capture, scene_id)

                        if scene_id >= prev_scene_id:
                            state["processed_captures"].append(capture)
                        
                        state["next_scene_id"] = scene_id
                        save_state(state)
                        
                        # Update progress bar
                        master_pbar.n = len(state["processed_captures"])
                        master_pbar.refresh()

            elif not args.watch:
                print("\n[INFO] No new captures found. Exiting.")
                break

            if not args.watch:
                break

            time.sleep(5)

    except KeyboardInterrupt:
        if master_pbar:
            master_pbar.close()
        print("\nStopping Laplacian watcher...")

    if not args.watch:
        run_full_pipeline = (
            args.mode == "full"
            if args.mode is not None
            else ask_yes_no(
                "Run the full pipeline after Laplacian? (1lap.py -> align4.py -> n256.py)",
                default=False
            )
        )
        run_alignment_pipeline(run_full_pipeline)

if __name__ == "__main__":
    main()
