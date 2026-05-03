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
BLUR_WINDOW = 8
BLUR_TO_SHARP_MAX_RATIO = 0.85 # OIS blur threshold ratio tuned to recover missed transitions in benchmark logs.
NONOIS_TRANSITION_RATIO = 0.9 # non-OIS often has shallower drop curves; use a looser transition gate.
SHARP_REFERENCE_PERCENTILE = 90 # P90 of the pre-drop window becomes the sharp baseline.
SHARP_PLATEAU_MIN_RATIO = 0.8 # Frames >= this ratio of P90 are considered sharp plateau support.
BLUR_REFERENCE_PERCENTILE = 10 # Select a representative blur frame near this percentile.
MIN_SHARP_BLUR_GAP = 3 # Require at least this many frames between selected sharp and blur frames.
MIN_DROP_RATIO = 0.16 # Require blur to be at least 16% lower than sharp to avoid near-equal pairs.
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
LINEAR_HORIZ_RATIO = 2.1
LINEAR_SIGN_SUPPORT = 0.75
LINEAR_MIN_COSINE = 0.55
NONLINEAR_MIN_COSINE = 0.35
NONLINEAR_LOW_MOTION_RELAX = 0.22
NONOIS_FALLBACK_BLUR_WINDOW = 12
NONOIS_FALLBACK_RATIO = 0.95

DEBUG_MODE = True
DEBUG_OUTPUT = os.path.join(DEBUG_ROOT, "laplacian")
SCENE_PATTERN = re.compile(r"^scene_(\d+)$")
REUSE_SHARP_REFERENCE = True  # If True, prefer a single sharp reference reused across scenes
FORCE_REUSE_SAME_SHARP = True  # In force-fill, prioritize reusing one sharp frame while varying blur picks.
FORCE_GUARANTEE_SCENES = True  # Ensure fallback can still emit scenes when strict quality gates reject extra pairs.
MIN_NEXT_BLUR_STEP = 2  # Require at least one-frame gap between consecutive selected blur frames.
INITIAL_SHARP_SKIP = SHARP_WINDOW  # Avoid selecting sharp references from the unstable initial frames.
CROSS_STREAM_SHARP_CAP_MARGIN = 0  # Keep sharp picks strictly before mapped blur onset from the other stream.
ONSET_ALIGN_TOLERANCE = SHARP_WINDOW  # If OIS onset lags this much behind mapped non-OIS onset, inject aligned onset.
ALIGNED_ONSET_BLUR_WINDOW = 12  # Search a bit wider blur region when using aligned onset fallback.
ALIGNED_ONSET_RATIO_LIMIT = 0.98  # Relax OIS ratio gate only for aligned-onset fallback.
ALIGNED_ONSET_MIN_DROP_RATIO = 0.03  # Relax min drop only for aligned-onset fallback.
ALIGNED_ONSET_SKIP_MOTION_CHECK = True  # Allow scene creation when aligned-onset motion vectors disagree.

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
        try:
            state = json.load(f)
        except (json.JSONDecodeError, ValueError):
            # Backup the corrupt/empty state file and return a fresh state
            bak = STATE_FILE + ".corrupt." + time.strftime("%Y%m%d%H%M%S")
            try:
                shutil.copy(STATE_FILE, bak)
                print(f"Warning: corrupted STATE_FILE backed up to {bak}")
            except Exception:
                print(f"Warning: failed to back up corrupted STATE_FILE: {STATE_FILE}")
            return {
                "processed_captures": [],
                "next_scene_id": discovered_next_scene_id
            }

    if not isinstance(state, dict):
        return {
            "processed_captures": [],
            "next_scene_id": discovered_next_scene_id
        }

    state.setdefault("processed_captures", [])
    state["next_scene_id"] = max(state.get("next_scene_id", 1), discovered_next_scene_id)
    return state

def save_state(state):
    # Write state atomically to avoid corruption when interrupted
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)

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

def classify_direction(motion):
    if motion is None:
        return "none"

    if motion["abs_median_dx"] < MOTION_MIN_ABS_DX:
        return "none"

    linear_like = (
        motion["horiz_ratio"] >= LINEAR_HORIZ_RATIO and
        motion["sign_support"] >= LINEAR_SIGN_SUPPORT
    )

    if not linear_like:
        return "none"

    return "right" if motion["median_dx"] > 0 else "left"

def motion_direction_between(prev_img, curr_img):
    if prev_img is None or curr_img is None:
        return "none"

    motion = estimate_motion_profile(prev_img, curr_img)
    if motion is None:
        return "none"

    return classify_direction(motion)

# ================= FRAME SCORING =================

def score_frames(folder, capture_name, cam_type):
    log_file = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{cam_type}.csv")
    is_sliding = capture_name.lower().startswith("sliding_method")

    if os.path.exists(log_file):
        scores = []
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Ensure the structure matches the new rules
            has_direction = "direction" in header
            if has_direction == is_sliding:
                for row in reader:
                    scores.append((row[0], float(row[1])))
                return scores
            else:
                # Structure mismatch (e.g., old file has direction for non-sliding), rebuild it
                pass

    if not os.path.exists(folder):
        return []

    files = sorted(os.listdir(folder))
    valid_files = [f for f in files if f.lower().endswith(('.dng', '.jpg', '.jpeg', '.png'))]
    if not valid_files:
        return []

    scores = []
    directions = []
    prev_img = None

    for f in tqdm(valid_files, desc=f"Scoring {capture_name}-{cam_type}", leave=False):
        img = read_image(os.path.join(folder, f))
        if img is None:
            continue
        scores.append((f, laplacian_score(img)))
        
        if is_sliding:
            directions.append(motion_direction_between(prev_img, img))
            prev_img = img

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        if is_sliding:
            writer.writerow(["frame", "score", "direction"])
            for score, direction in zip(scores, directions):
                writer.writerow([score[0], score[1], direction])
        else:
            writer.writerow(["frame", "score"])
            for score in scores:
                writer.writerow([score[0], score[1]])

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


def has_sharp_context(values, idx):
    if idx is None or idx < 0 or idx >= len(values):
        return False

    start = max(0, idx - SHARP_WINDOW + 1)
    window = values[start:idx + 1]
    if len(window) < 5:
        return False

    sharp_ref = float(np.percentile(window, SHARP_REFERENCE_PERCENTILE))
    sharp_floor = sharp_ref * SHARP_PLATEAU_MIN_RATIO
    support = sum(v >= sharp_floor for v in window)
    min_support = max(5, int(0.6 * len(window)))
    return support >= min_support and values[idx] >= sharp_floor


def collect_sharp_candidates(values, transitions, max_idx_exclusive=None):
    if not values:
        return []

    min_idx = min(INITIAL_SHARP_SKIP, len(values))
    upper = len(values) if max_idx_exclusive is None else max(min(len(values), int(max_idx_exclusive)), min_idx)
    if upper <= min_idx:
        return []

    if transitions:
        allowed = set()
        for t in transitions:
            start = max(min_idx, t - SHARP_WINDOW)
            end = min(upper, t)
            if start < end:
                allowed.update(range(start, end))
        if allowed:
            base = sorted(allowed)
        else:
            base = list(range(min_idx, upper))
    else:
        base = list(range(min_idx, upper))

    return [i for i in base if has_sharp_context(values, i)]


def filter_local_sharp_segment(values, indices):
    if not indices:
        return []

    return [
        i for i in indices
        if i >= INITIAL_SHARP_SKIP and has_sharp_context(values, i)
    ]


def pick_transition_anchor(transitions, sharp_idx):
    if not transitions:
        return None

    future = [t for t in transitions if t >= sharp_idx]
    if future:
        return future[0]

    return min(transitions, key=lambda t: abs(t - sharp_idx))


def build_blur_segment(values_len, sharp_idx, anchor_idx, half_window, fallback_window):
    start_min = sharp_idx + MIN_SHARP_BLUR_GAP
    if start_min >= values_len:
        return []

    if anchor_idx is not None:
        start = max(start_min, anchor_idx - half_window)
        end = min(values_len, anchor_idx + half_window + 1)
        if start < end:
            return list(range(start, end))

    end = min(values_len, sharp_idx + fallback_window + 1)
    return list(range(start_min, end))


def valid_sharp_blur_pair(values, sharp_idx, blur_idx, ratio_limit, min_drop_ratio=MIN_DROP_RATIO):
    if sharp_idx is None or blur_idx is None:
        return False

    if blur_idx - sharp_idx < MIN_SHARP_BLUR_GAP:
        return False

    sharp_val = float(values[sharp_idx])
    blur_val = float(values[blur_idx])

    if sharp_val <= 1e-6:
        return False

    if blur_val > sharp_val * ratio_limit:
        return False

    drop_ratio = (sharp_val - blur_val) / sharp_val
    return drop_ratio >= min_drop_ratio


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
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # If the file does not exist, write header + row
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
        return

    # If file exists, check for duplicate row and only append if new
    try:
        with open(file, "r", newline="") as f:
            reader = csv.reader(f)
            existing = [r for r in reader]
    except Exception:
        existing = []

    # Normalize row to strings for comparison; compare against any existing data row
    row_str = [str(x) for x in row]
    for r in existing:
        if [str(x) for x in r] == row_str:
            return

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
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

    # Keep blur frames unique across scenes; sharp frames may be reused.
    selected_ois_blur_files = set()
    selected_nonois_blur_files = set()
    selected_ois_blur_indices = set()
    selected_nonois_blur_indices = set()

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
    ois_first_t = transitions[0] if transitions else None
    nonois_first_t = nonois_transitions[0] if nonois_transitions else None
    mapped_nonois_first_t_for_ois = None
    if (
        nonois_first_t is not None and
        len(nonois_values) > 1 and
        len(ois_values) > 1
    ):
        mapped_nonois_first_t_for_ois = int(round(
            (nonois_first_t / (len(nonois_values) - 1)) * (len(ois_values) - 1)
        ))

    # If OIS transition detection is clearly late, align with mapped non-OIS onset
    # so sharp/blur selection can still produce scenes around the true blur start.
    if mapped_nonois_first_t_for_ois is not None:
        if not transitions:
            transitions = [mapped_nonois_first_t_for_ois]
        elif mapped_nonois_first_t_for_ois + ONSET_ALIGN_TOLERANCE < transitions[0]:
            transitions = sorted(set([mapped_nonois_first_t_for_ois] + transitions))

    ois_sharp_cap = None
    if ois_first_t is not None and mapped_nonois_first_t_for_ois is not None:
        ois_sharp_cap = min(ois_first_t, mapped_nonois_first_t_for_ois) - CROSS_STREAM_SHARP_CAP_MARGIN
    elif ois_first_t is not None:
        ois_sharp_cap = ois_first_t
    elif mapped_nonois_first_t_for_ois is not None:
        ois_sharp_cap = mapped_nonois_first_t_for_ois

    nonois_actual_t = nonois_transitions[0] if nonois_transitions else None

    scene_count = 0
    sharp_candidates = collect_sharp_candidates(ois_values, transitions, ois_sharp_cap)

    # Optionally compute a single sharp reference for the capture and reuse it
    global_sharp_idx = None
    global_sharp_ref = None
    if REUSE_SHARP_REFERENCE:
        if sharp_candidates:
            gs_idx, gs_ref = select_sharp_idx(ois_values, sharp_candidates)
            # only accept global sharp if it meets the threshold
            if gs_idx is not None and ois_values[gs_idx] > threshold_ois:
                global_sharp_idx = gs_idx
                global_sharp_ref = gs_ref

    for t in transitions:

        if scene_count >= MAX_SCENES_PER_VIDEO:
            break

        sharp_start = max(0, t - SHARP_WINDOW)
        sharp_end = t

        blur_start = t
        is_aligned_onset_t = (
            mapped_nonois_first_t_for_ois is not None and
            t == mapped_nonois_first_t_for_ois
        )
        blur_window_ois = ALIGNED_ONSET_BLUR_WINDOW if is_aligned_onset_t else BLUR_WINDOW
        blur_end = min(len(ois_values), t + blur_window_ois)

        sharp_seg = list(range(sharp_start, sharp_end))
        blur_seg = list(range(blur_start, blur_end))

        if len(sharp_seg) < 5 or len(blur_seg) == 0:
            continue

        # Prefer reusing a global sharp reference when available, otherwise pick locally
        if REUSE_SHARP_REFERENCE and global_sharp_idx is not None:
            sharp_idx = global_sharp_idx
            sharp_ref_ois = global_sharp_ref
            # if the global sharp falls inside or too close to the blur segment, fall back
            if not (sharp_idx < blur_start and (blur_start - sharp_idx) >= MIN_SHARP_BLUR_GAP):
                local_candidates = filter_local_sharp_segment(ois_values, sharp_seg)
                sharp_idx, sharp_ref_ois = select_sharp_idx(ois_values, local_candidates)
        else:
            local_candidates = filter_local_sharp_segment(ois_values, sharp_seg)
            sharp_idx, sharp_ref_ois = select_sharp_idx(ois_values, local_candidates)

        if is_aligned_onset_t:
            blur_idx = min(blur_seg, key=lambda i: ois_values[i]) if blur_seg else None
        else:
            blur_idx = select_blur_idx(ois_values, blur_seg)

        # Safety: ensure sharp and blur are taken from their intended segments and are distinct
        if sharp_idx is None or blur_idx is None:
            continue
        if sharp_idx in blur_seg or blur_idx in sharp_seg or sharp_idx == blur_idx:
            continue

        if sharp_idx is None or blur_idx is None:
            continue

        # validation
        if ois_values[sharp_idx] <= threshold_ois:
            continue

        # Blur frame must be sufficiently later and significantly blurrier than sharp.
        ois_ratio_limit = ALIGNED_ONSET_RATIO_LIMIT if is_aligned_onset_t else BLUR_TO_SHARP_MAX_RATIO
        ois_min_drop = ALIGNED_ONSET_MIN_DROP_RATIO if is_aligned_onset_t else MIN_DROP_RATIO
        if not valid_sharp_blur_pair(
            ois_values,
            sharp_idx,
            blur_idx,
            ois_ratio_limit,
            min_drop_ratio=ois_min_drop
        ):
            continue
        # Guard: ensure selected sharp score is strictly greater than selected blur score
        if ois_values[sharp_idx] <= ois_values[blur_idx] + 1e-6:
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

        # For non-OIS sharp selection, map the (possibly reused) OIS sharp index into nonois space
        if REUSE_SHARP_REFERENCE and global_sharp_idx is not None:
            mapped = int(round((global_sharp_idx / (len(ois_values) - 1)) * (len(nonois_values) - 1))) if len(ois_values) > 1 and len(nonois_values) > 0 else None
            if mapped is not None and sharp_seg_nonois:
                # pick nearest *valid sharp* index within the nonois sharp segment to the mapped position
                sharp_candidates_nonois = filter_local_sharp_segment(nonois_values, sharp_seg_nonois)
                if sharp_candidates_nonois:
                    sharp_idx_nonois = min(sharp_candidates_nonois, key=lambda i: abs(i - mapped))
                    sharp_ref_nonois = float(np.percentile([nonois_values[i] for i in sharp_candidates_nonois], SHARP_REFERENCE_PERCENTILE))
                else:
                    sharp_idx_nonois, sharp_ref_nonois = select_sharp_idx(nonois_values, sharp_seg_nonois)
            else:
                sharp_idx_nonois, sharp_ref_nonois = select_sharp_idx(nonois_values, sharp_seg_nonois)
        else:
            sharp_idx_nonois, sharp_ref_nonois = select_sharp_idx(nonois_values, sharp_seg_nonois)

        blur_idx_nonois = select_blur_idx(nonois_values, blur_seg_nonois)

        # Ensure nonois sharp and blur are distinct and in their segments
        if sharp_idx_nonois is None or blur_idx_nonois is None:
            continue
        if sharp_idx_nonois in blur_seg_nonois or blur_idx_nonois in sharp_seg_nonois or sharp_idx_nonois == blur_idx_nonois:
            continue

        if sharp_idx_nonois is None or blur_idx_nonois is None:
            continue

        if nonois_values[sharp_idx_nonois] <= threshold_nonois:
            continue

        ratio_limit_nonois = NONOIS_FALLBACK_RATIO if used_nonois_fallback else NONOIS_TRANSITION_RATIO
        if not valid_sharp_blur_pair(nonois_values, sharp_idx_nonois, blur_idx_nonois, ratio_limit_nonois):
            continue
        # Guard: ensure non-OIS sharp is strictly greater than non-OIS blur
        if nonois_values[sharp_idx_nonois] <= nonois_values[blur_idx_nonois] + 1e-6:
            continue

        sharp_file = ois_results[sharp_idx][0]
        blur_file = ois_results[blur_idx][0]
        nonois_sharp_file = nonois_results[sharp_idx_nonois][0]
        nonois_blur_file = nonois_results[blur_idx_nonois][0]

        if blur_file in selected_ois_blur_files or nonois_blur_file in selected_nonois_blur_files:
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

        if REQUIRE_MOTION_MATCH and not motion_pair_ok and not (
            is_aligned_onset_t and ALIGNED_ONSET_SKIP_MOTION_CHECK
        ):
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

        selected_ois_blur_files.add(blur_file)
        selected_nonois_blur_files.add(nonois_blur_file)
        selected_ois_blur_indices.add(blur_idx)
        selected_nonois_blur_indices.add(blur_idx_nonois)
        scene_id += 1
        scene_count += 1

    # Force-fill pass: if strict transition logic yields fewer scenes than requested,
    # add extra unique scene pairs from high-sharpness regions.
    if scene_count < MAX_SCENES_PER_VIDEO:
        ranked_sharp_candidates = sorted(
            sharp_candidates,
            key=lambda i: ois_values[i],
            reverse=True,
        )
        if FORCE_REUSE_SAME_SHARP and global_sharp_idx is not None and global_sharp_idx in ranked_sharp_candidates:
            ranked_sharp_candidates = [global_sharp_idx] + [i for i in ranked_sharp_candidates if i != global_sharp_idx]

        prev_forced_ois_blur_idx = None
        prev_forced_nonois_blur_idx = None

        for sharp_idx in ranked_sharp_candidates:
            if scene_count >= MAX_SCENES_PER_VIDEO:
                break

            if sharp_idx >= len(ois_values) - 1:
                continue

            ois_anchor = prev_forced_ois_blur_idx
            if ois_anchor is None:
                ois_anchor = pick_transition_anchor(transitions, sharp_idx)

            blur_indices = build_blur_segment(
                len(ois_values),
                sharp_idx,
                ois_anchor,
                half_window=max(BLUR_WINDOW, 2),
                fallback_window=NONOIS_FALLBACK_BLUR_WINDOW,
            )
            if not blur_indices:
                continue

            if len(ois_values) <= 1 or len(nonois_values) <= 1:
                break

            blur_indices_sorted = sorted(
                set(blur_indices),
                key=lambda i: (ois_values[i], abs(i - (ois_anchor if ois_anchor is not None else i)), i)
            )

            for blur_idx in blur_indices_sorted:
                if scene_count >= MAX_SCENES_PER_VIDEO:
                    break

                if blur_idx <= sharp_idx:
                    continue
                if selected_ois_blur_indices and blur_idx < (max(selected_ois_blur_indices) + MIN_NEXT_BLUR_STEP):
                    continue

                mapped = int(round((sharp_idx / (len(ois_values) - 1)) * (len(nonois_values) - 1)))
                half_window = max(1, SHARP_WINDOW // 2)
                sharp_start_nonois = max(0, mapped - half_window)
                sharp_end_nonois = min(len(nonois_values), mapped + half_window + 1)
                sharp_seg_nonois = list(range(sharp_start_nonois, sharp_end_nonois))
                if not sharp_seg_nonois:
                    continue

                sharp_candidates_nonois = filter_local_sharp_segment(nonois_values, sharp_seg_nonois)
                if sharp_candidates_nonois:
                    sharp_idx_nonois = min(sharp_candidates_nonois, key=lambda i: abs(i - mapped))
                else:
                    sharp_idx_nonois = max(sharp_seg_nonois, key=lambda i: nonois_values[i])

                if sharp_idx_nonois >= len(nonois_values) - 1:
                    continue

                mapped_blur_nonois = int(round(
                    (blur_idx / (len(ois_values) - 1)) * (len(nonois_values) - 1)
                ))

                nonois_anchor = prev_forced_nonois_blur_idx
                if nonois_anchor is None:
                    nonois_anchor = mapped_blur_nonois

                blur_seg_nonois = build_blur_segment(
                    len(nonois_values),
                    sharp_idx_nonois,
                    nonois_anchor,
                    half_window=max(BLUR_WINDOW, 2),
                    fallback_window=NONOIS_FALLBACK_BLUR_WINDOW,
                )
                if not blur_seg_nonois:
                    continue

                blur_nonois_candidates = sorted(
                    set(blur_seg_nonois),
                    key=lambda i: (nonois_values[i], abs(i - nonois_anchor), i)
                )

                blur_idx_nonois = None
                for candidate_idx in blur_nonois_candidates:
                    if candidate_idx <= sharp_idx_nonois:
                        continue
                    if selected_nonois_blur_indices and candidate_idx < (max(selected_nonois_blur_indices) + MIN_NEXT_BLUR_STEP):
                        continue
                    candidate_file = nonois_results[candidate_idx][0]
                    if candidate_file in selected_nonois_blur_files:
                        continue
                    blur_idx_nonois = candidate_idx
                    break

                if blur_idx_nonois is None:
                    continue

                is_aligned_forced = (
                    mapped_nonois_first_t_for_ois is not None and
                    mapped_nonois_first_t_for_ois <= blur_idx <= mapped_nonois_first_t_for_ois + ALIGNED_ONSET_BLUR_WINDOW
                )
                ois_ratio_limit = ALIGNED_ONSET_RATIO_LIMIT if is_aligned_forced else BLUR_TO_SHARP_MAX_RATIO
                ois_min_drop = ALIGNED_ONSET_MIN_DROP_RATIO if is_aligned_forced else MIN_DROP_RATIO

                # Keep force-filled pairs realistic: blur must be clearly lower and not too close.
                if not valid_sharp_blur_pair(
                    ois_values,
                    sharp_idx,
                    blur_idx,
                    ois_ratio_limit,
                    min_drop_ratio=ois_min_drop
                ):
                    continue
                # Guard: ensure selected sharp score is strictly greater than selected blur score
                if ois_values[sharp_idx] <= ois_values[blur_idx] + 1e-6:
                    continue

                if not valid_sharp_blur_pair(
                    nonois_values,
                    sharp_idx_nonois,
                    blur_idx_nonois,
                    NONOIS_FALLBACK_RATIO,
                    min_drop_ratio=ois_min_drop
                ):
                    continue
                # Guard: ensure non-OIS sharp is strictly greater than non-OIS blur
                if nonois_values[sharp_idx_nonois] <= nonois_values[blur_idx_nonois] + 1e-6:
                    continue

                sharp_file = ois_results[sharp_idx][0]
                blur_file = ois_results[blur_idx][0]
                nonois_sharp_file = nonois_results[sharp_idx_nonois][0]
                nonois_blur_file = nonois_results[blur_idx_nonois][0]

                if (
                    blur_file in selected_ois_blur_files or
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

                if REQUIRE_MOTION_MATCH and not motion_pair_ok and not (
                    is_aligned_forced and ALIGNED_ONSET_SKIP_MOTION_CHECK
                ):
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

                selected_ois_blur_files.add(blur_file)
                selected_nonois_blur_files.add(nonois_blur_file)
                selected_ois_blur_indices.add(blur_idx)
                selected_nonois_blur_indices.add(blur_idx_nonois)
                prev_forced_ois_blur_idx = blur_idx
                prev_forced_nonois_blur_idx = blur_idx_nonois
                scene_id += 1
                scene_count += 1

    # Hard guarantee pass: reuse a valid sharp and vary blur picks while preserving segment order.
    if FORCE_GUARANTEE_SCENES and scene_count < MAX_SCENES_PER_VIDEO and len(ois_values) > 1 and len(nonois_values) > 1:
        if global_sharp_idx is not None:
            base_sharp_idx = global_sharp_idx
        elif ranked_sharp_candidates:
            base_sharp_idx = ranked_sharp_candidates[0]
        else:
            base_sharp_idx = None

        if base_sharp_idx is not None and base_sharp_idx < len(ois_values) - 1:
            last_selected_ois_blur_idx = max(selected_ois_blur_indices) if selected_ois_blur_indices else None
            ois_anchor = last_selected_ois_blur_idx
            if ois_anchor is None:
                ois_anchor = pick_transition_anchor(transitions, base_sharp_idx)
            if ois_anchor is None and mapped_nonois_first_t_for_ois is not None:
                ois_anchor = mapped_nonois_first_t_for_ois

            blur_indices = build_blur_segment(
                len(ois_values),
                base_sharp_idx,
                ois_anchor,
                half_window=max(ALIGNED_ONSET_BLUR_WINDOW, BLUR_WINDOW),
                fallback_window=max(NONOIS_FALLBACK_BLUR_WINDOW, ALIGNED_ONSET_BLUR_WINDOW),
            )

            # Prefer lower-sharpness blur frames while keeping temporal order.
            min_ois_blur_idx = (last_selected_ois_blur_idx + MIN_NEXT_BLUR_STEP) if last_selected_ois_blur_idx is not None else (base_sharp_idx + 1)
            blur_indices_sorted = sorted(
                [i for i in set(blur_indices) if i >= min_ois_blur_idx],
                key=lambda i: (abs(i - min_ois_blur_idx), ois_values[i], i)
            )

            mapped_sharp_nonois = int(round(
                (base_sharp_idx / (len(ois_values) - 1)) * (len(nonois_values) - 1)
            ))
            half_window = max(1, SHARP_WINDOW // 2)
            sharp_start_nonois = max(0, mapped_sharp_nonois - half_window)
            sharp_end_nonois = min(len(nonois_values), mapped_sharp_nonois + half_window + 1)
            sharp_seg_nonois = list(range(sharp_start_nonois, sharp_end_nonois))
            sharp_candidates_nonois = filter_local_sharp_segment(nonois_values, sharp_seg_nonois)

            if sharp_candidates_nonois:
                base_sharp_idx_nonois = min(sharp_candidates_nonois, key=lambda i: abs(i - mapped_sharp_nonois))
            elif sharp_seg_nonois:
                base_sharp_idx_nonois = max(sharp_seg_nonois, key=lambda i: nonois_values[i])
            else:
                base_sharp_idx_nonois = None

            if base_sharp_idx_nonois is not None and base_sharp_idx_nonois < len(nonois_values) - 1:
                last_selected_nonois_blur_idx = max(selected_nonois_blur_indices) if selected_nonois_blur_indices else None
                for blur_idx in blur_indices_sorted:
                    if scene_count >= MAX_SCENES_PER_VIDEO:
                        break

                    blur_file = ois_results[blur_idx][0]
                    if blur_file in selected_ois_blur_files:
                        continue

                    mapped_blur_nonois = int(round(
                        (blur_idx / (len(ois_values) - 1)) * (len(nonois_values) - 1)
                    ))

                    blur_seg_nonois = build_blur_segment(
                        len(nonois_values),
                        base_sharp_idx_nonois,
                        mapped_blur_nonois,
                        half_window=max(ALIGNED_ONSET_BLUR_WINDOW, BLUR_WINDOW),
                        fallback_window=max(NONOIS_FALLBACK_BLUR_WINDOW, ALIGNED_ONSET_BLUR_WINDOW),
                    )
                    min_nonois_blur_idx = (last_selected_nonois_blur_idx + MIN_NEXT_BLUR_STEP) if last_selected_nonois_blur_idx is not None else (base_sharp_idx_nonois + 1)
                    blur_nonois_candidates = sorted(
                        [i for i in set(blur_seg_nonois) if i >= min_nonois_blur_idx],
                        key=lambda i: (abs(i - min_nonois_blur_idx), nonois_values[i], i)
                    )

                    blur_idx_nonois = None
                    for candidate_idx in blur_nonois_candidates:
                        candidate_file = nonois_results[candidate_idx][0]
                        if candidate_file not in selected_nonois_blur_files:
                            blur_idx_nonois = candidate_idx
                            break

                    if blur_idx_nonois is None:
                        continue

                    sharp_file = ois_results[base_sharp_idx][0]
                    nonois_sharp_file = nonois_results[base_sharp_idx_nonois][0]
                    nonois_blur_file = nonois_results[blur_idx_nonois][0]

                    ois_sharp_img = read_image(os.path.join(ois_dir, sharp_file))
                    ois_blur_img = read_image(os.path.join(ois_dir, blur_file))
                    nonois_sharp_img = read_image(os.path.join(nonois_dir, nonois_sharp_file))
                    nonois_blur_img = read_image(os.path.join(nonois_dir, nonois_blur_file))

                    if any(img is None for img in [ois_sharp_img, ois_blur_img, nonois_sharp_img, nonois_blur_img]):
                        continue

                    # Structural guard only: keep frame order and non-trivial gap.
                    if blur_idx - base_sharp_idx < MIN_SHARP_BLUR_GAP:
                        continue
                    if blur_idx_nonois - base_sharp_idx_nonois < MIN_SHARP_BLUR_GAP:
                        continue

                    # Guard: ensure selected sharp score is strictly greater than selected blur score
                    if ois_values[base_sharp_idx] <= ois_values[blur_idx] + 1e-6:
                        continue
                    if nonois_values[base_sharp_idx_nonois] <= nonois_values[blur_idx_nonois] + 1e-6:
                        continue

                    _, is_linear_scene = evaluate_motion_pair(
                        ois_sharp_img,
                        ois_blur_img,
                        nonois_sharp_img,
                        nonois_blur_img
                    )

                    if DEBUG_MODE:
                        visualize_selection(
                            scene_id,
                            ois_sharp_img,
                            ois_blur_img,
                            nonois_sharp_img,
                            nonois_blur_img,
                            ois_values[base_sharp_idx],
                            ois_values[blur_idx],
                            nonois_values[base_sharp_idx_nonois],
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

                    selected_ois_blur_files.add(blur_file)
                    selected_nonois_blur_files.add(nonois_blur_file)
                    selected_ois_blur_indices.add(blur_idx)
                    selected_nonois_blur_indices.add(blur_idx_nonois)
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

                        if scene_id > prev_scene_id:
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

                        if scene_id > prev_scene_id:
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
