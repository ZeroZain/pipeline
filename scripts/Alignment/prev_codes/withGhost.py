import os
import cv2
import csv
import argparse
import subprocess
import sys
import re
import rawpy
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import datetime
import time

# CONFIG

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")
LOG_ROOT = os.path.join(WORKSPACE_ROOT, "logs")
DEBUG_ROOT = os.path.join(WORKSPACE_ROOT, "debug")

DATASET_ROOT = os.path.join(DATA_ROOT, "dataset")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "aligned")
DEBUG_ALIGN_DIR = os.path.join(DEBUG_ROOT, "alignment")

GT_SOURCE = "ois"

GEO_INLIER_THRESHOLD = 0.1
FLOW_ROI_P90_THRESHOLD = 15.0
FLOW_METRIC_NAME = "roi_p90"
FLOW_ROI_BORDER = 0.15
#ghosting filtering
GHOST_EDGE_RATIO_THRESHOLD = 0.20
MIN_VALID_OVERLAP_RATIO = 0.80
GHOST_EDGE_DILATE_SIZE = 3
GHOST_EDGE_DILATE_ITERS = 1
COLOR_EDGE_BAND = 0.10
COLOR_EDGE_CHANGE_RATIO_FLOOR = 0.60
OIS_BLUR_MEAN_LUMA_MAX_DELTA = 1.5
PAIR_CHROMA_MEAN_BLEND = 0.85
PAIR_CHROMA_STD_BLEND = 0.75
PAIR_LUMA_MEAN_BLEND = 0.80
PAIR_HARD_BLEND = 0.20

SSIM_THRESHOLD_SHARP = 0.70
SSIM_THRESHOLD_BLUR = 0.45

ECC_MAX_ITERS = 100
ECC_EPS = 1e-6
SCENE_PATTERN = re.compile(r"^scene_(\d+)$")
MIN_LINEAR_DET = 0.05
FOLDOVER_GRID_SIZE = 6
FOLDOVER_MIN_JACOBIAN_RATIO = 0.01

# IO

def read_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            return None

    return cv2.imread(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_posix(path):
    return path.replace(os.sep, "/")


def slugify_path(path):
    parts = []

    for part in os.path.normpath(path).split(os.sep):
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_")
        parts.append(clean or "item")

    return "__".join(parts)


def list_scene_dirs(root):
    scenes = []

    if not os.path.exists(root):
        return scenes

    for current_root, dirs, _ in os.walk(root):
        dirs.sort()

        for name in dirs:
            if SCENE_PATTERN.match(name):
                full_path = os.path.join(current_root, name)
                rel_path = os.path.relpath(full_path, root)
                scenes.append((rel_path, full_path))

        dirs[:] = [name for name in dirs if not SCENE_PATTERN.match(name)]

    return sorted(scenes, key=lambda item: item[0].lower())


def init_log(path, header):
    if not os.path.exists(path):
        append_row(path, header)

def append_row(path, row):
    for _ in range(100):
        try:
            with open(path, "a", newline="", buffering=1) as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return
        except PermissionError:
            time.sleep(0.05)
    with open(path, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(row)


# Helper: check whether a CSV already contains an entry for scene/image
def csv_has_entry(path, scene_key, image_name=None):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", newline="") as fr:
            reader = csv.reader(fr)
            # skip header
            try:
                next(reader)
            except StopIteration:
                return False

            for row in reader:
                if not row:
                    continue
                if row[0] != scene_key:
                    continue
                if image_name is None:
                    return True
                if len(row) > 1 and row[1] == image_name:
                    return True
        return False
    except Exception:
        return False


# LOGS

def init_logs():
    ensure_dir(LOG_ROOT)
    geo_log_path = os.path.join(LOG_ROOT, "geo_log.csv")
    init_log(
        geo_log_path,
        ["scene", "image", "inlier_ratio", "mean_flow", "flow_p90", "flow_roi_p90", "ghost_edge_ratio", "flow_metric", "model", "valid", "fallback"]
    )

    photo_log_path = os.path.join(LOG_ROOT, "photo_log.csv")
    init_log(
        photo_log_path,
        ["scene", "image", "mean_before", "mean_after", "valid"]
    )

    color_log_path = os.path.join(LOG_ROOT, "color_log.csv")
    init_log(
        color_log_path,
        ["scene", "image",
         "deltaE_before", "deltaE_after",
         "ssim", "used_color", "overall_pass"]
    )

    scene_fail_log_path = os.path.join(LOG_ROOT, "scene_fail_log.csv")
    init_log(
        scene_fail_log_path,
        ["scene", "failed_images", "failed_count", "failed_phases"]
    )

    return (
        geo_log_path, photo_log_path, color_log_path, scene_fail_log_path
    )


# METRICS

def compute_ssim(gt, img):
    return ssim(
        cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    )


def deltaE(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    return float(np.mean(np.sqrt(np.sum(
        (ref_lab.astype(np.float32) - img_lab.astype(np.float32))**2,
        axis=2
    ))))


# ALIGNMENT

def get_alignment_matrix(ref, img):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.equalizeHist(ref_gray)
    img_gray = cv2.equalizeHist(img_gray)

    sift = cv2.SIFT_create(4000)

    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return None, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        return None, 0

    ratio = float(np.sum(inliers) / len(inliers)) if inliers is not None else 0
    return H, ratio


def get_ecc_affine_matrix(ref, img):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERS,
        ECC_EPS
    )

    try:
        _, warp = cv2.findTransformECC(
            ref_gray,
            img_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria
        )
    except cv2.error:
        return None

    return np.vstack([warp, np.array([0.0, 0.0, 1.0], dtype=np.float32)])


def get_partial_affine_matrix(ref, img):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.equalizeHist(ref_gray)
    img_gray = cv2.equalizeHist(img_gray)

    sift = cv2.SIFT_create(4000)

    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return None, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 15:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    A, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=4.0)

    if A is None:
        return None, 0

    ratio = float(np.sum(inliers) / len(inliers)) if inliers is not None else 0
    return np.vstack([A, np.array([0.0, 0.0, 1.0], dtype=np.float32)]), ratio


def warp_with_matrix(img, M, out_shape):
    h, w = out_shape[:2]
    # Replicate borders instead of reflecting to avoid mirrored-edge artifacts.
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def get_linear_det(M):
    if M is None or M.shape != (3, 3):
        return None

    H = M.astype(np.float64)
    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]

    A = H[:2, :2]
    det = float(np.linalg.det(A))
    if not np.isfinite(det):
        return None

    return det


def polygon_signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def is_homography_foldover(M, shape):
    if M is None or M.shape != (3, 3):
        return True

    h, w = shape[:2]

    corners = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0],
    ], dtype=np.float32)

    warped_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    if not np.all(np.isfinite(warped_corners)):
        return True

    area = polygon_signed_area(warped_corners)
    if area <= 1.0:
        return True

    gx = np.linspace(0.0, w - 1.0, FOLDOVER_GRID_SIZE, dtype=np.float32)
    gy = np.linspace(0.0, h - 1.0, FOLDOVER_GRID_SIZE, dtype=np.float32)

    cell_w = (w - 1.0) / max(1, FOLDOVER_GRID_SIZE - 1)
    cell_h = (h - 1.0) / max(1, FOLDOVER_GRID_SIZE - 1)
    src_cell_area = max(1e-3, cell_w * cell_h)
    min_det = FOLDOVER_MIN_JACOBIAN_RATIO * src_cell_area

    for yi in range(FOLDOVER_GRID_SIZE - 1):
        for xi in range(FOLDOVER_GRID_SIZE - 1):
            p00 = np.array([[gx[xi], gy[yi]]], dtype=np.float32)
            p10 = np.array([[gx[xi + 1], gy[yi]]], dtype=np.float32)
            p01 = np.array([[gx[xi], gy[yi + 1]]], dtype=np.float32)

            q00 = cv2.perspectiveTransform(p00.reshape(1, -1, 2), M).reshape(2)
            q10 = cv2.perspectiveTransform(p10.reshape(1, -1, 2), M).reshape(2)
            q01 = cv2.perspectiveTransform(p01.reshape(1, -1, 2), M).reshape(2)

            if not (np.all(np.isfinite(q00)) and np.all(np.isfinite(q10)) and np.all(np.isfinite(q01))):
                return True

            dx = q10 - q00
            dy = q01 - q00
            det = float(dx[0] * dy[1] - dx[1] * dy[0])

            if det <= min_det:
                return True

    return False


def is_mirrored_or_degenerate(M):
    det = get_linear_det(M)
    if det is None:
        return True

    return det < MIN_LINEAR_DET


def compute_valid_overlap_ratio(shape, M):
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return float(np.count_nonzero(warped_mask) / (h * w))


def has_local_reflection(shape, M):
    h, w = shape[:2]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    pts = np.array([[[cx, cy], [cx + 1.0, cy], [cx, cy + 1.0]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, M)[0]

    v1 = mapped[1] - mapped[0]
    v2 = mapped[2] - mapped[0]
    local_cross = float(v1[0] * v2[1] - v1[1] * v2[0])

    return local_cross <= 1e-3


def is_transform_usable(shape, M):
    if M is None or not np.all(np.isfinite(M)):
        return False, 0.0

    if has_local_reflection(shape, M):
        return False, 0.0

    valid_ratio = compute_valid_overlap_ratio(shape, M)
    if valid_ratio < MIN_VALID_OVERLAP_RATIO:
        return False, valid_ratio

    return True, valid_ratio


def compute_ghost_edge_ratio(ref, aligned):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    ref_edges = cv2.Canny(ref_gray, 60, 140)
    aligned_edges = cv2.Canny(aligned_gray, 60, 140)

    kernel = np.ones((GHOST_EDGE_DILATE_SIZE, GHOST_EDGE_DILATE_SIZE), np.uint8)
    ref_edges_dilated = cv2.dilate(ref_edges, kernel, iterations=GHOST_EDGE_DILATE_ITERS)

    h, w = ref_edges.shape
    by = int(h * FLOW_ROI_BORDER)
    bx = int(w * FLOW_ROI_BORDER)
    if by > 0 and bx > 0 and (h - 2 * by) > 5 and (w - 2 * bx) > 5:
        ref_roi = ref_edges_dilated[by:h - by, bx:w - bx]
        aligned_roi = aligned_edges[by:h - by, bx:w - bx]
    else:
        ref_roi = ref_edges_dilated
        aligned_roi = aligned_edges

    aligned_count = float(np.count_nonzero(aligned_roi))
    if aligned_count < 1.0:
        return 0.0

    ghost_edges = np.logical_and(aligned_roi > 0, ref_roi == 0)
    return float(np.count_nonzero(ghost_edges) / aligned_count)


def compute_flow_metrics(ref, aligned):
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_flow = float(np.mean(mag))
    p90_flow = float(np.percentile(mag, 90))

    h, w = mag.shape
    by = int(h * FLOW_ROI_BORDER)
    bx = int(w * FLOW_ROI_BORDER)
    if by > 0 and bx > 0 and (h - 2 * by) > 5 and (w - 2 * bx) > 5:
        roi = mag[by:h - by, bx:w - bx]
    else:
        roi = mag
    roi_p90_flow = float(np.percentile(roi, 90))

    return mean_flow, p90_flow, roi_p90_flow


def evaluate_candidate(ref, img, M):
    if M is None:
        return None

    usable, valid_ratio = is_transform_usable(ref.shape, M)
    if not usable:
        return None

    if is_mirrored_or_degenerate(M):
        return None

    if is_homography_foldover(M, ref.shape):
        return None

    aligned = warp_with_matrix(img, M, ref.shape)
    mean_flow, p90_flow, roi_p90_flow = compute_flow_metrics(ref, aligned)
    ghost_edge_ratio = compute_ghost_edge_ratio(ref, aligned)

    return {
        "matrix": M,
        "aligned": aligned,
        "mean_flow": mean_flow,
        "flow_p90": p90_flow,
        "flow_roi_p90": roi_p90_flow,
        "ghost_edge_ratio": ghost_edge_ratio,
        "valid_ratio": valid_ratio,
        "score": roi_p90_flow + (ghost_edge_ratio * 20.0),
    }


def select_best_transform(ref, img, allow_identity=True):
    candidates = []

    H, ratio_h = get_alignment_matrix(ref, img)
    if H is not None:
        eval_h = evaluate_candidate(ref, img, H)
        if eval_h is not None:
            eval_h.update({"model": "homography", "inlier_ratio": ratio_h, "fallback": "none" if ratio_h >= GEO_INLIER_THRESHOLD else "direct_low_inlier"})
            candidates.append(eval_h)

    A, ratio_a = get_partial_affine_matrix(ref, img)
    if A is not None:
        eval_a = evaluate_candidate(ref, img, A)
        if eval_a is not None:
            eval_a.update({"model": "partial_affine", "inlier_ratio": ratio_a, "fallback": "none" if ratio_a >= GEO_INLIER_THRESHOLD else "direct_low_inlier"})
            candidates.append(eval_a)

    H_ecc = get_ecc_affine_matrix(ref, img)
    if H_ecc is not None:
        eval_ecc = evaluate_candidate(ref, img, H_ecc)
        if eval_ecc is not None:
            eval_ecc.update({"model": "ecc_affine", "inlier_ratio": 0.0, "fallback": "ecc_affine"})
            candidates.append(eval_ecc)

    if allow_identity:
        I = np.eye(3, dtype=np.float32)
        eval_id = evaluate_candidate(ref, img, I)
        if eval_id is not None:
            eval_id.update({"model": "identity", "inlier_ratio": 0.0, "fallback": "identity"})
            candidates.append(eval_id)

    if not candidates:
        return None

    best = min(candidates, key=lambda c: c["score"])
    best["path"] = "direct"
    return best


# COLOR

def photometric_align(ref, img):
    ref_f = ref.astype(np.float32)
    img_f = img.astype(np.float32)

    corrected = np.zeros_like(img_f)

    for c in range(3):
        shift = np.mean(ref_f[:,:,c]) - np.mean(img_f[:,:,c])
        corrected[:,:,c] = img_f[:,:,c] + shift

    return np.clip(corrected, 0, 255).astype(np.uint8)


def photometric_align_luma_only(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = img_lab.copy()
    l_shift = np.mean(ref_lab[:, :, 0]) - np.mean(img_lab[:, :, 0])
    result[:, :, 0] = img_lab[:, :, 0] + l_shift

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def color_align(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = np.zeros_like(img_lab)

    for c in range(3):
        ref_mean, ref_std = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std()
        img_mean, img_std = img_lab[:,:,c].mean(), img_lab[:,:,c].std()

        result[:,:,c] = (img_lab[:,:,c] - img_mean) * (ref_std / (img_std + 1e-6)) + ref_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def color_align_luma_only(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = img_lab.copy()

    ref_l = ref_lab[:, :, 0]
    img_l = img_lab[:, :, 0]

    ref_mean, ref_std = ref_l.mean(), ref_l.std()
    img_mean, img_std = img_l.mean(), img_l.std()

    result[:, :, 0] = (img_l - img_mean) * (ref_std / (img_std + 1e-6)) + ref_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def color_align_chroma_only(ref, img, max_scale=1.25):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = img_lab.copy()

    for c in (1, 2):
        ref_mean, ref_std = ref_lab[:, :, c].mean(), ref_lab[:, :, c].std()
        img_mean, img_std = img_lab[:, :, c].mean(), img_lab[:, :, c].std()
        scale = ref_std / (img_std + 1e-6)
        scale = float(np.clip(scale, 1.0 / max_scale, max_scale))
        result[:, :, c] = (img_lab[:, :, c] - img_mean) * scale + ref_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def color_align_chroma_mean_only(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = img_lab.copy()

    for c in (1, 2):
        ref_mean = ref_lab[:, :, c].mean()
        img_mean = img_lab[:, :, c].mean()
        result[:, :, c] = img_lab[:, :, c] + (ref_mean - img_mean)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def chroma_std_mean(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float((lab[:, :, 1].std() + lab[:, :, 2].std()) / 2.0)


def enforce_chroma_floor(img, source, floor_ratio=1.0):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

    for c in (1, 2):
        img_mean = img_lab[:, :, c].mean()
        img_std = img_lab[:, :, c].std()
        src_std = src_lab[:, :, c].std()
        target_std = floor_ratio * src_std

        if img_std < target_std and img_std > 1e-6:
            scale = target_std / img_std
            img_lab[:, :, c] = (img_lab[:, :, c] - img_mean) * scale + img_mean

    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)


def build_edge_band_mask(shape, band_ratio=COLOR_EDGE_BAND, blur_ksize=41):
    h, w = shape[:2]
    by = max(1, int(h * band_ratio))
    bx = max(1, int(w * band_ratio))

    mask = np.zeros((h, w), dtype=np.float32)
    mask[:by, :] = 1.0
    mask[h - by:, :] = 1.0
    mask[:, :bx] = 1.0
    mask[:, w - bx:] = 1.0

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return np.clip(mask, 0.0, 1.0)


def enforce_edge_color_change(before_img, after_img, fallback_img):
    before_lab = cv2.cvtColor(before_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    after_lab = cv2.cvtColor(after_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    chroma_change = np.sqrt(
        (after_lab[:, :, 1] - before_lab[:, :, 1]) ** 2 +
        (after_lab[:, :, 2] - before_lab[:, :, 2]) ** 2
    )

    h, w = chroma_change.shape
    by = max(1, int(h * COLOR_EDGE_BAND))
    bx = max(1, int(w * COLOR_EDGE_BAND))

    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:by, :] = True
    edge_mask[h - by:, :] = True
    edge_mask[:, :bx] = True
    edge_mask[:, w - bx:] = True
    center_mask = ~edge_mask

    edge_change = float(np.mean(chroma_change[edge_mask]))
    center_change = float(np.mean(chroma_change[center_mask])) if np.any(center_mask) else edge_change

    if center_change <= 1e-4:
        return after_img

    if edge_change >= COLOR_EDGE_CHANGE_RATIO_FLOOR * center_change:
        return after_img

    edge_blend_mask = build_edge_band_mask(after_img.shape)
    edge_blend = np.repeat(edge_blend_mask[:, :, None], 3, axis=2)

    out = after_img.astype(np.float32) * (1.0 - edge_blend) + fallback_img.astype(np.float32) * edge_blend
    return np.clip(out, 0, 255).astype(np.uint8)


def harmonize_pair_color(target_img, ref_img,
                         chroma_mean_blend=PAIR_CHROMA_MEAN_BLEND,
                         chroma_std_blend=PAIR_CHROMA_STD_BLEND,
                         luma_mean_blend=PAIR_LUMA_MEAN_BLEND):
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    out = target_lab.copy()

    # Pull blur chroma stats toward sharp chroma stats for tighter pair consistency.
    for c in (1, 2):
        t = target_lab[:, :, c]
        r = ref_lab[:, :, c]

        t_mean, t_std = float(t.mean()), float(t.std())
        r_mean, r_std = float(r.mean()), float(r.std())

        desired_mean = (1.0 - chroma_mean_blend) * t_mean + chroma_mean_blend * r_mean
        desired_std = (1.0 - chroma_std_blend) * t_std + chroma_std_blend * r_std

        if t_std > 1e-6:
            normalized = (t - t_mean) / t_std
            out[:, :, c] = normalized * desired_std + desired_mean
        else:
            out[:, :, c] = desired_mean

    # Keep luminance closer to sharp without forcing contrast match.
    t_l = target_lab[:, :, 0]
    r_l = ref_lab[:, :, 0]
    l_shift = luma_mean_blend * (float(r_l.mean()) - float(t_l.mean()))
    out[:, :, 0] = t_l + l_shift

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


# DEBUG

def save_alignment_visual(scene,
                          ois_sharp_raw, ois_blur_raw,
                          nonois_sharp_raw, nonois_blur_raw,
                          ois_sharp_proc, ois_blur_proc,
                          nonois_sharp_proc, nonois_blur_proc):

    if any(img is None for img in [
        ois_sharp_raw, ois_blur_raw,
        nonois_sharp_raw, nonois_blur_raw,
        ois_sharp_proc, ois_blur_proc,
        nonois_sharp_proc, nonois_blur_proc
    ]):
        return

    ensure_dir(DEBUG_ALIGN_DIR)

    h, w = ois_sharp_raw.shape[:2]

    top = np.hstack([ois_sharp_raw, ois_blur_raw, nonois_sharp_raw, nonois_blur_raw])
    bottom = np.hstack([ois_sharp_proc, ois_blur_proc, nonois_sharp_proc, nonois_blur_proc])

    vis = np.vstack([top, bottom])

    labels_top = ["OIS SHARP\n(RAW)", "OIS BLUR\n(RAW)", "NONOIS SHARP\n(RAW)", "NONOIS BLUR\n(RAW)"]
    labels_bot = ["OIS SHARP\n(GT)", "OIS BLUR\n(PROC)", "NONOIS SHARP\n(PROC)", "NONOIS BLUR\n(PROC)"]

    for i, label in enumerate(labels_top):
        lines = label.split("\n")
        for j, line in enumerate(lines):
            cv2.putText(vis, line, (i*w + 10, 25 + j*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    for i, label in enumerate(labels_bot):
        lines = label.split("\n")
        for j, line in enumerate(lines):
            cv2.putText(vis, line, (i*w + 10, h + 25 + j*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imwrite(os.path.join(DEBUG_ALIGN_DIR, f"{slugify_path(scene)}.jpg"), vis)


# MAIN

def run_pipeline(target_scene=None, target_method=None):

    output_root = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "color")
    ensure_dir(output_root)

    # Define paths first so we can use them for cleanup even before logs are opened for writing
    geo_log_path = os.path.join(LOG_ROOT, "geo_log.csv")
    photo_log_path = os.path.join(LOG_ROOT, "photo_log.csv")
    color_log_path = os.path.join(LOG_ROOT, "color_log.csv")
    scene_fail_log_path = os.path.join(LOG_ROOT, "scene_fail_log.csv")

    scenes = list_scene_dirs(DATASET_ROOT)

    # Single-scene mode: filter to the requested scene and force reprocessing
    if target_scene is not None:
        target_posix = to_posix(target_scene)
        scenes = [(rel, full) for rel, full in scenes if to_posix(rel) == target_posix]
        if not scenes:
            print(f"Scene not found: {target_scene}")
            return
        # Remove .processed flag and existing outputs so the scene is fully regenerated
        for rel, full in scenes:
            scene_out = os.path.join(output_root, rel)
            flag = os.path.join(scene_out, ".processed")
            if os.path.exists(flag):
                os.remove(flag)
            # Remove existing aligned outputs
            for img_name in ["ois_sharp.jpg", "ois_blur.jpg", "nonois_sharp.jpg", "nonois_blur.jpg"]:
                img_path = os.path.join(scene_out, img_name)
                if os.path.exists(img_path):
                    os.remove(img_path)

        # Cleanup existing log entries for this scene across all alignment logs
        for log_path in [geo_log_path, photo_log_path, color_log_path, scene_fail_log_path]:
            if not os.path.exists(log_path):
                continue
            temp_path = log_path + ".tmp"
            target_posix = to_posix(target_scene)
            for _ in range(20):
                try:
                    with open(log_path, "r", newline="") as fin, open(temp_path, "w", newline="") as fout:
                        reader = csv.reader(fin)
                        writer = csv.writer(fout)
                        header = next(reader, None)
                        if header:
                            writer.writerow(header)
                        for row in reader:
                            if row and row[0] != target_posix:
                                writer.writerow(row)
                    os.replace(temp_path, log_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
                    
    elif target_method is not None:
        target_posix = to_posix(target_method)
        scenes = [(rel, full) for rel, full in scenes if to_posix(rel).startswith(target_posix)]
        if not scenes:
            print(f"No scenes found for method: {target_method}")
            return

    # Now initialize and open the logs for the actual processing
    (
        geo_log_path, photo_log_path, color_log_path, scene_fail_log_path
    ) = init_logs()

    for scene_rel, scene_path in tqdm(scenes):

        scene_key = to_posix(scene_rel)
        scene_output_dir = os.path.join(output_root, scene_rel)
        ensure_dir(scene_output_dir)

        # Skip scene if already processed (resume support)
        processed_flag = os.path.join(scene_output_dir, ".processed")
        if os.path.exists(processed_flag):
            continue

        gt = read_image(os.path.join(scene_path, "ois_sharp.dng"))
        nonois = read_image(os.path.join(scene_path, "nonois_sharp.dng"))

        if gt is None or nonois is None:
            continue

        nonois_anchor = select_best_transform(gt, nonois, allow_identity=True)

        if nonois_anchor is not None:
            H_nonois_to_ois = nonois_anchor["matrix"]
            aligned_nonois = nonois_anchor["aligned"]
            nonois_anchor_model = nonois_anchor["model"]
        else:
            H_nonois_to_ois = np.eye(3, dtype=np.float32)
            aligned_nonois = nonois
            nonois_anchor_model = "identity"

        scene_failed = []
        scene_failed_phases = set()

        ois_blur_raw = None
        nonois_sharp_raw = nonois
        nonois_blur_raw = None
        ois_blur_proc = None
        nonois_sharp_proc = aligned_nonois
        nonois_blur_proc = None

        file_priority = {
            "nonois_sharp": 0,
            "nonois_blur": 1,
            "ois_blur": 2,
            "ois_sharp": 3,
        }

        scene_files = sorted(
            os.listdir(scene_path),
            key=lambda f: (file_priority.get(os.path.splitext(f)[0].lower(), 99), f.lower())
        )

        for file in scene_files:

            stem = os.path.splitext(file)[0].lower()
            if stem == "ois_sharp":
                continue

            is_ois_blur = stem == "ois_blur"
            is_nonois_sharp = stem == "nonois_sharp"
            is_nonois_blur = stem == "nonois_blur"

            img = read_image(os.path.join(scene_path, file))
            if img is None:
                scene_failed.append(f"{file}|io:READ_FAIL_FALLBACK")
                scene_failed_phases.add("io")

                # Keep output completeness even when raw decode fails.
                if is_ois_blur:
                    img = gt.copy()
                elif is_nonois_sharp:
                    img = aligned_nonois.copy() if aligned_nonois is not None else nonois.copy()
                elif is_nonois_blur:
                    if nonois_sharp_proc is not None:
                        img = nonois_sharp_proc.copy()
                    elif aligned_nonois is not None:
                        img = aligned_nonois.copy()
                    else:
                        img = nonois.copy()
                else:
                    img = gt.copy()

            if is_ois_blur:
                ois_blur_raw = img.copy()
            elif is_nonois_sharp:
                nonois_sharp_raw = img.copy()
            elif is_nonois_blur:
                nonois_blur_raw = img.copy()

            direct_best = select_best_transform(gt, img, allow_identity=True)
            best = direct_best

            chain_best = None
            blur_to_nonois = select_best_transform(nonois, img, allow_identity=False)
            if blur_to_nonois is not None and H_nonois_to_ois is not None:
                H_chain = H_nonois_to_ois @ blur_to_nonois["matrix"]
                chain_eval = evaluate_candidate(gt, img, H_chain)
                if chain_eval is not None:
                    chain_eval.update({
                        "model": f"chain({nonois_anchor_model}+{blur_to_nonois['model']})",
                        "inlier_ratio": min(float(nonois_anchor.get("inlier_ratio", 0.0)) if nonois_anchor else 0.0, float(blur_to_nonois.get("inlier_ratio", 0.0))),
                        "fallback": "chain",
                        "path": "chain"
                    })
                    chain_best = chain_eval

            if best is None and chain_best is None:
                scene_failed.append(f"{file}|geo:NO_TRANSFORM_FORCED_IDENTITY")
                scene_failed_phases.add("geo")

                identity_eval = evaluate_candidate(gt, img, np.eye(3, dtype=np.float32))
                if identity_eval is not None:
                    identity_eval.update({
                        "model": "identity",
                        "inlier_ratio": 0.0,
                        "fallback": "forced_identity_no_transform",
                        "path": "forced"
                    })
                    best = identity_eval
                else:
                    best = {
                        "matrix": np.eye(3, dtype=np.float32),
                        "aligned": img.copy(),
                        "mean_flow": 0.0,
                        "flow_p90": 0.0,
                        "flow_roi_p90": 0.0,
                        "ghost_edge_ratio": 0.0,
                        "valid_ratio": 1.0,
                        "score": 0.0,
                        "model": "identity",
                        "inlier_ratio": 0.0,
                        "fallback": "forced_identity_no_transform",
                        "path": "forced"
                    }

            if best is None:
                best = chain_best
            elif chain_best is not None:
                if is_nonois_sharp or is_nonois_blur:
                    chain_is_consistent = (
                        chain_best["flow_roi_p90"] <= (best["flow_roi_p90"] * 1.10) and
                        chain_best["ghost_edge_ratio"] <= (best["ghost_edge_ratio"] + 0.02)
                    )
                    if chain_is_consistent or chain_best["score"] < best["score"]:
                        best = chain_best
                elif chain_best["score"] < best["score"]:
                    best = chain_best

            if best["model"] == "identity":
                scene_failed.append(f"{file}|geo:IDENTITY_FALLBACK")
                scene_failed_phases.add("geo")

            geo_img = best["aligned"]
            mean_flow = best["mean_flow"]
            flow_p90 = best["flow_p90"]
            flow_roi_p90 = best["flow_roi_p90"]
            ghost_edge_ratio = best.get("ghost_edge_ratio", 0.0)
            ratio = best["inlier_ratio"]
            fallback_mode = best["fallback"]
            geo_ok = (
                flow_roi_p90 < FLOW_ROI_P90_THRESHOLD and
                ghost_edge_ratio < GHOST_EDGE_RATIO_THRESHOLD and
                best["model"] != "identity"
            )

            if not csv_has_entry(geo_log_path, scene_key, file):
                append_row(geo_log_path, [
                    scene_key,
                    file,
                    ratio,
                    mean_flow,
                    flow_p90,
                    flow_roi_p90,
                    ghost_edge_ratio,
                    FLOW_METRIC_NAME,
                    best["model"],
                    geo_ok,
                    fallback_mode
                ])

            if is_nonois_blur or is_nonois_sharp:
                # Keep native non-OIS luminance; forcing it toward GT can look washed/light.
                photo_img = geo_img.copy()
            else:
                photo_img = photometric_align(gt, geo_img)

            mean_before = abs(np.mean(gt) - np.mean(geo_img))
            mean_after = abs(np.mean(gt) - np.mean(photo_img))
            photo_ok = mean_after < mean_before

            if not csv_has_entry(photo_log_path, scene_key, file):
                append_row(photo_log_path, [scene_key, file, mean_before, mean_after, photo_ok])

            dE_before = deltaE(gt, photo_img)

            if is_nonois_blur:
                chroma_ref = nonois_sharp_proc if nonois_sharp_proc is not None else gt
                candidate = color_align_chroma_mean_only(chroma_ref, photo_img)
            elif is_nonois_sharp:
                candidate = color_align_chroma_mean_only(gt, photo_img)
            elif is_ois_blur:
                candidate = color_align_luma_only(gt, photo_img)
            else:
                candidate = color_align(gt, photo_img)
            dE_after_candidate = deltaE(gt, candidate)

            if is_nonois_sharp:
                # Preserve OIS-referenced color direction for sharp frames; quality guard is below.
                final_img = candidate
            else:
                final_img = candidate if dE_after_candidate < dE_before else photo_img
            dE_after = min(dE_before, dE_after_candidate)

            if is_ois_blur:
                # Reduce the bright/filter-like cast when blur color processing overshoots.
                mean_luma_delta = float(np.mean(final_img) - np.mean(geo_img))
                if mean_luma_delta > OIS_BLUR_MEAN_LUMA_MAX_DELTA:
                    blend_proc = 0.60
                    final_img = cv2.addWeighted(final_img, blend_proc, geo_img, 1.0 - blend_proc, 0.0)
                    dE_after = deltaE(gt, final_img)

            if is_nonois_sharp:
                # Hard floor on chroma spread to avoid persistent ashy appearance.
                final_img = enforce_chroma_floor(final_img, geo_img, floor_ratio=1.00)
                dE_after = deltaE(gt, final_img)

            # Guard against over-desaturation on non-OIS outputs.
            if is_nonois_blur or is_nonois_sharp:
                base_chroma = chroma_std_mean(geo_img)
                final_chroma = chroma_std_mean(final_img)
                chroma_floor = 0.98 if is_nonois_sharp else 0.88
                if final_chroma < (chroma_floor * base_chroma):
                    # Preserve OIS color transfer while preventing the ashy desaturation failure mode.
                    blend_proc = 0.55 if is_nonois_sharp else 0.70
                    blend_geo = 1.0 - blend_proc
                    final_img = cv2.addWeighted(final_img, blend_proc, geo_img, blend_geo, 0.0)
                    dE_after = deltaE(gt, final_img)

                # Guard against unintended brightening on non-OIS outputs.
                if np.mean(final_img) > (np.mean(geo_img) + 1.5):
                    blend_proc = 0.35 if is_nonois_blur else 0.25
                    final_img = cv2.addWeighted(final_img, blend_proc, geo_img, 1.0 - blend_proc, 0.0)
                    dE_after = deltaE(gt, final_img)

            # Controlled restoration of OIS brightness (25% blend toward GT) without reintroducing ashiness.
            if is_nonois_blur or is_nonois_sharp:
                final_lab = cv2.cvtColor(final_img, cv2.COLOR_BGR2LAB).astype(np.float32)
                gt_lab = cv2.cvtColor(gt, cv2.COLOR_BGR2LAB).astype(np.float32)
                # Blend L channel: keep 75% current, 25% toward GT
                final_lab[:,:,0] = 0.75 * final_lab[:,:,0] + 0.25 * gt_lab[:,:,0]
                final_img = cv2.cvtColor(final_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                dE_after = deltaE(gt, final_img)

            # Final luminance match: ensure blur matches sharp after all processing.
            if is_nonois_blur and nonois_sharp_proc is not None:
                final_img = photometric_align_luma_only(nonois_sharp_proc, final_img)
                # Stronger pair harmonization so non-OIS sharp/blur stay color-near-identical.
                final_img = harmonize_pair_color(final_img, nonois_sharp_proc)
                dE_after = deltaE(gt, final_img)
            elif is_nonois_sharp and nonois_sharp_proc is not None:
                # Keep a small direct blend for the sharp pair only; blur frames should not borrow texture.
                final_img = cv2.addWeighted(final_img, 1.0 - PAIR_HARD_BLEND, nonois_sharp_proc, PAIR_HARD_BLEND, 0.0)
                dE_after = deltaE(gt, final_img)

            if is_ois_blur or is_nonois_blur or is_nonois_sharp:
                # If edge regions receive much weaker color change than center, re-apply edge color from candidate.
                final_img = enforce_edge_color_change(photo_img, final_img, candidate)
                dE_after = deltaE(gt, final_img)

            ssim_val = compute_ssim(gt, final_img)

            if is_nonois_sharp:
                ssim_pass = ssim_val > SSIM_THRESHOLD_SHARP
            else:
                ssim_pass = ssim_val > SSIM_THRESHOLD_BLUR

            overall_pass = geo_ok and ssim_pass

            if not overall_pass:
                fail_tags = []
                if not geo_ok:
                    geo_detail = []
                    if best["model"] == "identity":
                        geo_detail.append("IDENTITY")
                    if flow_roi_p90 >= FLOW_ROI_P90_THRESHOLD:
                        geo_detail.append(f"FLOW({flow_roi_p90:.1f})")
                    if ghost_edge_ratio >= GHOST_EDGE_RATIO_THRESHOLD:
                        geo_detail.append(f"GHOST({ghost_edge_ratio:.2f})")
                    if not geo_detail:
                        geo_detail.append("UNKNOWN")
                    fail_tags.append("geo:" + "-".join(geo_detail))
                    scene_failed_phases.add("geo")
                if not ssim_pass:
                    fail_tags.append(f"ssim:LT_THRESH({ssim_val:.2f})")
                    scene_failed_phases.add("ssim")

                if fail_tags:
                    scene_failed.append(f"{file}|{'&'.join(fail_tags)}")
                else:
                    scene_failed.append(f"{file}|unknown")
                    scene_failed_phases.add("unknown")

            if not csv_has_entry(color_log_path, scene_key, file):
                append_row(color_log_path, [
                    scene_key, file,
                    dE_before, dE_after,
                    ssim_val, dE_after_candidate < dE_before, overall_pass
                ])

            if is_ois_blur:
                ois_blur_proc = final_img.copy()
            elif is_nonois_sharp:
                nonois_sharp_proc = final_img.copy()
            elif is_nonois_blur:
                nonois_blur_proc = final_img.copy()

            if is_ois_blur:
                output_name = "ois_blur.jpg"
            elif is_nonois_sharp:
                output_name = "nonois_sharp.jpg"
            elif is_nonois_blur:
                output_name = "nonois_blur.jpg"
            else:
                continue

            cv2.imwrite(
                os.path.join(scene_output_dir, output_name),
                final_img
            )

        cv2.imwrite(
            os.path.join(scene_output_dir, "ois_sharp.jpg"),
            gt
        )

        save_alignment_visual(
            scene_rel,
            gt,
            ois_blur_raw,
            nonois_sharp_raw,
            nonois_blur_raw,
            gt,
            ois_blur_proc,
            nonois_sharp_proc,
            nonois_blur_proc
        )

        if scene_failed:
            if not csv_has_entry(scene_fail_log_path, scene_key):
                append_row(scene_fail_log_path, [
                    scene_key,
                    ";".join(scene_failed),
                    len(scene_failed),
                    ";".join(sorted(scene_failed_phases))
                ])

        # mark scene processed so reruns skip it
        try:
            with open(processed_flag, "w") as pf:
                pf.write(datetime.datetime.now().isoformat())
        except Exception:
            pass

    print("pipeline complete")


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


def run_interpolation_script():
    interpolation_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Interpolation", "n256.py")
    )

    print("Alignment stage complete. Starting interpolation...")
    subprocess.run([sys.executable, interpolation_script], check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align images and optionally continue to interpolation."
    )
    parser.add_argument(
        "--run-next",
        action="store_true",
        help="Run n256.py after alignment completes"
    )
    parser.add_argument(
        "--align-only",
        action="store_true",
        help="Run alignment only and stop"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Process only a specific scene (relative path like 'HandShake Method/scene_001')"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Process only a specific method directory (e.g. 'HandShake Method')"
    )
    parser.add_argument(
        "--parallel-methods",
        action="store_true",
        help="Automatically spawn a subprocess for each Method folder in dataset."
    )
    args = parser.parse_args()

    if args.run_next and args.align_only:
        parser.error("Use either --run-next or --align-only, not both.")
    if sum(bool(x) for x in [args.scene, args.method, args.parallel_methods]) > 1:
        parser.error("Use only one of --scene, --method, or --parallel-methods.")

    return args


def main():
    args = parse_args()

    if args.parallel_methods:
        methods = ["HandShake Method", "Sliding Method", "Vibration Method"]
        print(f"Spawning parallel runs for {len(methods)} methods...")
        processes = []
        for method in methods:
            # Re-run self with --method
            print(f"Starting {method}...")
            cmd = [sys.executable, __file__, "--method", method]
            if args.align_only:
                cmd.append("--align-only")
            p = subprocess.Popen(cmd)
            processes.append(p)
            
        for p in processes:
            p.wait()
            
        print("All method subprocesses complete.")
    else:
        run_pipeline(target_scene=args.scene, target_method=args.method)

    if args.parallel_methods or args.method or args.scene:
        # If running a single scene/method or parallel sub-runs, don't auto-prompt for interpolation
        # unless explicitly requested via --run-next
        pass

    if args.run_next:
        run_interpolation_script()
        return

    if args.align_only or args.parallel_methods or args.method or args.scene:
        print("Alignment stage complete.")
        return

    should_run_next = ask_yes_no(
        "Run interpolation now? (align4.py -> n256.py)",
        default=False
    )

    if should_run_next:
        run_interpolation_script()
    else:
        print("Alignment stage complete. Interpolation was not started.")


if __name__ == "__main__":
    main()
