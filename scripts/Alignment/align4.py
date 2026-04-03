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

SSIM_THRESHOLD_SHARP = 0.70
SSIM_THRESHOLD_BLUR = 0.45

ECC_MAX_ITERS = 100
ECC_EPS = 1e-6
SCENE_PATTERN = re.compile(r"^scene_(\d+)$")

# IO

def read_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False
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


def open_log(path, header):
    exists = os.path.exists(path)
    f = open(path, "a", newline="", buffering=1)
    writer = csv.writer(f)
    if not exists:
        writer.writerow(header)
    return f, writer


# LOGS

def init_logs():
    ensure_dir(LOG_ROOT)

    geo_log, geo_writer = open_log(
        os.path.join(LOG_ROOT, "geo_log.csv"),
        ["scene", "image", "inlier_ratio", "mean_flow", "flow_p90", "flow_roi_p90", "flow_metric", "model", "valid", "fallback"]
    )

    photo_log, photo_writer = open_log(
        os.path.join(LOG_ROOT, "photo_log.csv"),
        ["scene", "image", "mean_before", "mean_after", "valid"]
    )

    color_log, color_writer = open_log(
        os.path.join(LOG_ROOT, "color_log.csv"),
        ["scene", "image",
         "deltaE_before", "deltaE_after",
         "ssim", "used_color", "overall_pass"]
    )

    scene_fail_log, scene_fail_writer = open_log(
        os.path.join(LOG_ROOT, "scene_fail_log.csv"),
        ["scene", "failed_images", "failed_count", "failed_phases"]
    )

    return (
        geo_log, photo_log, color_log, scene_fail_log,
        geo_writer, photo_writer, color_writer, scene_fail_writer
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
    return cv2.warpPerspective(img, M, (w, h))


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

    aligned = warp_with_matrix(img, M, ref.shape)
    mean_flow, p90_flow, roi_p90_flow = compute_flow_metrics(ref, aligned)

    return {
        "matrix": M,
        "aligned": aligned,
        "mean_flow": mean_flow,
        "flow_p90": p90_flow,
        "flow_roi_p90": roi_p90_flow,
        "score": roi_p90_flow,
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

def run_pipeline():

    (
        geo_log, photo_log, color_log, scene_fail_log,
        geo_writer, photo_writer, color_writer, scene_fail_writer
    ) = init_logs()

    output_root = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "color")
    ensure_dir(output_root)

    scenes = list_scene_dirs(DATASET_ROOT)

    for scene_rel, scene_path in tqdm(scenes):

        scene_key = to_posix(scene_rel)
        scene_output_dir = os.path.join(output_root, scene_rel)
        ensure_dir(scene_output_dir)

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

        for file in os.listdir(scene_path):

            stem = os.path.splitext(file)[0].lower()
            if stem == "ois_sharp":
                continue

            is_ois_blur = stem == "ois_blur"
            is_nonois_sharp = stem == "nonois_sharp"
            is_nonois_blur = stem == "nonois_blur"

            img = read_image(os.path.join(scene_path, file))
            if img is None:
                continue

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
                scene_failed.append(f"{file}|geo:NO_TRANSFORM")
                scene_failed_phases.add("geo")
                continue

            if best is None:
                best = chain_best
            elif chain_best is not None and chain_best["score"] < best["score"]:
                best = chain_best

            if best["model"] == "identity":
                scene_failed.append(f"{file}|geo:IDENTITY_FALLBACK")
                scene_failed_phases.add("geo")

            geo_img = best["aligned"]
            mean_flow = best["mean_flow"]
            flow_p90 = best["flow_p90"]
            flow_roi_p90 = best["flow_roi_p90"]
            ratio = best["inlier_ratio"]
            fallback_mode = best["fallback"]
            geo_ok = flow_roi_p90 < FLOW_ROI_P90_THRESHOLD

            geo_writer.writerow([
                scene_key,
                file,
                ratio,
                mean_flow,
                flow_p90,
                flow_roi_p90,
                FLOW_METRIC_NAME,
                best["model"],
                geo_ok,
                fallback_mode
            ])

            photo_img = photometric_align(gt, geo_img)

            mean_before = abs(np.mean(gt) - np.mean(geo_img))
            mean_after = abs(np.mean(gt) - np.mean(photo_img))
            photo_ok = mean_after < mean_before

            photo_writer.writerow([scene_key, file, mean_before, mean_after, photo_ok])

            dE_before = deltaE(gt, photo_img)

            candidate = color_align(gt, photo_img)
            dE_after_candidate = deltaE(gt, candidate)

            final_img = candidate if dE_after_candidate < dE_before else photo_img
            dE_after = min(dE_before, dE_after_candidate)

            ssim_val = compute_ssim(gt, final_img)

            if is_nonois_sharp:
                ssim_pass = ssim_val > SSIM_THRESHOLD_SHARP
            else:
                ssim_pass = ssim_val > SSIM_THRESHOLD_BLUR

            overall_pass = geo_ok and ssim_pass

            if not overall_pass:
                fail_tags = []
                if not geo_ok:
                    fail_tags.append("geo")
                    scene_failed_phases.add("geo")
                if not ssim_pass:
                    fail_tags.append("ssim")
                    scene_failed_phases.add("ssim")

                if fail_tags:
                    scene_failed.append(f"{file}|{'&'.join(fail_tags)}")
                else:
                    scene_failed.append(f"{file}|unknown")
                    scene_failed_phases.add("unknown")

            color_writer.writerow([
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
            scene_fail_writer.writerow([
                scene_key,
                ";".join(scene_failed),
                len(scene_failed),
                ";".join(sorted(scene_failed_phases))
            ])

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
    args = parser.parse_args()

    if args.run_next and args.align_only:
        parser.error("Use either --run-next or --align-only, not both.")

    return args


def main():
    args = parse_args()

    run_pipeline()

    if args.run_next:
        run_interpolation_script()
        return

    if args.align_only:
        print("Alignment stage complete. Interpolation was not started.")
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
