import os
import cv2
import csv
import rawpy
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ================= CONFIG =================

DATASET_ROOT = "dataset"
OUTPUT_ROOT = "aligned"
DEBUG_ALIGN_DIR = "debug_alignment"

GT_SOURCE = "ois"

GEO_INLIER_THRESHOLD = 0.1
FLOW_THRESHOLD = 15.0
SSIM_THRESHOLD = 0.75

# ================= IO =================

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


def open_log(path, header):
    exists = os.path.exists(path)
    f = open(path, "a", newline="", buffering=1)
    writer = csv.writer(f)
    if not exists:
        writer.writerow(header)
    return f, writer


# ================= LOGS =================

def init_logs():
    ensure_dir("logs")

    geo_log, geo_writer = open_log(
        "logs/geo_log.csv",
        ["scene", "image", "inlier_ratio", "mean_flow", "valid", "fallback"]
    )

    photo_log, photo_writer = open_log(
        "logs/photo_log.csv",
        ["scene", "image", "mean_before", "mean_after", "valid"]
    )

    color_log, color_writer = open_log(
        "logs/color_log.csv",
        ["scene", "image",
         "deltaE_before", "deltaE_after",
         "ssim", "used_color", "overall_pass"]
    )

    scene_fail_log, scene_fail_writer = open_log(
        "logs/scene_fail_log.csv",
        ["scene", "failed_images", "failed_count"]
    )

    return (
        geo_log, photo_log, color_log, scene_fail_log,
        geo_writer, photo_writer, color_writer, scene_fail_writer
    )


# ================= METRICS =================

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


# ================= ALIGNMENT =================

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


# ================= COLOR =================

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


# ================= DEBUG =================

def save_alignment_visual(scene,
                          ois_sharp, ois_blur,
                          nonois_raw, nonois_aligned,
                          blur_before, blur_after,
                          ssim_val, dE, flow):

    if any(img is None for img in [
        ois_sharp, ois_blur,
        nonois_raw, nonois_aligned,
        blur_before, blur_after
    ]):
        return

    ensure_dir(DEBUG_ALIGN_DIR)

    h, w = ois_sharp.shape[:2]

    top = np.hstack([ois_sharp, ois_blur, nonois_raw, blur_before])
    bottom = np.hstack([ois_sharp, ois_blur, nonois_aligned, blur_after])

    vis = np.vstack([top, bottom])

    labels_top = ["OIS SHARP", "OIS BLUR", "NONOIS RAW", "BLUR RAW"]
    labels_bot = ["OIS SHARP", "OIS BLUR", "NONOIS ALIGNED", "BLUR ALIGNED"]

    for i, label in enumerate(labels_top):
        cv2.putText(vis, label, (i*w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    for i, label in enumerate(labels_bot):
        cv2.putText(vis, label, (i*w + 10, h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(vis, f"SSIM {ssim_val:.3f}", (10, 2*h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(vis, f"dE {dE:.2f}", (300, 2*h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(vis, f"Flow {flow:.2f}", (550, 2*h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imwrite(os.path.join(DEBUG_ALIGN_DIR, f"{scene}.jpg"), vis)


# ================= MAIN =================

def run_pipeline():

    (
        geo_log, photo_log, color_log, scene_fail_log,
        geo_writer, photo_writer, color_writer, scene_fail_writer
    ) = init_logs()

    output_dir = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "color")
    ensure_dir(output_dir)

    scenes = sorted([
        s for s in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, s))
    ])

    for scene in tqdm(scenes):

        scene_path = os.path.join(DATASET_ROOT, scene)

        gt = read_image(os.path.join(scene_path, "ois_sharp.dng"))
        nonois = read_image(os.path.join(scene_path, "nonois_sharp.dng"))

        if gt is None or nonois is None:
            continue

        # align non-OIS → OIS
        H_nonois_to_ois, _ = get_alignment_matrix(gt, nonois)

        if H_nonois_to_ois is not None:
            aligned_nonois = cv2.warpPerspective(
                nonois,
                H_nonois_to_ois,
                (gt.shape[1], gt.shape[0])
            )
        else:
            aligned_nonois = nonois

        ois_blur_img = None
        for f in os.listdir(scene_path):
            if "ois_blur" in f:
                ois_blur_img = read_image(os.path.join(scene_path, f))

        scene_failed = []

        for file in os.listdir(scene_path):

            if "ois_sharp" in file:
                continue

            img = read_image(os.path.join(scene_path, file))
            if img is None:
                continue

            H_direct, ratio = get_alignment_matrix(gt, img)

            used_fallback = False

            if H_direct is not None and ratio >= GEO_INLIER_THRESHOLD:
                H_final = H_direct
            else:
                H_blur_to_nonois, _ = get_alignment_matrix(nonois, img)

                if H_blur_to_nonois is not None and H_nonois_to_ois is not None:
                    H_final = H_nonois_to_ois @ H_blur_to_nonois
                    used_fallback = True
                else:
                    continue

            geo_img = cv2.warpPerspective(img, H_final, (gt.shape[1], gt.shape[0]))

            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(geo_img, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            mean_flow = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
            geo_ok = mean_flow < FLOW_THRESHOLD

            geo_writer.writerow([scene, file, ratio, mean_flow, geo_ok, used_fallback])

            photo_img = photometric_align(gt, geo_img)
            dE_before = deltaE(gt, photo_img)

            candidate = color_align(gt, photo_img)
            dE_after_candidate = deltaE(gt, candidate)

            final_img = candidate if dE_after_candidate < dE_before else photo_img
            dE_after = min(dE_before, dE_after_candidate)

            ssim_val = compute_ssim(gt, final_img)

            overall_pass = geo_ok and (ssim_val > SSIM_THRESHOLD)

            if not overall_pass:
                scene_failed.append(file)

            save_alignment_visual(
                scene,
                gt,
                ois_blur_img if ois_blur_img is not None else gt,
                nonois,
                aligned_nonois,
                img,
                final_img,
                ssim_val,
                dE_after,
                mean_flow
            )

            cv2.imwrite(
                os.path.join(output_dir, f"{scene}_{file}.jpg"),
                final_img
            )

    print("pipeline complete")


if __name__ == "__main__":
    run_pipeline()