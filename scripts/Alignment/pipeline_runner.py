import os
import cv2
import csv
import rawpy
import shutil
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
OUTPUT_ROOT = os.path.join(BASE_DIR, "aligned")
LOG_ROOT = os.path.join(BASE_DIR, "logs")

GT_SOURCE = "ois"

GEO_INLIER_THRESHOLD = 0.1
FLOW_THRESHOLD = 15.0


# ================= IMAGE =================

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
        except Exception as e:
            print(f"Failed to read RAW: {path}")
            return None

    return cv2.imread(path)


def to_jpg(name):
    return os.path.splitext(name)[0] + ".jpg"


# ================= METRICS =================

def compute_psnr(gt, img):
    mse = np.mean((gt.astype(np.float32) - img.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(gt, img):
    return ssim(
        cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    )


# ================= UTILS =================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def open_log(path, header):
    exists = os.path.exists(path)
    f = open(path, "a", newline="", buffering=1)
    writer = csv.writer(f)
    if not exists:
        writer.writerow(header)
    return f, writer


def init_logs():

    ensure_dir(LOG_ROOT)

    geo_log, geo_writer = open_log(
        os.path.join(LOG_ROOT, "geo_log.csv"),
        ["scene", "image", "inlier_ratio", "mean_flow", "valid", "fallback"]
    )

    photo_log, photo_writer = open_log(
        os.path.join(LOG_ROOT, "photo_log.csv"),
        ["scene", "image", "mean_before", "mean_after", "valid"]
    )

    color_log, color_writer = open_log(
        os.path.join(LOG_ROOT, "color_log.csv"),
        ["scene", "image",
         "deltaE_before", "deltaE_after",
         "psnr_raw", "psnr_photo", "psnr_final",
         "ssim", "used_color", "overall_pass"]
    )

    scene_fail_log, scene_fail_writer = open_log(
        os.path.join(LOG_ROOT, "scene_fail_log.csv"),
        ["scene", "failed_images", "failed_count"]
    )

    return (
        geo_log, photo_log, color_log, scene_fail_log,
        geo_writer, photo_writer, color_writer, scene_fail_writer
    )


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

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 20:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        return None, 0

    inlier_ratio = float(np.sum(inliers) / len(inliers)) if inliers is not None else 0

    return H, inlier_ratio


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

    corrected = np.zeros_like(img_lab)

    for c in range(3):
        shift = np.mean(ref_lab[:,:,c]) - np.mean(img_lab[:,:,c])
        corrected[:,:,c] = img_lab[:,:,c] + shift

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def deltaE(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(np.mean(np.sqrt(np.sum((ref_lab.astype(np.float32) - img_lab.astype(np.float32))**2, axis=2))))


# ================= MAIN =================

def run_pipeline():

    logs = init_logs()
    geo_log, photo_log, color_log, scene_fail_log = logs[:4]
    geo_writer, photo_writer, color_writer, scene_fail_writer = logs[4:]

    output_dir = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "color")
    ensure_dir(output_dir)

    print("DATASET ROOT:", DATASET_ROOT)
    print("OUTPUT ROOT:", output_dir)

    if not os.path.exists(DATASET_ROOT):
        print("❌ DATASET ROOT NOT FOUND")
        return

    scenes = sorted([
        s for s in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, s))
    ])

    if len(scenes) == 0:
        print("❌ NO SCENES FOUND IN DATASET")
        return

    finished_scenes = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

    for scene in tqdm(scenes, desc="Total Alignment Progress"):

        if scene in finished_scenes:
            continue

        scene_path = os.path.join(DATASET_ROOT, scene)

        gt_prefix = "ois_sharp" if GT_SOURCE == "ois" else "nonois_sharp"
        other_prefix = "nonois_sharp" if GT_SOURCE == "ois" else "ois_sharp"

        gt_path = os.path.join(scene_path, f"{gt_prefix}.dng")
        other_path = os.path.join(scene_path, f"{other_prefix}.dng")

        if not os.path.exists(gt_path):
            print(f"Missing GT: {gt_path}")
            continue

        if not os.path.exists(other_path):
            print(f"Missing other: {other_path}")
            continue

        gt = read_image(gt_path)
        other = read_image(other_path)

        if gt is None or other is None:
            print(f"Failed reading images in {scene}")
            continue

        master_H, _ = get_alignment_matrix(gt, other)

        color_out = os.path.join(output_dir, scene)
        ensure_dir(color_out)

        cv2.imwrite(os.path.join(color_out, to_jpg(os.path.basename(gt_path))), gt)

        files = sorted([f for f in os.listdir(scene_path) if not f.startswith(gt_prefix)])

        scene_failed_files = []

        for file in tqdm(files, desc=f"Aligning {scene}", leave=False):

            img = read_image(os.path.join(scene_path, file))
            if img is None:
                continue

            psnr_raw = compute_psnr(gt, img)

            H, ratio = get_alignment_matrix(gt, img)

            used_fallback = False
            if H is None or ratio < GEO_INLIER_THRESHOLD:
                H = master_H
                used_fallback = True

            if H is None:
                continue

            geo_img = cv2.warpPerspective(img, H, (gt.shape[1], gt.shape[0]))

            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(geo_img, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            mean_flow = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))

            geo_writer.writerow([scene, file, ratio, mean_flow, ratio >= GEO_INLIER_THRESHOLD, used_fallback])

            mean_before = abs(np.mean(gt) - np.mean(geo_img))
            photo_img = photometric_align(gt, geo_img)
            mean_after = abs(np.mean(gt) - np.mean(photo_img))

            photo_writer.writerow([scene, file, mean_before, mean_after, mean_after < mean_before])

            psnr_photo = compute_psnr(gt, photo_img)

            dE_before = deltaE(gt, photo_img)
            candidate = color_align(gt, photo_img)
            dE_after_candidate = deltaE(gt, candidate)

            if dE_after_candidate < dE_before:
                final_img = candidate
                dE_after = dE_after_candidate
                used_color = True
            else:
                final_img = photo_img
                dE_after = dE_before
                used_color = False

            psnr_final = compute_psnr(gt, final_img)
            ssim_val = compute_ssim(gt, final_img)

            geo_ok = mean_flow < FLOW_THRESHOLD
            photo_ok = mean_after < mean_before
            color_ok = dE_after <= dE_before
            quality_ok = psnr_final > psnr_raw

            overall_pass = geo_ok and photo_ok and color_ok and quality_ok

            if not overall_pass:
                scene_failed_files.append(file)

            color_writer.writerow([
                scene, file,
                dE_before, dE_after,
                psnr_raw, psnr_photo, psnr_final,
                ssim_val, used_color, overall_pass
            ])

            cv2.imwrite(os.path.join(color_out, to_jpg(file)), final_img)

        if scene_failed_files:
            scene_fail_writer.writerow([
                scene,
                ";".join(scene_failed_files),
                len(scene_failed_files)
            ])

    geo_log.close()
    photo_log.close()
    color_log.close()
    scene_fail_log.close()

    print("\n✅ Pipeline Complete.")


if __name__ == "__main__":
    run_pipeline()