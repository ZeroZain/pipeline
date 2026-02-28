import os
import cv2
import csv
import numpy as np
from skimage.metrics import structural_similarity as ssim


DATASET_ROOT = "dataset"
OUTPUT_ROOT = "aligned"
GT_SOURCE = "ois"  # "ois" or "nonois"

GEO_INLIER_THRESHOLD = 0.3
FLOW_THRESHOLD = 1.5


def compute_psnr(gt, img):
    mse = np.mean((gt.astype(np.float32) - img.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def init_logs():
    ensure_dir("logs")

    geo_log = open("logs/geo_log.csv", "w", newline="")
    photo_log = open("logs/photo_log.csv", "w", newline="")
    color_log = open("logs/color_log.csv", "w", newline="")

    geo_writer = csv.writer(geo_log)
    photo_writer = csv.writer(photo_log)
    color_writer = csv.writer(color_log)

    geo_writer.writerow(["scene", "image", "inlier_ratio", "mean_flow", "valid"])
    photo_writer.writerow(["scene", "image", "mean_before", "mean_after", "valid"])
    color_writer.writerow(["scene", "image", "deltaE_before", "deltaE_after", "valid"])

    return geo_log, photo_log, color_log, geo_writer, photo_writer, color_writer


def geo_align(ref, img):

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return None, 0, 999

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return None, 0, 999

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)

    if M is None:
        return None, 0, 999

    aligned = cv2.warpAffine(img, M, (ref.shape[1], ref.shape[0]))

    inlier_ratio = float(np.sum(inliers) / len(inliers))

    flow = cv2.calcOpticalFlowFarneback(ref_gray,
                                       cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
                                       None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    mean_flow = float(np.mean(magnitude))

    return aligned, inlier_ratio, mean_flow


def photometric_align(ref, img):

    ref = ref.astype(np.float32)
    img = img.astype(np.float32)

    corrected = np.zeros_like(img)

    for c in range(3):
        ref_mean = np.mean(ref[:,:,c])
        img_mean = np.mean(img[:,:,c])
        ref_std = np.std(ref[:,:,c])
        img_std = np.std(img[:,:,c])

        scale = ref_std / img_std if img_std > 1e-6 else 1.0
        shift = ref_mean - scale * img_mean

        corrected[:,:,c] = scale * img[:,:,c] + shift

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def color_align(ref, img):

    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    corrected = np.zeros_like(img_lab)

    for c in range(3):
        ref_mean, ref_std = cv2.meanStdDev(ref_lab[:,:,c])
        img_mean, img_std = cv2.meanStdDev(img_lab[:,:,c])

        ref_mean, ref_std = ref_mean[0][0], ref_std[0][0]
        img_mean, img_std = img_mean[0][0], img_std[0][0]

        scale = ref_std / img_std if img_std > 1e-6 else 1.0
        corrected[:,:,c] = (img_lab[:,:,c] - img_mean) * scale + ref_mean

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def deltaE(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(np.mean(np.sqrt(np.sum((ref_lab - img_lab)**2, axis=2))))


def run_pipeline():

    geo_log, photo_log, color_log, geo_writer, photo_writer, color_writer = init_logs()

    for scene in sorted(os.listdir(DATASET_ROOT)):

        scene_path = os.path.join(DATASET_ROOT, scene)
        if not os.path.isdir(scene_path):
            continue

        gt_path = os.path.join(scene_path,
                               "ois_sharp.jpg" if GT_SOURCE=="ois" else "nonois_sharp.jpg")

        gt = cv2.imread(gt_path)
        if gt is None:
            continue

        geo_out = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "geo", scene)
        photo_out = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "photo", scene)
        color_out = os.path.join(OUTPUT_ROOT, f"gt_{GT_SOURCE}", "color", scene)

        ensure_dir(geo_out)
        ensure_dir(photo_out)
        ensure_dir(color_out)

        for file in os.listdir(scene_path):

            if file == os.path.basename(gt_path):
                continue

            img = cv2.imread(os.path.join(scene_path, file))
            if img is None:
                continue

            geo_img, inlier_ratio, mean_flow = geo_align(gt, img)
            geo_valid = (inlier_ratio >= GEO_INLIER_THRESHOLD and mean_flow <= FLOW_THRESHOLD)

            geo_writer.writerow([scene, file, inlier_ratio, mean_flow, geo_valid])
            if not geo_valid:
                continue

            cv2.imwrite(os.path.join(geo_out, file), geo_img)

            mean_before = abs(np.mean(gt) - np.mean(geo_img))
            photo_img = photometric_align(gt, geo_img)
            mean_after = abs(np.mean(gt) - np.mean(photo_img))
            photo_valid = mean_after < mean_before

            photo_writer.writerow([scene, file, mean_before, mean_after, photo_valid])
            if not photo_valid:
                continue

            cv2.imwrite(os.path.join(photo_out, file), photo_img)

            delta_before = deltaE(gt, photo_img)
            color_img = color_align(gt, photo_img)
            delta_after = deltaE(gt, color_img)
            color_valid = delta_after < delta_before

            color_writer.writerow([scene, file, delta_before, delta_after, color_valid])
            if not color_valid:
                continue

            cv2.imwrite(os.path.join(color_out, file), color_img)

    geo_log.close()
    photo_log.close()
    color_log.close()


if __name__ == "__main__":
    run_pipeline()