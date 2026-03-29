import os
import cv2
import csv
from tqdm import tqdm

# CONFIG

GT_SOURCE = "ois"

INPUT_DIR = f"aligned/gt_{GT_SOURCE}/color"
OUTPUT_DIR = f"dataset_256/gt_{GT_SOURCE}"

TARGET_SIZE = 256

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "interpolation_log.csv")

DEBUG_DIR = "debug_interpolation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


# CENTER CROP

def center_crop(img):
    h, w = img.shape[:2]
    crop_factor = 0.8
    size = int(min(h, w) * crop_factor)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    return img[start_y:start_y + size, start_x:start_x + size]


# DEBUG VISUAL

def save_debug_visual(scene, images_dict):
    required = ["ois_sharp", "ois_blur", "nonois_sharp", "nonois_blur"]

    if not all(k in images_dict for k in required):
        return

    imgs = [images_dict[k] for k in required]

    if any(img is None for img in imgs):
        return

    vis = np.hstack(imgs)

    labels = ["OIS SHARP", "OIS BLUR", "NONOIS SHARP", "NONOIS BLUR"]

    for i, label in enumerate(labels):
        cv2.putText(
            vis,
            label,
            (i * TARGET_SIZE + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{scene}.jpg"), vis)


# LOG STORAGE

log_rows = []

scenes = sorted(os.listdir(INPUT_DIR))

for scene in tqdm(scenes, desc="Processing Scenes", unit="scene"):

    scene_path = os.path.join(INPUT_DIR, scene)

    if not os.path.isdir(scene_path):
        continue

    output_scene = os.path.join(OUTPUT_DIR, scene)
    os.makedirs(output_scene, exist_ok=True)

    debug_images = {}

    for img_name in os.listdir(scene_path):

        output_path = os.path.join(output_scene, img_name)

        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            if img is not None:
                key = os.path.splitext(img_name)[0]
                debug_images[key] = img
            continue

        img_path = os.path.join(scene_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        if min(h, w) < TARGET_SIZE:
            log_rows.append([scene, img_name, w, h, "SKIPPED_SMALL"])
            continue

        cropped = center_crop(img)

        resized = cv2.resize(
            cropped,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        cv2.imwrite(
            output_path,
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        key = os.path.splitext(img_name)[0]
        debug_images[key] = resized

        log_rows.append([scene, img_name, w, h, "SUCCESS"])

    # SAVE DEBUG IMAGE PER SCENE
    import numpy as np
    save_debug_visual(scene, debug_images)


# SAVE LOG

with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        writer.writerow(["scene", "image", "original_width", "original_height", "status"])
    writer.writerows(log_rows)


print("\nInterpolation stage complete.")
print(f"Processed scenes: {len(scenes)}")
print(f"Check your output at: {OUTPUT_DIR}")