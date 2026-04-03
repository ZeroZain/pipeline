import os
import cv2
import csv
import re
import numpy as np
from tqdm import tqdm

# CONFIG

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")
LOG_ROOT = os.path.join(WORKSPACE_ROOT, "logs")
DEBUG_ROOT = os.path.join(WORKSPACE_ROOT, "debug")

GT_SOURCE = "ois"

INPUT_DIR = os.path.join(DATA_ROOT, "aligned", f"gt_{GT_SOURCE}", "color")
OUTPUT_DIR = os.path.join(DATA_ROOT, "dataset_256", f"gt_{GT_SOURCE}")

TARGET_SIZE = 256

LOG_DIR = LOG_ROOT
LOG_FILE = os.path.join(LOG_ROOT, "interpolation_log.csv")

DEBUG_DIR = os.path.join(DEBUG_ROOT, "interpolation")
SCENE_PATTERN = re.compile(r"^scene_(\d+)$")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


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

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{slugify_path(scene)}.jpg"), vis)


def run_pipeline():
    if not os.path.exists(INPUT_DIR):
        print(f"Input folder not found: {INPUT_DIR}")
        return

    log_rows = []
    scenes = list_scene_dirs(INPUT_DIR)

    for scene_rel, scene_path in tqdm(scenes, desc="Processing Scenes", unit="scene"):

        scene_key = to_posix(scene_rel)
        output_scene = os.path.join(OUTPUT_DIR, scene_rel)
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
                log_rows.append([scene_key, img_name, w, h, "SKIPPED_SMALL"])
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

            log_rows.append([scene_key, img_name, w, h, "SUCCESS"])

        save_debug_visual(scene_rel, debug_images)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
            writer.writerow(["scene", "image", "original_width", "original_height", "status"])
        writer.writerows(log_rows)

    print("\nInterpolation stage complete.")
    print(f"Processed scenes: {len(scenes)}")
    print(f"Check your output at: {OUTPUT_DIR}")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
