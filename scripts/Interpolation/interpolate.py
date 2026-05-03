import os
import cv2
import csv
import re
import argparse
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
OUTPUT_DIR = os.path.join(DATA_ROOT, "dataset_1080", f"gt_{GT_SOURCE}")

TARGET_SIZE = 1080

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
    crop_factor = 0.9
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


def run_pipeline(target_scene=None, target_size=1080):
    # Dynamically set paths based on size
    current_output_dir = os.path.join(DATA_ROOT, f"dataset_{target_size}", f"gt_{GT_SOURCE}")
    os.makedirs(current_output_dir, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"Input folder not found: {INPUT_DIR}")
        return

    log_rows = []
    scenes = list_scene_dirs(INPUT_DIR)

    # Single-scene mode: filter and force regeneration
    if target_scene is not None:
        target_posix = to_posix(target_scene)
        scenes = [(rel, full) for rel, full in scenes if to_posix(rel) == target_posix]
        if not scenes:
            print(f"Scene not found: {target_scene}")
            return
        # Remove existing outputs so they get regenerated
        for rel, full in scenes:
            out_dir = os.path.join(current_output_dir, rel)
            if os.path.exists(out_dir):
                for img_name in tqdm(os.listdir(out_dir), desc="Cleaning old output", leave=False):
                    img_path = os.path.join(out_dir, img_name)
                    if os.path.isfile(img_path):
                        os.remove(img_path)
        
        # Cleanup existing log entries for this scene in the interpolation log
        if os.path.exists(LOG_FILE):
            temp_path = LOG_FILE + ".tmp"
            target_posix = to_posix(target_scene)
            with open(LOG_FILE, "r", newline="") as fin, open(temp_path, "w", newline="") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                header = next(reader, None)
                if header:
                    writer.writerow(header)
                for row in reader:
                    if row and row[0] != target_posix:
                        writer.writerow(row)
            os.replace(temp_path, LOG_FILE)

    for scene_rel, scene_path in tqdm(scenes, desc=f"Processing Scenes ({target_size}x{target_size})", unit="scene"):

        scene_key = to_posix(scene_rel)
        output_scene = os.path.join(current_output_dir, scene_rel)
        os.makedirs(output_scene, exist_ok=True)

        debug_images = {}

        images = [f for f in os.listdir(scene_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.dng'))]
        for img_name in tqdm(images, desc=f"  └ {scene_rel[:30]}...", unit="img", leave=False):

            output_path = os.path.join(output_scene, img_name)

            # Check if output exists AND has correct dimensions
            if os.path.exists(output_path):
                img = cv2.imread(output_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if h == target_size and w == target_size:
                        key = os.path.splitext(img_name)[0]
                        debug_images[key] = img
                        continue
                    else:
                        # Dimension mismatch, will re-process
                        pass

            img_path = os.path.join(scene_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            if min(h, w) < target_size:
                log_rows.append([scene_key, img_name, w, h, "SKIPPED_SMALL"])
                continue

            cropped = center_crop(img)

            resized = cv2.resize(
                cropped,
                (target_size, target_size),
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

    print(f"\nInterpolation stage complete for {target_size}x{target_size}.")
    print(f"Processed scenes: {len(scenes)}")
    print(f"Check your output at: {current_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Center-crop and resize aligned images."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Process only a specific scene (relative path like 'HandShake Method/scene_001')"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TARGET_SIZE,
        help=f"Target resize dimension (default: {TARGET_SIZE})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(target_scene=args.scene, target_size=args.size)


if __name__ == "__main__":
    main()
