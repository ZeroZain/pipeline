import os
import cv2
import csv
import re
from tqdm import tqdm  # Standard for progress tracking

# Configuration
GT_SOURCE = "ois"  # "ois" or "nonois"

INPUT_DIR = f"aligned/gt_{GT_SOURCE}/color"
OUTPUT_DIR = f"dataset_256/gt_{GT_SOURCE}"

TARGET_SIZE = 256

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "interpolation_log.csv")
SCENE_PATTERN = re.compile(r"^scene_(\d+)$")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def to_posix(path):
    return path.replace(os.sep, "/")


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

# Center crop function
def center_crop(img):
    h, w = img.shape[:2]
    
    # 0.8 takes the center 80%. Increase to 0.9 for less crop, 
    # or decrease to 0.7 if black borders still appear.
    crop_factor = 0.8 
    
    size = int(min(h, w) * crop_factor)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    cropped = img[start_y:start_y + size, start_x:start_x + size]
    return cropped

# Log storage
log_rows = []

# Process scenes
scenes = list_scene_dirs(INPUT_DIR)

# Wrap scenes in tqdm for the visual progress bar
for scene_rel, scene_path in tqdm(scenes, desc="Processing Scenes", unit="scene"):
    scene_key = to_posix(scene_rel)
    output_scene = os.path.join(OUTPUT_DIR, scene_rel)
    os.makedirs(output_scene, exist_ok=True)

    for img_name in os.listdir(scene_path):
        # 1. Define the output path
        output_path = os.path.join(output_scene, img_name)

        # 2. Skip if the file already exists (Resume logic)
        if os.path.exists(output_path):
            continue

        img_path = os.path.join(scene_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        # Skip images smaller than the target resolution
        if min(h, w) < TARGET_SIZE:
            log_rows.append([scene_key, img_name, w, h, "SKIPPED_SMALL"])
            continue

        # Center crop to remove black artifacts and make square
        cropped = center_crop(img)

        # Resize using bicubic interpolation
        resized = cv2.resize(
            cropped,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        # Save the processed image
        cv2.imwrite(
            output_path,
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        log_rows.append([scene_key, img_name, w, h, "SUCCESS"])

# Save/Update interpolation log
# Note: This will overwrite the log file with ONLY the newly processed files.
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    # Check if file is empty to write header
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        writer.writerow(["scene", "image", "original_width", "original_height", "status"])
    writer.writerows(log_rows)

print("\nInterpolation stage complete.")
print(f"Processed scenes: {len(scenes)}")
print(f"Check your output at: {OUTPUT_DIR}")
