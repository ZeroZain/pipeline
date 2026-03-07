import os
import cv2
import csv

# Configuration

GT_SOURCE = "ois"  # "ois" or "nonois"

INPUT_DIR = f"aligned/gt_{GT_SOURCE}/color"
OUTPUT_DIR = f"dataset_256/gt_{GT_SOURCE}"

TARGET_SIZE = 256

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "interpolation_log.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Center crop function

def center_crop(img):
    h, w = img.shape[:2]
    size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    cropped = img[start_y:start_y + size, start_x:start_x + size]

    return cropped


# Log storage

log_rows = []

# Process scenes

scenes = sorted(os.listdir(INPUT_DIR))

for scene in scenes:

    scene_path = os.path.join(INPUT_DIR, scene)

    if not os.path.isdir(scene_path):
        continue

    output_scene = os.path.join(OUTPUT_DIR, scene)
    os.makedirs(output_scene, exist_ok=True)

    for img_name in os.listdir(scene_path):

        img_path = os.path.join(scene_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        # Skip images smaller than the target resolution
        if min(h, w) < TARGET_SIZE:
            log_rows.append([
                scene,
                img_name,
                w,
                h,
                "SKIPPED_SMALL"
            ])
            continue

        # Center crop to square
        cropped = center_crop(img)

        # Resize using bicubic interpolation
        resized = cv2.resize(
            cropped,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        output_path = os.path.join(output_scene, img_name)

        cv2.imwrite(
            output_path,
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        log_rows.append([
            scene,
            img_name,
            w,
            h,
            "SUCCESS"
        ])


# Save interpolation log

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "scene",
        "image",
        "original_width",
        "original_height",
        "status"
    ])

    writer.writerows(log_rows)

print("Interpolation stage complete.")
print(f"Processed scenes: {len(scenes)}")
print(f"Log saved to: {LOG_FILE}")