import os
import cv2
import json
import csv
import shutil
import numpy as np
import rawpy

DECODED_ROOT = "decoded_frames"
DATASET_ROOT = "dataset"
LOG_ROOT = "logs"

LAPLACIAN_LOG_DIR = os.path.join(LOG_ROOT, "laplacian")
SCENE_LOG = os.path.join(LOG_ROOT, "scene_selection_log.csv")
SHARP_LOG = os.path.join(LOG_ROOT, "sharp_usage_log.csv")

STATE_FILE = os.path.join(DATASET_ROOT, "dataset_state.json")

MIN_BLUR_FRAMES = 4


def ensure_dirs():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(LAPLACIAN_LOG_DIR, exist_ok=True)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "next_scene_id": 1,
            "processed_captures": []
        }

    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def read_image(path):

    ext = os.path.splitext(path)[1].lower()

    if ext == ".dng":
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return img
        except:
            return None

    else:
        return cv2.imread(path)


def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_frames(folder, capture_name, cam_type):

    files = sorted(os.listdir(folder))
    scores = []

    for f in files:

        path = os.path.join(folder, f)
        img = read_image(path)

        if img is None:
            continue

        score = laplacian_score(img)
        scores.append((f, score))

    log_file = os.path.join(
        LAPLACIAN_LOG_DIR,
        f"{capture_name}_{cam_type}.csv"
    )

    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "score"])

        for f, s in scores:
            writer.writerow([f, s])

    return scores


def detect_segments(scores):

    values = np.array([s for _, s in scores])

    sharp_idx = int(np.argmax(values))
    sharp_score = values[sharp_idx]

    blur_threshold = sharp_score * 0.4

    segments = []
    current = []

    for i, v in enumerate(values):

        if v < blur_threshold:
            current.append(i)

        else:
            if len(current) >= MIN_BLUR_FRAMES:
                segments.append(current)

            current = []

    if len(current) >= MIN_BLUR_FRAMES:
        segments.append(current)

    return sharp_idx, segments


def append_csv(file, header, row):

    exists = os.path.exists(file)

    with open(file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not exists:
            writer.writerow(header)

        writer.writerow(row)


def build_scene(scene_id,
                capture_name,
                sharp_frame,
                blur_frame,
                ois_dir,
                nonois_dir):

    scene_name = f"scene_{scene_id:03d}"
    scene_path = os.path.join(DATASET_ROOT, scene_name)

    os.makedirs(scene_path, exist_ok=True)

    ext = os.path.splitext(sharp_frame)[1]

    ois_sharp_src = os.path.join(ois_dir, sharp_frame)
    ois_blur_src = os.path.join(ois_dir, blur_frame)

    nonois_sharp_src = os.path.join(nonois_dir, sharp_frame)
    nonois_blur_src = os.path.join(nonois_dir, blur_frame)

    shutil.copy(ois_sharp_src, os.path.join(scene_path, f"ois_sharp{ext}"))
    shutil.copy(ois_blur_src, os.path.join(scene_path, f"ois_blur{ext}"))

    shutil.copy(nonois_sharp_src,
                os.path.join(scene_path, f"nonois_sharp{ext}"))
    shutil.copy(nonois_blur_src,
                os.path.join(scene_path, f"nonois_blur{ext}"))

    append_csv(
        SCENE_LOG,
        ["scene", "capture",
         "ois_sharp", "ois_blur",
         "nonois_sharp", "nonois_blur"],
        [scene_name, capture_name,
         sharp_frame, blur_frame,
         sharp_frame, blur_frame]
    )

    return scene_name


def process_capture(capture_name, state):

    capture_path = os.path.join(DECODED_ROOT, capture_name)

    ois_dir = os.path.join(capture_path, "ois")
    nonois_dir = os.path.join(capture_path, "nonois")

    print(f"Processing {capture_name}")

    ois_scores = score_frames(ois_dir, capture_name, "ois")
    nonois_scores = score_frames(nonois_dir, capture_name, "nonois")

    if len(ois_scores) == 0:
        print("No valid frames detected.")
        return

    sharp_idx, blur_segments = detect_segments(ois_scores)

    sharp_frame = ois_scores[sharp_idx][0]

    scene_list = []

    for segment in blur_segments:

        blur_idx = segment[len(segment)//2]
        blur_frame = ois_scores[blur_idx][0]

        scene_id = state["next_scene_id"]

        scene_name = build_scene(
            scene_id,
            capture_name,
            sharp_frame,
            blur_frame,
            ois_dir,
            nonois_dir
        )

        scene_list.append(scene_name)

        state["next_scene_id"] += 1

    append_csv(
        SHARP_LOG,
        ["capture", "sharp_frame", "scenes"],
        [capture_name, sharp_frame, "|".join(scene_list)]
    )

    state["processed_captures"].append(capture_name)


def main():

    ensure_dirs()

    state = load_state()

    captures = sorted(os.listdir(DECODED_ROOT))

    for capture in captures:

        if capture in state["processed_captures"]:
            continue

        process_capture(capture, state)

        save_state(state)

    print("Dataset building complete.")


if __name__ == "__main__":
    main()