import os
import cv2
import json
import csv
import shutil
import numpy as np

DECODED_ROOT = "decoded_frames"
DATASET_ROOT = "dataset"
LOG_ROOT = "logs"
LAPLACIAN_LOG_DIR = os.path.join(LOG_ROOT, "laplacian")
SCENE_LOG_FILE = os.path.join(LOG_ROOT, "scene_selection_log.csv")
STATE_FILE = os.path.join(DATASET_ROOT, "dataset_state.json")

MIN_BLUR_LENGTH = 4


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


def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_scores(folder, capture_name, mode):

    frames = sorted(os.listdir(folder))
    scores = []

    for frame in frames:
        path = os.path.join(folder, frame)
        img = cv2.imread(path)
        if img is None:
            continue

        score = laplacian_score(img)
        scores.append((frame, score))

    log_path = os.path.join(LAPLACIAN_LOG_DIR, f"{capture_name}_{mode}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score"])
        for frame, score in scores:
            writer.writerow([frame, score])

    return scores


def detect_segments(scores):

    values = [s for _, s in scores]
    sharp_index = int(np.argmax(values))
    sharp_score = values[sharp_index]

    threshold = sharp_score * 0.4

    blur_segments = []
    start = None

    for i, val in enumerate(values):

        if val < threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= MIN_BLUR_LENGTH:
                    blur_segments.append((start, i - 1))
                start = None

    if start is not None and len(values) - start >= MIN_BLUR_LENGTH:
        blur_segments.append((start, len(values) - 1))

    return sharp_index, blur_segments


def pick_blur_frame(scores, segment):

    start, end = segment
    segment_scores = scores[start:end + 1]

    frame = min(segment_scores, key=lambda x: x[1])[0]

    return frame


def get_frame(scores, index):
    return scores[index][0]


def copy_frame(src_folder, frame_name, dst_path):

    src = os.path.join(src_folder, frame_name)
    shutil.copy(src, dst_path)


def append_scene_log(row):

    file_exists = os.path.exists(SCENE_LOG_FILE)

    with open(SCENE_LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "scene",
                "capture",
                "ois_sharp",
                "ois_blur",
                "nonois_sharp",
                "nonois_blur"
            ])

        writer.writerow(row)


def process_capture(capture_name, state):

    print("Processing", capture_name)

    cap_path = os.path.join(DECODED_ROOT, capture_name)

    ois_folder = os.path.join(cap_path, "ois")
    nonois_folder = os.path.join(cap_path, "nonois")

    ois_scores = compute_scores(ois_folder, capture_name, "ois")
    nonois_scores = compute_scores(nonois_folder, capture_name, "nonois")

    sharp_index, blur_segments = detect_segments(ois_scores)

    ois_sharp = get_frame(ois_scores, sharp_index)
    nonois_sharp = get_frame(nonois_scores, sharp_index)

    for segment in blur_segments:

        scene_id = state["next_scene_id"]
        scene_name = f"scene_{scene_id:03d}"

        scene_path = os.path.join(DATASET_ROOT, scene_name)
        os.makedirs(scene_path, exist_ok=True)

        ois_blur = pick_blur_frame(ois_scores, segment)
        nonois_blur = pick_blur_frame(nonois_scores, segment)

        copy_frame(ois_folder, ois_sharp, os.path.join(scene_path, "ois_sharp.jpg"))
        copy_frame(ois_folder, ois_blur, os.path.join(scene_path, "ois_blur.jpg"))

        copy_frame(nonois_folder, nonois_sharp, os.path.join(scene_path, "nonois_sharp.jpg"))
        copy_frame(nonois_folder, nonois_blur, os.path.join(scene_path, "nonois_blur.jpg"))

        append_scene_log([
            scene_name,
            capture_name,
            ois_sharp,
            ois_blur,
            nonois_sharp,
            nonois_blur
        ])

        print("Created", scene_name)

        state["next_scene_id"] += 1

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


if __name__ == "__main__":
    main()