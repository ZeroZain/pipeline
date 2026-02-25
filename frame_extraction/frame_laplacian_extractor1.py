import cv2
import numpy as np
import os

# input in terminal: pip install opencv-python numpy

# USER SETTINGS

VIDEO_ROOT = "dataset_videos"   # main video folder
OUTPUT_ROOT = "dataset_frames"

FRAME_INTERVAL = 1
SHARPNESS_THRESHOLD = 100
JPG_QUALITY = 95


# HELPER: CREATE NEXT SESSION FOLDER

def get_next_session_folder(base_path):
    os.makedirs(base_path, exist_ok=True)
    existing = [int(x) for x in os.listdir(base_path) if x.isdigit()]
    next_id = max(existing) + 1 if existing else 1
    return os.path.join(base_path, str(next_id))


# LAPLACIAN SHARPNESS FUNCTION

def laplacian_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# FRAME EXTRACTION PROCESS

def process_video(video_path, condition_name, pair_id):
    print(f"\nProcessing {condition_name} video for pair {pair_id}...")

    condition_root = os.path.join(OUTPUT_ROOT, condition_name)
    session_folder = get_next_session_folder(condition_root)

    sharp_dir = os.path.join(session_folder, "sharp")
    blur_dir = os.path.join(session_folder, "blur")

    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blur_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # safety fallback
    if fps == 0:
        fps = 30

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            continue

        sharpness_score = laplacian_sharpness(frame)
        timestamp = frame_idx / fps

        filename = (
            f"pair{pair_id}_frame_{saved_idx:06d}"
            f"_t{timestamp:.3f}_lap{sharpness_score:.1f}.jpg"
        )

        if sharpness_score >= SHARPNESS_THRESHOLD:
            save_path = os.path.join(sharp_dir, filename)
            label = "SHARP"
        else:
            save_path = os.path.join(blur_dir, filename)
            label = "BLUR"

        cv2.imwrite(
            save_path,
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]
        )

        print(f"{condition_name} | pair {pair_id} | {filename} | {label}")

        saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Finished {condition_name} pair {pair_id}. Saved {saved_idx} frames.")


# AUTO-PAIRING LOGIC

def main():
    ois_dir = os.path.join(VIDEO_ROOT, "OIS")
    non_ois_dir = os.path.join(VIDEO_ROOT, "NON_OIS")

    if not os.path.exists(ois_dir) or not os.path.exists(non_ois_dir):
        print("ERROR: OIS or NON_OIS folder missing.")
        return

    ois_files = {
        os.path.splitext(f)[0]: os.path.join(ois_dir, f)
        for f in os.listdir(ois_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    }

    non_ois_files = {
        os.path.splitext(f)[0]: os.path.join(non_ois_dir, f)
        for f in os.listdir(non_ois_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    }

    common_keys = sorted(set(ois_files.keys()) & set(non_ois_files.keys()))

    if not common_keys:
        print("No matching OIS/NON_OIS pairs found.")
        return

    print(f"Found {len(common_keys)} video pair(s).")

    for key in common_keys:
        process_video(ois_files[key], "OIS", key)
        process_video(non_ois_files[key], "NON_OIS", key)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()