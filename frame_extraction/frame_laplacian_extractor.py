import cv2
import numpy as np
import os

# input in terminal: pip install opencv-python numpy


# USER SETTINGS

OIS_VIDEO = "ois_video.mp4"          # change
NON_OIS_VIDEO = "non_ois_video.mp4"  # change
MAIN_OUTPUT = "dataset_frames"

FRAME_INTERVAL = 1               # extract every N frames
SHARPNESS_THRESHOLD = 100        # tune later
JPG_QUALITY = 95                 # JPEG quality (0–100)


# HELPER: CREATE NEXT SESSION FOLDER

def get_next_session_folder(base_path):
    os.makedirs(base_path, exist_ok=True)
    existing = [int(x) for x in os.listdir(base_path) if x.isdigit()]
    next_id = max(existing) + 1 if existing else 1
    session_path = os.path.join(base_path, str(next_id))
    return session_path


# LAPLACIAN SHARPNESS FUNCTION

def laplacian_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# VIDEO PROCESSING

def process_video(video_path, condition_name):
    print(f"\nProcessing {condition_name} video...")

    # Create condition folder
    condition_root = os.path.join(MAIN_OUTPUT, condition_name)
    session_folder = get_next_session_folder(condition_root)

    sharp_dir = os.path.join(session_folder, "sharp")
    blur_dir = os.path.join(session_folder, "blur")

    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blur_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

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

        filename = f"frame_{saved_idx:06d}_t{timestamp:.3f}_lap{sharpness_score:.1f}.jpg"

        if sharpness_score >= SHARPNESS_THRESHOLD:
            save_path = os.path.join(sharp_dir, filename)
            label = "SHARP"
        else:
            save_path = os.path.join(blur_dir, filename)
            label = "BLUR"

        # Save as JPG
        cv2.imwrite(
            save_path,
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]
        )

        print(f"{condition_name} | {filename} | {label}")

        saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Finished {condition_name}. Saved {saved_idx} frames.")


# RUN BOTH VIDEOS

process_video(OIS_VIDEO, "OIS")
process_video(NON_OIS_VIDEO, "NON_OIS")

print("\nAll processing complete.")