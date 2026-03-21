import time
import os
import zipfile
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =====================================================
# === PATH CONFIGURATION ===============================
# =====================================================

# Google Drive synced folder (input)
WATCH_FOLDER = r"G:\My Drive\Thesis or Crisis\Videos\automationInput"

# Local staging folder (temporary processing)
STAGING_FOLDER = r"C:\Users\Windows 11\Documents\GitHub\pipeline\staging"

# Final output folder (clean dataset)
OUTPUT_FOLDER = r"C:\Users\Windows 11\Documents\GitHub\pipeline\decoded_frames"

# Track processed folders to avoid duplicates
processed_folders = set()


# =====================================================
# === WATCHER: DETECT NEW FOLDERS ======================
# =====================================================

class WatcherHandler(FileSystemEventHandler):
    """
    Watches the input folder and triggers pipeline
    when a new capture folder is created.
    """

    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path

            if folder_path in processed_folders:
                return

            print(f"\n[NEW FOLDER DETECTED] {folder_path}")

            handle_pipeline(folder_path)
            processed_folders.add(folder_path)


# =====================================================
# === HELPER: FIND ZIP FILES RECURSIVELY ===============
# =====================================================

def find_zip_files(folder):
    """
    Recursively find all .zip files inside a folder.
    Supports nested folder structures.
    """
    zip_files = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".zip"):
                zip_files.append(os.path.join(root, f))

    return zip_files


def format_duration(seconds):
    """
    Convert seconds into human-readable format (HH:MM:SS)
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# =====================================================
# === WAIT FOR COMPLETE UPLOAD =========================
# =====================================================

def wait_for_complete_capture(folder_path, timeout=300):
    """
    Wait until:
    - 'ois' and 'nonois' folders exist
    - ZIP files exist (even if nested)
    - ZIP files are fully uploaded (size stable)
    """

    print("[WAITING] Ensuring full capture upload is complete...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            ois_path = os.path.join(folder_path, "ois")
            nonois_path = os.path.join(folder_path, "nonois")

            # Check subfolders exist
            if not os.path.exists(ois_path) or not os.path.exists(nonois_path):
                print("[WAITING] Missing subfolders...")
                time.sleep(2)
                continue

            # Find zip files recursively
            ois_zips = find_zip_files(ois_path)
            nonois_zips = find_zip_files(nonois_path)

            if not ois_zips or not nonois_zips:
                print("[WAITING] ZIP files not yet available...")
                time.sleep(2)
                continue

            # Check file stability
            all_ready = True

            for zip_list in [ois_zips, nonois_zips]:
                for file_path in zip_list:
                    size1 = os.path.getsize(file_path)
                    time.sleep(1)
                    size2 = os.path.getsize(file_path)

                    if size1 == 0 or size1 != size2:
                        all_ready = False

            if all_ready:
                print("[READY] Capture fully uploaded and stable")
                return True

        except Exception as e:
            print(f"[WAIT ERROR] {e}")

        time.sleep(2)

    print("[TIMEOUT] Capture may be incomplete")
    return False


# =====================================================
# === MAIN PIPELINE ===================================
# =====================================================

def handle_pipeline(src_path):
    """
    Main pipeline:
    1. Wait for upload completion
    2. Copy to staging
    3. Process (extract + clean)
    4. Move to output
    5. Measure execution time
    """

    folder_name = os.path.basename(src_path)

    print(f"[DEBUG] Starting pipeline for {folder_name}")

    # ⏱️ START TIMER
    start_time = time.time()

    # Step 1: Wait for upload completion
    wait_for_complete_capture(src_path)

    # Step 2: Copy to staging
    staging_path = copy_to_staging(src_path)

    if not staging_path:
        print("[ERROR] Copy failed, skipping")
        return

    # Step 3: Process files
    process_capture_folder(staging_path)

    # Step 4: Move to output
    move_to_output(staging_path)

    # ⏱️ END TIMER
    end_time = time.time()
    duration = end_time - start_time

    formatted_time = format_duration(duration)

    print(f"[TIME] {folder_name} processed in {formatted_time}\n")


# =====================================================
# === COPY TO STAGING =================================
# =====================================================

def copy_to_staging(src_path):
    """
    Copy capture folder from Drive to local staging
    """

    folder_name = os.path.basename(src_path)
    dst_path = os.path.join(STAGING_FOLDER, folder_name)

    print(f"[COPYING] {folder_name} → staging")

    if os.path.exists(dst_path):
        print("[SKIP] Already copied")
        return dst_path

    try:
        shutil.copytree(src_path, dst_path)
    except Exception as e:
        print(f"[ERROR COPYING] {e}")
        return None

    print("[COPIED] Successfully")
    return dst_path


# =====================================================
# === PROCESS EXISTING FOLDERS =========================
# =====================================================

def process_existing_folders():
    """
    Process folders that already exist before watcher starts
    """

    print("[INIT] Checking existing folders...")

    for folder in sorted(os.listdir(WATCH_FOLDER)):
        folder_path = os.path.join(WATCH_FOLDER, folder)

        if os.path.isdir(folder_path):
            if folder_path in processed_folders:
                continue

            print(f"[FOUND EXISTING] {folder_path}")
            handle_pipeline(folder_path)
            processed_folders.add(folder_path)


# =====================================================
# === PROCESS CAPTURE =================================
# =====================================================

def process_capture_folder(capture_path):
    """
    Extract all ZIP files (even nested) and clean output
    """

    print(f"[PROCESSING] {capture_path}")

    for subfolder in ["ois", "nonois"]:
        subfolder_path = os.path.join(capture_path, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"[WARNING] Missing {subfolder}")
            continue

        # Find all zip files recursively
        zip_files = find_zip_files(subfolder_path)

        # Extract each zip into root subfolder
        for zip_path in zip_files:
            extract_zip(zip_path, subfolder_path)

        # Clean unwanted files
        clean_non_dng_files(subfolder_path)


# =====================================================
# === EXTRACT + FLATTEN ================================
# =====================================================

def extract_zip(zip_path, extract_to):
    """
    Extract ZIP and flatten nested structure
    """

    print(f"[EXTRACTING] {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        print(f"[ERROR] Failed to extract: {e}")
        return

    extracted_name = os.path.splitext(os.path.basename(zip_path))[0]
    extracted_folder = os.path.join(extract_to, extracted_name)

    # Flatten structure
    if os.path.exists(extracted_folder):
        for item in os.listdir(extracted_folder):
            shutil.move(
                os.path.join(extracted_folder, item),
                os.path.join(extract_to, item)
            )

        shutil.rmtree(extracted_folder)
        print(f"[FLATTENED] {extracted_folder}")


# =====================================================
# === CLEAN NON-DNG FILES ==============================
# =====================================================

def clean_non_dng_files(folder_path):
    """
    Remove all non-.dng files (including .zip, .ini, .wav)
    """

    print(f"[CLEANING] {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(".dng"):
                try:
                    os.remove(os.path.join(root, file))
                except:
                    pass


# =====================================================
# === MOVE TO OUTPUT ==================================
# =====================================================

def move_to_output(staging_path):
    """
    Move processed folder from staging to final output
    """

    folder_name = os.path.basename(staging_path)
    destination = os.path.join(OUTPUT_FOLDER, folder_name)

    if os.path.exists(destination):
        print("[SKIP] Already exists in output")
        return

    try:
        shutil.move(staging_path, destination)

        # Clean again (handles desktop.ini)
        clean_non_dng_files(destination)

        print(f"[DONE] Moved → {destination}")
        print(f"[COMPLETE] {folder_name} processed\n")

    except Exception as e:
        print(f"[ERROR MOVING] {e}")


# =====================================================
# === MAIN ============================================
# =====================================================

if __name__ == "__main__":
    # Process existing folders first
    process_existing_folders()

    # Start watcher for new folders
    event_handler = WatcherHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    print("Watching for new capture folders...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()