import time
import os
import zipfile
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =====================================================
# === PATH CONFIGURATION ===============================
# =====================================================

# Folder monitored from Google Drive (must be synced locally)
WATCH_FOLDER = r"G:\My Drive\Thesis or Crisis\Videos\automationInput"

# Temporary working folder (local, fast processing)
STAGING_FOLDER = r"C:\Users\Windows 11\Documents\GitHub\pipeline\staging"

# Final output folder (clean dataset)
OUTPUT_FOLDER = r"C:\Users\Windows 11\Documents\GitHub\pipeline\decoded_frames"

# Track already processed folders
processed_folders = set()


# =====================================================
# === WATCHER: DETECT NEW FOLDERS ======================
# =====================================================

class WatcherHandler(FileSystemEventHandler):
    def on_created(self, event):
        """
        Triggered when a new folder is created inside WATCH_FOLDER
        """
        if event.is_directory:
            folder_path = event.src_path

            # Avoid duplicate processing
            if folder_path in processed_folders:
                return

            print(f"\n[NEW FOLDER DETECTED] {folder_path}")

            handle_pipeline(folder_path)
            processed_folders.add(folder_path)


# =====================================================
# === WAIT UNTIL FULL UPLOAD IS COMPLETE ===============
# =====================================================

def wait_for_complete_capture(folder_path, timeout=300):
    """
    Wait until:
    - 'ois' and 'nonois' folders exist
    - Both contain ZIP files
    - ZIP files are fully uploaded (size stable)
    """

    print("[WAITING] Ensuring full capture upload is complete...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            ois_path = os.path.join(folder_path, "ois")
            nonois_path = os.path.join(folder_path, "nonois")

            # Step 1: Check subfolders exist
            if not os.path.exists(ois_path) or not os.path.exists(nonois_path):
                print("[WAITING] Missing subfolders...")
                time.sleep(2)
                continue

            # Step 2: Check ZIP files exist
            ois_zips = [f for f in os.listdir(ois_path) if f.endswith(".zip")]
            nonois_zips = [f for f in os.listdir(nonois_path) if f.endswith(".zip")]

            if not ois_zips or not nonois_zips:
                print("[WAITING] ZIP files not yet available...")
                time.sleep(2)
                continue

            # Step 3: Check if ZIP files are stable (fully uploaded)
            all_ready = True

            for folder in [ois_path, nonois_path]:
                for f in os.listdir(folder):
                    if f.endswith(".zip"):
                        file_path = os.path.join(folder, f)

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
    Full pipeline:
    1. Wait for upload completion
    2. Copy to staging
    3. Process (extract + clean)
    4. Move to output
    """

    print("[DEBUG] Starting pipeline...")

    # Step 1: Wait until upload is complete
    wait_for_complete_capture(src_path)

    # Step 2: Copy to staging
    staging_path = copy_to_staging(src_path)

    if not staging_path:
        print("[ERROR] Copy failed, skipping")
        return

    # Step 3: Process files
    process_capture_folder(staging_path)

    # Step 4: Move to final output
    move_to_output(staging_path)


# =====================================================
# === COPY TO STAGING =================================
# =====================================================

def copy_to_staging(src_path):
    """
    Copy capture folder from Drive → local staging
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
    Extract ZIPs and clean unwanted files
    """

    print(f"[PROCESSING] {capture_path}")

    for subfolder in ["ois", "nonois"]:
        subfolder_path = os.path.join(capture_path, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"[WARNING] Missing {subfolder}")
            continue

        # Extract all ZIP files
        for file in os.listdir(subfolder_path):
            if file.endswith(".zip"):
                zip_path = os.path.join(subfolder_path, file)
                extract_zip(zip_path, subfolder_path)

        # Remove unwanted files
        clean_non_dng_files(subfolder_path)


# =====================================================
# === EXTRACT + FLATTEN ================================
# =====================================================

def extract_zip(zip_path, extract_to):
    """
    Extract ZIP and flatten nested folder
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
    Remove all files except .dng
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
    Move processed data from staging → final output
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
        print(f"[COMPLETE] {folder_name} processed ✅\n")

    except Exception as e:
        print(f"[ERROR MOVING] {e}")


# =====================================================
# === MAIN ============================================
# =====================================================

if __name__ == "__main__":
    # Process existing folders first
    process_existing_folders()

    # Start watcher
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