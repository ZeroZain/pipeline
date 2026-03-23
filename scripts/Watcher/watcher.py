import time
import os
import zipfile
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =====================================================
# === PATH CONFIG =====================================
# =====================================================

WATCH_FOLDER = r"G:\My Drive\Thesis or Crisis\Videos\automationInput"
STAGING_FOLDER = r"C:\Users\Windows 11\Documents\GitHub\pipeline\staging"
OUTPUT_FOLDER = r"C:\Users\Windows 11\Documents\GitHub\pipeline\decoded_frames"

# Queue + state
processing_queue = []
is_processing = False
retry_limit = 2
retry_counts = {}


# =====================================================
# === HELPER: FIND ZIP FILES ===========================
# =====================================================

def find_zip_files(folder):
    """Find all .zip files recursively"""
    zip_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".zip"):
                zip_files.append(os.path.join(root, f))
    return zip_files


# =====================================================
# === HELPER: FORMAT TIME ==============================
# =====================================================

def format_duration(seconds):
    """Convert seconds → HH:MM:SS"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# =====================================================
# === HELPER: CHECK IF PROCESSED =======================
# =====================================================

def is_already_processed(folder_name):
    """Check if already exists in output"""
    return os.path.exists(os.path.join(OUTPUT_FOLDER, folder_name))


# =====================================================
# === WAIT FOR UPLOAD =================================
# =====================================================

def wait_for_complete_capture(folder_path, timeout=300):
    """Wait until zip files exist and are stable"""

    print("[WAITING] Checking upload completion...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            ois = os.path.join(folder_path, "ois")
            nonois = os.path.join(folder_path, "nonois")

            if not os.path.exists(ois) or not os.path.exists(nonois):
                time.sleep(2)
                continue

            zips = find_zip_files(ois) + find_zip_files(nonois)

            if not zips:
                time.sleep(2)
                continue

            stable = True
            for f in zips:
                s1 = os.path.getsize(f)
                time.sleep(1)
                s2 = os.path.getsize(f)
                if s1 == 0 or s1 != s2:
                    stable = False

            if stable:
                print("[READY] Upload complete")
                return True

        except:
            pass

        time.sleep(2)

    print("[TIMEOUT] Proceeding anyway")
    return False


# =====================================================
# === PIPELINE ========================================
# =====================================================

def handle_pipeline(src_path):
    """Full pipeline execution"""

    folder_name = os.path.basename(src_path)
    print(f"[START] {folder_name}")

    start_time = time.time()

    try:
        wait_for_complete_capture(src_path)

        staging_path = copy_to_staging(src_path)
        if not staging_path:
            raise Exception("Copy failed")

        process_capture_folder(staging_path)
        move_to_output(staging_path)

        duration = format_duration(time.time() - start_time)
        print(f"[SUCCESS] {folder_name} ({duration})\n")

    except Exception as e:
        print(f"[ERROR] {folder_name}: {e}")

        # Retry logic
        count = retry_counts.get(folder_name, 0)
        if count < retry_limit:
            retry_counts[folder_name] = count + 1
            print(f"[RETRY] {folder_name} (attempt {count + 1})")
            processing_queue.append(src_path)
        else:
            print(f"[FAILED] {folder_name} exceeded retry limit\n")


# =====================================================
# === COPY ============================================
# =====================================================

def copy_to_staging(src_path):
    """Copy to staging folder"""

    name = os.path.basename(src_path)
    dst = os.path.join(STAGING_FOLDER, name)

    if os.path.exists(dst):
        return dst

    print(f"[COPY] {name}")

    try:
        shutil.copytree(src_path, dst)
        return dst
    except Exception as e:
        print(f"[COPY ERROR] {e}")
        return None


# =====================================================
# === PROCESS =========================================
# =====================================================

def process_capture_folder(path):
    """Extract + clean"""

    for sub in ["ois", "nonois"]:
        sub_path = os.path.join(path, sub)

        if not os.path.exists(sub_path):
            continue

        zips = find_zip_files(sub_path)

        for z in zips:
            extract_zip(z, sub_path)

        clean_non_dng_files(sub_path)


# =====================================================
# === EXTRACT =========================================
# =====================================================

def extract_zip(zip_path, extract_to):
    """Extract zip and flatten"""

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
    except:
        return

    name = os.path.splitext(os.path.basename(zip_path))[0]
    folder = os.path.join(extract_to, name)

    if os.path.exists(folder):
        for item in os.listdir(folder):
            shutil.move(
                os.path.join(folder, item),
                os.path.join(extract_to, item)
            )
        shutil.rmtree(folder)


# =====================================================
# === CLEAN ===========================================
# =====================================================

def clean_non_dng_files(folder):
    """Remove non-DNG files"""

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".dng"):
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass


# =====================================================
# === MOVE ============================================
# =====================================================

def move_to_output(staging_path):
    """Move to final output"""

    name = os.path.basename(staging_path)
    dst = os.path.join(OUTPUT_FOLDER, name)

    if os.path.exists(dst):
        return

    shutil.move(staging_path, dst)
    clean_non_dng_files(dst)


# =====================================================
# === SCANNER =========================================
# =====================================================

def scan_for_new_folders():
    """Scan Drive and add new folders to queue"""

    for folder in os.listdir(WATCH_FOLDER):
        path = os.path.join(WATCH_FOLDER, folder)

        if not os.path.isdir(path):
            continue

        if is_already_processed(folder):
            continue

        if path not in processing_queue:
            print(f"[QUEUE] {folder}")
            processing_queue.append(path)


# =====================================================
# === QUEUE PROCESSOR ================================
# =====================================================

def process_queue():
    """Process one folder at a time"""

    global is_processing

    if is_processing or not processing_queue:
        return

    is_processing = True

    src_path = processing_queue.pop(0)
    handle_pipeline(src_path)

    is_processing = False


# =====================================================
# === MAIN ============================================
# =====================================================

if __name__ == "__main__":

    print("Pipeline started...")

    # Watcher (optional trigger)
    observer = Observer()
    observer.schedule(WatcherHandler(), WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            scan_for_new_folders()   # detect new
            process_queue()          # process one
            time.sleep(5)

    except KeyboardInterrupt:
        observer.stop()
        observer.join()