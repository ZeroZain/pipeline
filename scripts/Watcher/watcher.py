import time
import os
import zipfile
import shutil
from threading import Thread, Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =====================================================
# === PATH CONFIG =====================================
# =====================================================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")

WATCH_FOLDER = r"G:\My Drive\Thesis or Crisis\Videos\Dataset Capture"
STAGING_FOLDER = os.path.join(DATA_ROOT, "staging")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "decoded_frames")

# =====================================================
# === GLOBAL STATE ====================================
# =====================================================

processing_queue = []
queue_lock = Lock()
copy_lock = Lock()
queued_captures = set()
active_captures = set()
failed_captures = {}

retry_limit = 1
retry_counts = {}
CAPTURE_PREFIX = "capture_"

CAPTURE_METHODS = ("HandShake Method", "Sliding Method", "Vibration Method")
SCAN_INTERVAL = 15
UPLOAD_TIMEOUT = 300

# =====================================================
# === HELPER: FIND ZIP FILES ===========================
# =====================================================

def find_zip_files(folder):
    """Recursively find all .zip files"""
    zips = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".zip"):
                zips.append(os.path.join(root, f))
    return zips


# =====================================================
# === HELPER: FORMAT TIME ==============================
# =====================================================

def format_duration(seconds):
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def capture_id(path, root):
    return os.path.relpath(path, root)


def capture_key(path, root):
    return os.path.normcase(capture_id(path, root))


def is_within_root(path, root):
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    return path_abs == root_abs or path_abs.startswith(root_abs + os.sep)


def ensure_runtime_dirs():
    if not os.path.isdir(WATCH_FOLDER):
        raise FileNotFoundError(f"WATCH_FOLDER not found: {WATCH_FOLDER}")

    os.makedirs(STAGING_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def capture_subfolders(folder):
    return os.path.join(folder, "ois"), os.path.join(folder, "nonois")


def zip_details(folder, capture_root):
    details = []

    if not os.path.isdir(folder):
        return details

    for path in sorted(find_zip_files(folder)):
        try:
            details.append((
                os.path.relpath(path, capture_root),
                os.path.getsize(path),
                int(os.path.getmtime(path))
            ))
        except OSError:
            continue

    return details


def capture_signature(folder):
    ois, nonois = capture_subfolders(folder)
    return (
        os.path.isdir(ois),
        os.path.isdir(nonois),
        tuple(zip_details(ois, folder)),
        tuple(zip_details(nonois, folder))
    )


def inspect_capture(folder):
    ois, nonois = capture_subfolders(folder)
    ois_zips = zip_details(ois, folder)
    nonois_zips = zip_details(nonois, folder)

    if not os.path.isdir(folder):
        return "missing_capture", ()

    if not os.path.isdir(ois) or not os.path.isdir(nonois):
        return "missing_subfolders", capture_signature(folder)

    if not ois_zips:
        return "missing_ois_zips", capture_signature(folder)

    if not nonois_zips:
        return "missing_nonois_zips", capture_signature(folder)

    return "ready", (
        True,
        True,
        tuple(ois_zips),
        tuple(nonois_zips)
    )


# =====================================================
# === HELPER: CHECK PROCESSED ==========================
# =====================================================

def is_already_processed(path):
    """Check if folder already exists in output"""
    rel_path = capture_id(path, WATCH_FOLDER)
    return os.path.exists(os.path.join(OUTPUT_FOLDER, rel_path))


def find_capture_folders(root):
    captures = []

    category_roots = []

    for method_name in CAPTURE_METHODS:
        method_path = os.path.join(root, method_name)

        if os.path.isdir(method_path):
            category_roots.append(method_path)

    if category_roots:
        for method_path in category_roots:
            for name in sorted(os.listdir(method_path)):
                path = os.path.join(method_path, name)

                if os.path.isdir(path) and name.startswith(CAPTURE_PREFIX):
                    captures.append(path)

        return captures

    for current_root, dirs, _ in os.walk(root):
        dirs.sort()

        for name in dirs:
            if name.startswith(CAPTURE_PREFIX):
                captures.append(os.path.join(current_root, name))

        dirs[:] = [name for name in dirs if not name.startswith(CAPTURE_PREFIX)]

    return captures


# =====================================================
# === WAIT FOR UPLOAD =================================
# =====================================================

def wait_for_complete_capture(folder, timeout=300):
    """Wait until ZIP files exist and are stable"""

    start = time.time()

    while time.time() - start < timeout:
        try:
            status, _ = inspect_capture(folder)

            if status != "ready":
                time.sleep(2)
                continue

            ois, nonois = capture_subfolders(folder)
            zips = find_zip_files(ois) + find_zip_files(nonois)

            stable = True
            for z in zips:
                s1 = os.path.getsize(z)
                time.sleep(1)
                s2 = os.path.getsize(z)
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
    """Run full pipeline for one capture folder"""

    rel_path = capture_id(src_path, WATCH_FOLDER)
    print(f"[START] {rel_path}")

    start_time = time.time()

    try:
        if not wait_for_complete_capture(src_path, timeout=UPLOAD_TIMEOUT):
            raise Exception("Upload not complete (timeout)")

        # 🔒 Copy only one at a time (prevents Drive errors)
        with copy_lock:
            staging = copy_to_staging(src_path)

        if not staging:
            raise Exception("Copy failed")

        time.sleep(1)  # stabilize after copy

        process_capture_folder(staging)
        move_to_output(staging)

        duration = format_duration(time.time() - start_time)
        print(f"[DONE] {rel_path} ({duration})\n")
        return "done"

    except Exception as e:
        print(f"[ERROR] {rel_path}: {e}")

        count = retry_counts.get(rel_path, 0)
        if count < retry_limit:
            retry_counts[rel_path] = count + 1
            print(f"[RETRY] {rel_path}")
            return "retry"
        else:
            print(f"[FAILED] {rel_path}")
            return "failed"


# =====================================================
# === COPY ============================================
# =====================================================

def copy_to_staging(src):
    """Robust file-by-file copy with retry"""

    rel_path = capture_id(src, WATCH_FOLDER)
    dst = os.path.join(STAGING_FOLDER, rel_path)

    if os.path.exists(dst):
        return dst

    print(f"[COPY] {rel_path}")

    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        for root, dirs, files in os.walk(src):

            rel = os.path.relpath(root, src)
            target_dir = os.path.join(dst, rel)

            os.makedirs(target_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)

                success = False

                for attempt in range(3):  # retry per file
                    try:
                        shutil.copy2(src_file, dst_file)
                        success = True
                        break
                    except:
                        time.sleep(1)

                if not success:
                    raise Exception(f"Failed copying {file}")

        return dst

    except Exception as e:
        print(f"[COPY ERROR] {e}")

        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)

        return None


# =====================================================
# === PROCESS =========================================
# =====================================================

def process_capture_folder(path):
    """Extract ZIP files and clean output"""

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
    """Extract zip and flatten nested folder"""

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
    """Remove all non-DNG files"""

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

def move_to_output(staging):
    """Move processed folder to output"""

    rel_path = capture_id(staging, STAGING_FOLDER)
    dst = os.path.join(OUTPUT_FOLDER, rel_path)

    if os.path.exists(dst):
        if os.path.exists(staging) and is_within_root(staging, STAGING_FOLDER):
            shutil.rmtree(staging, ignore_errors=True)
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(staging, dst)
    clean_non_dng_files(dst)


# =====================================================
# === STATUS ==========================================
# =====================================================

def print_scan_summary(stats):
    print(
        "[SCAN] "
        f"found={stats['found']} "
        f"ready={stats['ready']} "
        f"enqueued={stats['enqueued']} "
        f"busy={stats['busy']} "
        f"processed={stats['processed']} "
        f"incomplete={stats['incomplete']} "
        f"failed_hold={stats['failed_hold']} "
        f"queued={stats['queued']} "
        f"active={stats['active']} "
        f"failed={stats['failed']}"
    )


def print_incomplete_items(stats):
    for rel_path, reason in stats.get("incomplete_items", []):
        print(f"[INCOMPLETE] {rel_path} ({reason})")


# =====================================================
# === QUEUE ===========================================
# =====================================================

def enqueue(path):
    """Add folder to queue safely"""

    key = capture_key(path, WATCH_FOLDER)

    with queue_lock:
        if key not in queued_captures and key not in active_captures:
            processing_queue.append(path)
            queued_captures.add(key)
            return True
    return False


def scan_folders():
    """Scan Drive and enqueue unprocessed folders"""

    stats = {
        "found": 0,
        "ready": 0,
        "enqueued": 0,
        "busy": 0,
        "processed": 0,
        "incomplete": 0,
        "failed_hold": 0,
        "queued": 0,
        "active": 0,
        "failed": 0,
        "incomplete_items": []
    }

    for path in find_capture_folders(WATCH_FOLDER):
        stats["found"] += 1
        key = capture_key(path, WATCH_FOLDER)

        if is_already_processed(path):
            with queue_lock:
                failed_captures.pop(key, None)
            stats["processed"] += 1
            continue

        status, signature = inspect_capture(path)

        if status != "ready":
            with queue_lock:
                failed_captures.pop(key, None)
            stats["incomplete_items"].append((capture_id(path, WATCH_FOLDER), status))
            stats["incomplete"] += 1
            continue

        stats["ready"] += 1

        with queue_lock:
            if failed_captures.get(key) == signature:
                stats["failed_hold"] += 1
                continue

            if key in failed_captures:
                failed_captures.pop(key, None)

        if enqueue(path):
            stats["enqueued"] += 1
        else:
            stats["busy"] += 1

    with queue_lock:
        stats["queued"] = len(processing_queue)
        stats["active"] = len(active_captures)
        stats["failed"] = len(failed_captures)

    return stats


# =====================================================
# === WORKER ==========================================
# =====================================================

def worker():
    """Worker thread that processes queue"""

    while True:
        with queue_lock:
            src = processing_queue.pop(0) if processing_queue else None

            if src:
                key = capture_key(src, WATCH_FOLDER)
                queued_captures.discard(key)
                active_captures.add(key)
            else:
                key = None

        if src:
            result = handle_pipeline(src)
            rel_path = capture_id(src, WATCH_FOLDER)
            signature = capture_signature(src)

            with queue_lock:
                active_captures.discard(key)

                if result == "retry" and key not in queued_captures:
                    processing_queue.append(src)
                    queued_captures.add(key)
                elif result == "failed":
                    failed_captures[key] = signature
                elif result == "done":
                    retry_counts.pop(rel_path, None)
                    failed_captures.pop(key, None)
        else:
            time.sleep(2)


# =====================================================
# === MAIN ============================================
# =====================================================

if __name__ == "__main__":

    ensure_runtime_dirs()
    print("Pipeline started...")

    # Optional watcher (not critical)
    observer = Observer()
    observer.schedule(FileSystemEventHandler(), WATCH_FOLDER, recursive=True)
    observer.start()

    # Start 2 workers (stable for Drive)
    for _ in range(2):
        Thread(target=worker, daemon=True).start()

    try:
        while True:
            stats = scan_folders()
            print_scan_summary(stats)
            print_incomplete_items(stats)
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("Stopping...")
        observer.stop()
        observer.join()
