import os
import shutil
import re
from datetime import datetime

# =========================
# PATH CONFIG
# =========================

DRIVE_BASE = r"G:\My Drive\Thesis or Crisis\Videos\Dataset Capture"

BASE_PATH = os.path.join(
    os.path.expanduser("~"),
    "Documents",
    "GitHub",
    "pipeline",
    "workspace",
    "data",
    "encoded_frames"
)

INCOMING = os.path.join(BASE_PATH, "incoming")
OIS_PATH = os.path.join(INCOMING, "ois")
NONOIS_PATH = os.path.join(INCOMING, "nonois")

FAILED = os.path.join(BASE_PATH, "failed")
LOGS = os.path.join(BASE_PATH, "logs")

os.makedirs(FAILED, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# =========================
# SETTINGS
# =========================
methods = ["handshake", "sliding", "vibration"]

METHOD_MAP = {
    "handshake": "HandShake Method",
    "sliding": "Sliding Method",
    "vibration": "Vibration Method"
}

# =========================
# LOGGING
# =========================
log_file = os.path.join(LOGS, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(msg):
    """Write message to console and log file."""
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# =========================
# VALIDATION
# =========================
pattern = re.compile(r"(\d{6}_\d{6})_VIDEO_(\d{2}mm)")

def extract_key(name):
    """Extract timestamp key from folder name."""
    match = pattern.match(name)
    return match.group(1) if match else None

def parse_datetime(key):
    """Convert timestamp string to datetime object."""
    return datetime.strptime(key, "%y%m%d_%H%M%S")

def safe_move(src, dst):
    """Move file/folder safely without overwriting."""
    if not os.path.exists(src):
        return
    if os.path.exists(dst):
        dst += "_dup"
    shutil.move(src, dst)

def move_failed(path, reason):
    """Move invalid or unmatched folder to failed directory."""
    if not os.path.exists(path):
        return
    dest = os.path.join(FAILED, os.path.basename(path))
    safe_move(path, dest)
    log(f"Moved to FAILED ({reason}): {path}")

# =========================
# LOAD
# =========================
def load_folder_map(path):
    """Load folders and detect duplicates based on timestamp."""
    data = {}
    duplicates = set()

    if not os.path.exists(path):
        log(f"Missing folder: {path}")
        return data, duplicates

    for f in os.listdir(path):
        full = os.path.join(path, f)

        key = extract_key(f)
        if not key:
            move_failed(full, "invalid_name")
            continue

        if key in data:
            duplicates.add(key)
            move_failed(full, "duplicate")
            continue

        data[key] = f

    return data, duplicates

# =========================
# MATCH WITH TOLERANCE
# =========================
def match_with_tolerance(ois_map, nonois_map, tolerance_sec=2):
    """Match OIS and NONOIS folders within time tolerance."""
    ois_items = sorted([(k, parse_datetime(k)) for k in ois_map.keys()], key=lambda x: x[1])
    nonois_items = sorted([(k, parse_datetime(k)) for k in nonois_map.keys()], key=lambda x: x[1])

    matched = []
    used_nonois = set()

    for ois_key, ois_time in ois_items:
        best_match = None
        best_diff = None

        for nonois_key, nonois_time in nonois_items:
            if nonois_key in used_nonois:
                continue

            diff = abs((ois_time - nonois_time).total_seconds())

            if diff <= tolerance_sec:
                if best_diff is None or diff < best_diff:
                    best_match = nonois_key
                    best_diff = diff

        if best_match:
            matched.append((ois_key, best_match))
            used_nonois.add(best_match)
            log(f"Matched {ois_key} ↔ {best_match} (Δ={best_diff}s)")
        else:
            move_failed(os.path.join(OIS_PATH, ois_map[ois_key]), "no_close_match")

    # Remaining nonois unmatched
    for nonois_key in nonois_map:
        if nonois_key not in used_nonois:
            move_failed(os.path.join(NONOIS_PATH, nonois_map[nonois_key]), "no_close_match")

    return matched

# =========================
# LOAD DATA
# =========================
ois_map, dup1 = load_folder_map(OIS_PATH)
nonois_map, dup2 = load_folder_map(NONOIS_PATH)

# =========================
# MATCH PAIRS
# =========================
matched_pairs = match_with_tolerance(ois_map, nonois_map, tolerance_sec=2)
log(f"Valid pairs: {len(matched_pairs)}")

# =========================
# CAPTURE INDEX
# =========================
def get_existing_max_index(path):
    """Return the highest existing capture index in a folder, or 0 if none."""
    if not os.path.exists(path):
        return 0
    nums = [
        int(f.split("_")[1])
        for f in os.listdir(path)
        if f.startswith("capture_") and "_" in f
    ]
    return max(nums) if nums else 0

# =========================
# USER INPUT — PER-METHOD START INDEX
# =========================
print("\n--- Capture Start Index Configuration ---")
print("For each method, enter the desired starting capture index.")
print("The script will skip to the next available index if existing captures are found.\n")

capture_counters = {}

for method_key in methods:
    drive_method_folder = METHOD_MAP[method_key]
    method_path = os.path.join(DRIVE_BASE, drive_method_folder)

    existing_max = get_existing_max_index(method_path)
    if existing_max > 0:
        print(f"  [{drive_method_folder}] Existing captures detected (up to capture_{existing_max:03d})")

    while True:
        raw = input(f"  Start index for {drive_method_folder} (default {existing_max + 1}): ").strip()
        if not raw:
            start = existing_max + 1
            break
        try:
            start = int(raw)
            if start < 1:
                print("    Index must be >= 1.")
                continue
            break
        except ValueError:
            print("    Please enter a valid integer.")

    # Ensure we never overwrite existing captures
    effective_start = max(start, existing_max + 1)
    if effective_start != start:
        log(f"  ⚠ {drive_method_folder}: requested start {start} conflicts with existing captures. Using {effective_start} instead.")
    else:
        log(f"  {drive_method_folder}: starting at capture_{effective_start:03d}")

    capture_counters[method_key] = effective_start

print()

# =========================
# PROCESS
# =========================
for i, (ois_key, nonois_key) in enumerate(matched_pairs):
    method = methods[i % 3]
    drive_method_folder = METHOD_MAP[method]

    method_path = os.path.join(DRIVE_BASE, drive_method_folder)
    os.makedirs(method_path, exist_ok=True)

    idx = capture_counters[method]
    capture_counters[method] += 1

    capture = f"capture_{idx:03d}"
    capture_path = os.path.join(method_path, capture)

    ois_dest = os.path.join(capture_path, "ois")
    nonois_dest = os.path.join(capture_path, "nonois")

    os.makedirs(ois_dest, exist_ok=True)
    os.makedirs(nonois_dest, exist_ok=True)

    ois_src = os.path.join(OIS_PATH, ois_map[ois_key])
    nonois_src = os.path.join(NONOIS_PATH, nonois_map[nonois_key])

    shutil.move(ois_src, os.path.join(ois_dest, ois_map[ois_key]))
    shutil.move(nonois_src, os.path.join(nonois_dest, nonois_map[nonois_key]))

    log(f"Created {capture} ({drive_method_folder})")

log("Pipeline completed successfully")