import html
import json
import re
from pathlib import Path
import shutil
import sys
import subprocess
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rawpy
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Scene Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent.parent.parent / "workspace"

DATASET_TARGET = ROOT / "data" / "dataset_512" / "gt_ois"
ALIGNED_COLOR = ROOT / "data" / "aligned" / "gt_ois" / "color"
DECODED_FRAMES = ROOT / "data" / "decoded_frames"

EXPORT_ROOT = ROOT / "data"
SPLIT_OPTIONS = ["Unassigned", "Training", "Validation", "Testing"]
SPLIT_EXPORT_MAP = {
    "Training": "train",
    "Validation": "val",
    "Testing": "test",
}

# Stratified distribution targets (Figure 9) — 500 total pairs
STRATIFIED_QUOTAS = {
    "Training":   {"Handshake": 100, "Vibration": 100, "Sliding": 100},  # 300
    "Validation": {"Handshake": 34,  "Vibration": 33,  "Sliding": 33},   # 100
    "Testing":    {"Handshake": 34,  "Vibration": 33,  "Sliding": 33},   # 100
}

LOG_DIR = ROOT / "logs"
LAPLACIAN_DIR = LOG_DIR / "laplacian"
OVERRIDES_CSV = LOG_DIR / "scene_overrides.csv"
FRAME_OVERRIDE_LOG = LOG_DIR / "frame_override_log.csv"
LOG_BACKUP_DIR = LOG_DIR / "backups"
DATASET_ROOT = ROOT / "data" / "dataset"
SPLIT_ASSIGNMENTS_CSV = LOG_DIR / "split_assignments.csv"
FINAL_DATASET_ROOT = ROOT / "data" / "final_dataset"
SCENE_SETS_DIR = LOG_DIR / "scene_sets"

FLOW_THRESHOLD = 15.0
SSIM_THRESHOLD = 0.70

IMAGE_ORDER = [
    "ois_sharp.jpg",
    "ois_blur.jpg",
    "nonois_sharp.jpg",
    "nonois_blur.jpg",
]
IMAGE_LABELS = {
    "ois_sharp.jpg": "OIS Sharp",
    "ois_blur.jpg": "OIS Blur",
    "nonois_sharp.jpg": "Non-OIS Sharp",
    "nonois_blur.jpg": "Non-OIS Blur",
}

# Simplified frame groups with flattened layouts so we can allocate identical columns
FRAME_GROUPS = {
    "ois": [
        ("Sharp", "ois_sharp"),
        ("Drop", "ois_drop_frame_actual"),
        ("Blur", "ois_blur"),
    ],
    "nonois": [
        ("Sharp", "nonois_sharp"),
        ("Drop Actual", "nonois_drop_frame_actual"),
        ("Drop Fallback", "nonois_drop_frame"),
        ("Blur", "nonois_blur"),
    ],
}


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def split_rel_path(value):
    if value is None or pd.isna(value):
        return []
    return [part for part in re.split(r"[\\/]+", str(value)) if part]


def slugify_rel_path(value):
    parts = []
    for part in split_rel_path(value):
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_")
        parts.append(clean or "item")
    return "__".join(parts)


def parse_scene(scene):
    parts = split_rel_path(scene)
    scene_name = parts[-1] if parts else "unknown_scene"
    match = re.search(r"(\d+)$", scene_name)
    return {
        "split": "Unassigned",
        "method": parts[0].replace(" Method", "").replace("HandShake", "Handshake") if len(parts) > 0 else "Unknown",
        "scene_name": scene_name,
        "scene_number": int(match.group(1)) if match else 10**9,
    }


def format_num(value, digits=2):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def above(value, threshold):
    return value is not None and not pd.isna(value) and float(value) > threshold


def below(value, threshold):
    return value is not None and not pd.isna(value) and float(value) < threshold


def as_bool(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def bool_label(value):
    parsed = as_bool(value)
    if parsed is True:
        return "Yes"
    if parsed is False:
        return "No"
    return "-"


def extract_capture_number(capture):
    if capture is None or pd.isna(capture):
        return "-"
    match = re.search(r"capture_(\d+)", str(capture))
    return match.group(1) if match else capture


# ---- Overrides Handling ----
def load_overrides():
    if OVERRIDES_CSV.exists():
        try:
            df = pd.read_csv(OVERRIDES_CSV)
            if "reviewer" not in df.columns:
                df["reviewer"] = ""
            return df.drop_duplicates(subset=["scene", "override_type"], keep="last")
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame(columns=["scene", "override_type", "old_value", "new_value", "timestamp", "reviewer"])


def save_override(scene, override_type, old_value, new_value, reviewer=""):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{
        "scene": scene,
        "override_type": override_type,
        "old_value": old_value,
        "new_value": new_value,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reviewer": reviewer
    }])
    
    if OVERRIDES_CSV.exists():
        try:
            df = pd.read_csv(OVERRIDES_CSV)
            if "reviewer" not in df.columns:
                df["reviewer"] = ""
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(OVERRIDES_CSV, index=False)
            else:
                new_row.to_csv(OVERRIDES_CSV, mode="a", header=False, index=False)
        except pd.errors.EmptyDataError:
            new_row.to_csv(OVERRIDES_CSV, index=False)
    else:
        new_row.to_csv(OVERRIDES_CSV, index=False)


def save_assignment(scene, field, old_value, new_value, reviewer=""):
    """Append a split-assignment row to the dedicated assignments CSV."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{
        "scene": scene,
        "field": field,
        "old_value": old_value,
        "new_value": new_value,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reviewer": reviewer,
    }])
    if SPLIT_ASSIGNMENTS_CSV.exists():
        new_row.to_csv(SPLIT_ASSIGNMENTS_CSV, mode="a", header=False, index=False)
    else:
        new_row.to_csv(SPLIT_ASSIGNMENTS_CSV, index=False)


# ---- Scene Set Management ----
def list_scene_sets():
    SCENE_SETS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in SCENE_SETS_DIR.glob("*.json"))

def load_scene_set(name):
    path = SCENE_SETS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_scene_set(name, data):
    SCENE_SETS_DIR.mkdir(parents=True, exist_ok=True)
    data["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = SCENE_SETS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def delete_scene_set(name):
    path = SCENE_SETS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()

def create_new_scene_set(name, keep_scenes_df, creator=""):
    scenes = {}
    for _, row in keep_scenes_df.iterrows():
        scenes[row["scene"]] = {"status": "Included", "split": "Unassigned"}
    return {
        "name": name,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": creator,
        "scenes": scenes,
    }

def list_final_datasets():
    if not FINAL_DATASET_ROOT.exists():
        return []
    return sorted(d.name for d in FINAL_DATASET_ROOT.iterdir() if d.is_dir())

def export_dataset_v2(scene_set_data, source_root, dataset_name):
    """Export included scenes from a scene set into final_dataset/<dataset_name>/."""
    export_root = FINAL_DATASET_ROOT / dataset_name
    manifest = []
    included = {
        k: v for k, v in scene_set_data["scenes"].items()
        if str(v.get("status", "")).lower() in ["include", "included"] and v.get("split") in SPLIT_EXPORT_MAP
    }
    if not included:
        return pd.DataFrame()
    for ds_type in ["ois", "nonois"]:
        for split_key in SPLIT_EXPORT_MAP.values():
            (export_root / ds_type / split_key / "input").mkdir(parents=True, exist_ok=True)
            (export_root / ds_type / split_key / "target").mkdir(parents=True, exist_ok=True)
    for split_label, split_key in SPLIT_EXPORT_MAP.items():
        split_scenes = sorted(
            [s for s, d in included.items() if d["split"] == split_label],
            key=lambda s: parse_scene(s)["scene_number"],
        )
        for idx, scene in enumerate(split_scenes, start=1):
            number = f"{idx:03d}"
            scene_dir = resolve_scene_dir(source_root, scene)
            for ds_type, blur_file, sharp_file in [
                ("ois", "ois_blur.jpg", "ois_sharp.jpg"),
                ("nonois", "nonois_blur.jpg", "nonois_sharp.jpg"),
            ]:
                input_src = scene_dir / blur_file
                target_src = scene_dir / sharp_file
                input_dst = export_root / ds_type / split_key / "input" / f"{number}.jpg"
                target_dst = export_root / ds_type / split_key / "target" / f"{number}.jpg"
                input_ok = input_src.exists()
                target_ok = target_src.exists()
                if input_ok:
                    shutil.copy2(str(input_src), str(input_dst))
                if target_ok:
                    shutil.copy2(str(target_src), str(target_dst))
                manifest.append({
                    "scene": scene, "split": split_key, "dataset": ds_type,
                    "number": number, "input_exists": input_ok, "target_exists": target_ok,
                })
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(export_root / "manifest.csv", index=False)
    with open(export_root / "scene_set.json", "w", encoding="utf-8") as f:
        json.dump(scene_set_data, f, indent=2, ensure_ascii=False)
    return manifest_df


st.markdown(
    """
    <style>
    /* Theme-aware dashboard styles: use Streamlit theme variables */
    :root {
        --sb-bg: var(--backgroundColor);
        --sb-fg: var(--textColor);
        --sb-card: var(--secondaryBackgroundColor);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        color: var(--sb-fg);
        background: var(--sb-bg);
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.10);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        background: var(--sb-card);
        color: var(--sb-fg);
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .dashboard-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 12px;
        background: var(--sb-card);
        color: var(--sb-fg);
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .dashboard-card .muted { 
        color: var(--textColor); 
        opacity: 0.6; 
        font-weight: 600; 
        font-size: 0.75rem; 
        text-transform: uppercase; 
        letter-spacing: 0.8px;
    }
    .dashboard-badge { 
        display: inline-block; 
        margin: 0 0.4rem 0.4rem 0; 
        padding: 0.3rem 0.8rem; 
        border-radius: 999px; 
        font-size: 0.75rem; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
    }
    .stImage { border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    
    /* Clean Badge Colors */
    .badge-ok { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-info { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-warn { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-bad { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-muted { background: rgba(128, 128, 128, 0.15); color: var(--textColor); border: 1px solid rgba(128, 128, 128, 0.3); }
    .badge-neutral { background: var(--secondaryBackgroundColor); color: var(--textColor); border: 1px solid rgba(128, 128, 128, 0.2); }
    
    .badge-keep { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.5); }
    .badge-reject { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.5); }
    .badge-flag { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.5); }

    /* Bulletproof Alignment Fixes for Review Panel */
    div[data-testid="stHorizontalBlock"]:has(.review-action-panel-marker) {
        align-items: flex-end !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.review-action-panel-marker) div[data-testid="stTextInput"] input {
        height: 42px !important;
        min-height: 42px !important;
        box-sizing: border-box;
    }
    div[data-testid="stHorizontalBlock"]:has(.review-action-panel-marker) button {
        height: 42px !important;
        min-height: 42px !important;
        margin: 0 !important;
        box-sizing: border-box;
    }

    /* Responsive CSS Grid for Scorecards */
    .metric-grid-4 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-grid-2 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .scorecard-card {
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 8px;
        padding: 1rem;
        background: var(--secondaryBackgroundColor);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .scorecard-title {
        font-size: 0.75rem; 
        font-weight: 600; 
        color: var(--textColor);
        opacity: 0.6; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        margin-bottom: 0.3rem;
    }
    .scorecard-value {
        font-size: 1.6rem; 
        font-weight: 700; 
        color: var(--textColor);
        line-height: 1.2;
    }
    .scorecard-status {
        font-size: 0.8rem; 
        font-weight: 700; 
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_scene_dir(root, rel_path):
    parts = split_rel_path(rel_path)
    return root.joinpath(*parts) if parts else root


def laplacian_csv_path(capture, cam_type):
    if capture is None or pd.isna(capture):
        return None

    capture_text = str(capture)
    candidates = [
        LAPLACIAN_DIR / f"{slugify_rel_path(capture_text)}_{cam_type}.csv",
        LAPLACIAN_DIR
        / f"{capture_text.replace('/', '__').replace(chr(92), '__')}_{cam_type}.csv",
    ]
    return first_existing_path(candidates)


def has_named_images(root, scene):
    scene_dir = resolve_scene_dir(root, scene)
    return any((scene_dir / name).exists() for name in IMAGE_ORDER)


def file_signature(path):
    path = Path(path)
    if not path.exists():
        return (str(path), 0, 0)
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def log_signature():
    return tuple(
        file_signature(path)
        for path in (
            LOG_DIR / "geo_log.csv",
            LOG_DIR / "photo_log.csv",
            LOG_DIR / "color_log.csv",
            LOG_DIR / "scene_fail_log.csv",
            LOG_DIR / "scene_selection_log.csv",
            LOG_DIR / "interpolation_log.csv",
            LOG_DIR / "manual_review.csv",
            OVERRIDES_CSV,
            FRAME_OVERRIDE_LOG,
        )
    )


@st.cache_data(show_spinner=False)
def load_all_logs(_signature):
    return {
        "geo": safe_read_csv(LOG_DIR / "geo_log.csv"),
        "photo": safe_read_csv(LOG_DIR / "photo_log.csv"),
        "color": safe_read_csv(LOG_DIR / "color_log.csv"),
        "fail": safe_read_csv(LOG_DIR / "scene_fail_log.csv"),
        "selection": safe_read_csv(LOG_DIR / "scene_selection_log.csv"),
        "interpolation": safe_read_csv(LOG_DIR / "interpolation_log.csv"),
        "review": safe_read_csv(LOG_DIR / "manual_review.csv"),
        "frame_override": safe_read_csv(FRAME_OVERRIDE_LOG),
    }




def list_capture_frames(capture, cam_type):
    """Return sorted list of .dng filenames in decoded_frames/{capture}/{cam_type}/."""
    if capture is None or pd.isna(capture):
        return []
    cap_dir = resolve_scene_dir(DECODED_FRAMES, capture) / cam_type
    if not cap_dir.exists():
        return []
    return sorted(f.name for f in cap_dir.iterdir() if f.suffix.lower() == ".dng")


def save_frame_override_log(scene, image_type, old_frame, new_frame, reviewer, action="frame_override"):
    """Append one row to frame_override_log.csv."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "scene": scene,
        "image_type": image_type,
        "old_frame": old_frame,
        "new_frame": new_frame,
        "reviewer": reviewer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
    }])
    if not FRAME_OVERRIDE_LOG.exists():
        row.to_csv(FRAME_OVERRIDE_LOG, index=False)
    else:
        # Ensure file ends with a newline before appending to prevent corruption
        try:
            with open(FRAME_OVERRIDE_LOG, "r+b") as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        f.write(b"\n")
        except Exception:
            pass
        row.to_csv(FRAME_OVERRIDE_LOG, mode="a", header=False, index=False)


def update_selection_log(scene, updates: dict):
    """
    Update the scene_selection_log.csv in-place for a given scene.
    `updates` is a dict like {"ois_sharp": "frame123.dng", ...}.
    Makes a timestamped backup first.
    """
    sel_path = LOG_DIR / "scene_selection_log.csv"
    if not sel_path.exists():
        return False
    LOG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup = LOG_BACKUP_DIR / f"scene_selection_log_{timestamp}.csv"
    shutil.copy2(sel_path, backup)
    df = safe_read_csv(sel_path)
    if df.empty or "scene" not in df.columns:
        return False
    mask = df["scene"] == scene
    if not mask.any():
        return False
    for col, val in updates.items():
        if col in df.columns:
            df.loc[mask, col] = val
    df.to_csv(sel_path, index=False)
    return True


def run_scene_pipeline(scene, stage="both"):
    """
    Run alignment and/or interpolation for a single scene.
    stage: "align" | "interp" | "both"
    Returns (stdout, stderr, returncode).
    """
    script_root = Path(__file__).resolve().parent.parent.parent
    outputs = []
    rcode = 0

    if stage in ("align", "both"):
        cmd = [sys.executable, str(script_root / "scripts" / "Alignment" / "align4.py"), "--scene", scene]
        result = subprocess.run(cmd, cwd=str(script_root), capture_output=True, text=True)
        outputs.append(("Alignment", result.stdout, result.stderr))
        rcode = max(rcode, result.returncode)

    if stage in ("interp", "both"):
        cmd = [sys.executable, str(script_root / "scripts" / "Interpolation" / "interpolate.py"), "--scene", scene]
        result = subprocess.run(cmd, cwd=str(script_root), capture_output=True, text=True)
        outputs.append(("Interpolation", result.stdout, result.stderr))
        rcode = max(rcode, result.returncode)

    return outputs, rcode


@st.cache_data(show_spinner=False, max_entries=256)
def load_display_image(path_str):
    path = Path(path_str)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    try:
        if suffix == ".dng":
            with rawpy.imread(str(path)) as raw:
                image = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                )
        else:
            image = cv2.imread(str(path))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1400:
        scale = 1400 / max_dim
        image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def load_laplacian(capture, cam_type):
    path = laplacian_csv_path(capture, cam_type)
    if path is None:
        return None

    df = safe_read_csv(path)
    if df.empty or "frame" not in df.columns or "score" not in df.columns:
        return None

    df = df.copy()
    df["frame"] = df["frame"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df.dropna(subset=["score"])


def load_scene_row(df, scene):
    if df.empty or "scene" not in df.columns:
        return None
    rows = df[df["scene"] == scene]
    if rows.empty:
        return None
    return rows.iloc[0]


def compute_quality_score(flow, ssim, delta):
    if pd.isna(flow) or pd.isna(ssim) or pd.isna(delta):
        return None
    return 0.5 * float(flow) + 0.3 * (1 - float(ssim)) * 100 + 0.2 * float(delta)


def build_scene_summary(logs):
    scenes = sorted(
        {
            str(scene)
            for df in logs.values()
            if "scene" in df.columns
            for scene in df["scene"].dropna().astype(str)
        }
    )

    summary = pd.DataFrame({"scene": scenes})
    parsed = pd.DataFrame([parse_scene(scene) for scene in scenes])
    summary = pd.concat([summary, parsed], axis=1)

    geo = logs["geo"]
    if not geo.empty:
        geo_group = (
            geo.groupby("scene", as_index=False)
            .agg(
                mean_flow=("flow_roi_p90", "mean"),
                max_flow=("flow_roi_p90", "max"),
                geo_rows=("scene", "size"),
            )
        )
        summary = summary.merge(geo_group, on="scene", how="left")

    photo = logs["photo"]
    if not photo.empty:
        photo_group = (
            photo.groupby("scene", as_index=False)
            .agg(photo_before=("mean_before", "mean"), photo_after=("mean_after", "mean"))
        )
        summary = summary.merge(photo_group, on="scene", how="left")

    color = logs["color"]
    if not color.empty:
        color_group = (
            color.groupby("scene", as_index=False)
            .agg(
                min_ssim=("ssim", "min"),
                mean_ssim=("ssim", "mean"),
                mean_deltaE_after=("deltaE_after", "mean"),
            )
        )
        summary = summary.merge(color_group, on="scene", how="left")

    fail = logs["fail"]
    if not fail.empty:
        fail_group = (
            fail.groupby("scene", as_index=False)
            .agg(
                failed_count=("failed_count", "sum"),
                failed_phases=(
                    "failed_phases",
                    lambda s: "; ".join(
                        sorted(
                            {
                                part
                                for value in s.dropna().astype(str)
                                for part in value.split(";")
                                if part
                            }
                        )
                    ),
                ),
            )
        )
        summary = summary.merge(fail_group, on="scene", how="left")

    interp = logs["interpolation"]
    if not interp.empty:
        interp_group = (
            interp.groupby("scene", as_index=False)
            .agg(
                interpolation_rows=("scene", "size"),
                interpolation_success=(
                    "status",
                    lambda s: int(s.astype(str).str.upper().eq("SUCCESS").sum()),
                ),
            )
        )
        summary = summary.merge(interp_group, on="scene", how="left")

    selection = logs["selection"]
    if not selection.empty:
        selection_group = selection.drop_duplicates("scene")[
            ["scene", "capture", "nonois_used_fallback", "is_linear_scene"]
        ]
        summary = summary.merge(selection_group, on="scene", how="left")

    review = logs["review"]
    if not review.empty and "scene" in review.columns:
        review_group = review.groupby("scene", as_index=False).agg(review_count=("scene", "size"))
        summary = summary.merge(review_group, on="scene", how="left")

    # Apply Manual Overrides for Methods and Splits logically
    overrides = load_overrides()
    if not overrides.empty:
        split_overrides = overrides[overrides["override_type"] == "split"].set_index("scene")["new_value"].to_dict()
        method_overrides = overrides[overrides["override_type"] == "method"].set_index("scene")["new_value"].to_dict()
        summary["split"] = summary.apply(lambda row: split_overrides.get(row["scene"], row["split"]), axis=1)
        summary["method"] = summary.apply(lambda row: method_overrides.get(row["scene"], row["method"]), axis=1)

    defaults = {
        "max_flow": pd.NA,
        "min_ssim": pd.NA,
        "mean_deltaE_after": pd.NA,
        "failed_count": 0,
        "geo_rows": 0,
        "interpolation_success": 0,
        "capture": pd.NA,
        "nonois_used_fallback": pd.NA,
        "is_linear_scene": pd.NA,
        "review_count": 0,
    }
    for column, default in defaults.items():
        if column not in summary.columns:
            summary[column] = default

    summary["quality_score"] = summary.apply(
        lambda row: compute_quality_score(
            row.get("max_flow"),
            row.get("min_ssim"),
            row.get("mean_deltaE_after"),
        ),
        axis=1,
    )
    summary["has_dataset_target"] = summary["scene"].map(lambda s: has_named_images(DATASET_TARGET, s))
    summary["has_aligned_color"] = summary["scene"].map(lambda s: has_named_images(ALIGNED_COLOR, s))
    summary["has_ois_laplacian"] = summary["capture"].map(
        lambda c: laplacian_csv_path(c, "ois") is not None
    )
    summary["has_nonois_laplacian"] = summary["capture"].map(
        lambda c: laplacian_csv_path(c, "nonois") is not None
    )
    summary["interpolation_complete"] = summary["interpolation_success"].fillna(0).ge(4)
    summary["has_alignment_data"] = (
        summary["geo_rows"].fillna(0).gt(0)
        | summary["min_ssim"].notna()
        | summary["mean_deltaE_after"].notna()
        | summary["failed_count"].fillna(0).gt(0)
    )
    summary["needs_attention"] = (
        summary["max_flow"].fillna(-1).gt(FLOW_THRESHOLD)
        | summary["min_ssim"].fillna(1).lt(SSIM_THRESHOLD)
        | summary["failed_count"].fillna(0).gt(0)
    )
    summary["missing_assets"] = ~(
        summary["has_dataset_target"]
        & summary["has_aligned_color"]
        & summary["has_ois_laplacian"]
        & summary["has_nonois_laplacian"]
    )
    
    # Strictly filter to only the approved methods
    summary = summary[summary["method"].isin(["Sliding", "Vibration", "Handshake"])]
    
    return summary


# -------------------------------------------------------------------------------------
# Perceptual Hash & Scene Grouping
# -------------------------------------------------------------------------------------

def compute_phash(image_path, hash_size=8):
    """Compute perceptual hash for an image using DCT."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    return (dct_low > median).flatten()


def hamming_distance(h1, h2):
    """Compute Hamming distance between two binary hash arrays."""
    if h1 is None or h2 is None:
        return 64
    return int(np.sum(h1 != h2))


@st.cache_data(show_spinner="Computing scene fingerprints...")
def compute_all_phashes(_scenes_tuple, source_root_str):
    """Compute pHash for every scene's OIS Sharp image. Accepts a tuple for cacheability."""
    hashes = {}
    source_root = Path(source_root_str)
    for scene in _scenes_tuple:
        scene_dir = resolve_scene_dir(source_root, scene)
        img_path = scene_dir / "ois_sharp.jpg"
        if img_path.exists():
            h = compute_phash(img_path)
            if h is not None:
                hashes[scene] = h
    return hashes


def find_similar_pairs(hashes, threshold=14):
    """Return pairs of scenes with Hamming distance <= threshold, sorted by distance."""
    scenes = list(hashes.keys())
    pairs = []
    for i in range(len(scenes)):
        for j in range(i + 1, len(scenes)):
            dist = hamming_distance(hashes[scenes[i]], hashes[scenes[j]])
            if dist <= threshold:
                pairs.append((scenes[i], scenes[j], dist))
    return sorted(pairs, key=lambda x: x[2])


def cluster_pairs_into_groups(pairs):
    """Use Union-Find to cluster similar-scene pairs into groups."""
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s1, s2, _ in pairs:
        union(s1, s2)

    groups = {}
    for scene in parent:
        root = find(scene)
        if root not in groups:
            groups[root] = []
        groups[root].append(scene)

    result = {}
    for idx, (_, members) in enumerate(sorted(groups.items()), start=1):
        if len(members) >= 2:
            result[f"group_{idx:03d}"] = sorted(members)
    return result


# -------------------------------------------------------------------------------------
# Quota & Validation Helpers
# -------------------------------------------------------------------------------------

def compute_quota_status(scenes_data):
    """Return dict of {(split, method): count} for included+assigned scenes."""
    counts = {}
    for sc, data in scenes_data.items():
        status = str(data.get("status", "")).lower()
        if status not in ["include", "included"]:
            continue
        split = data.get("split", "Unassigned")
        if split not in SPLIT_EXPORT_MAP:
            continue
        method = parse_scene(sc)["method"]
        key = (split, method)
        counts[key] = counts.get(key, 0) + 1
    return counts


def compute_unassigned_by_method(scenes_data):
    """Return dict of {method: count} for included+unassigned scenes."""
    counts = {}
    for sc, data in scenes_data.items():
        status = str(data.get("status", "")).lower()
        if status not in ["include", "included"]:
            continue
        if data.get("split", "Unassigned") == "Unassigned":
            method = parse_scene(sc)["method"]
            counts[method] = counts.get(method, 0) + 1
    return counts


def validate_split_groups(scenes_data, groups):
    """Check if any group has members in different splits. Returns violations list."""
    violations = []
    for group_id, members in groups.items():
        splits_in_group = set()
        for sc in members:
            sc_data = scenes_data.get(sc, {})
            split = sc_data.get("split", "Unassigned")
            if split != "Unassigned":
                splits_in_group.add(split)
        if len(splits_in_group) > 1:
            violations.append({
                "group": group_id,
                "members": members,
                "splits": sorted(splits_in_group),
            })
    return violations


def validate_pre_export(scenes_data, groups, source_root):
    """Run all pre-export checks. Returns list of {check, status, detail} dicts."""
    checks = []
    counts = compute_quota_status(scenes_data)
    methods = ["Handshake", "Vibration", "Sliding"]
    splits = ["Training", "Validation", "Testing"]

    # 1. Quota check
    all_met = True
    quota_details = []
    for split in splits:
        for method in methods:
            current = counts.get((split, method), 0)
            target = STRATIFIED_QUOTAS.get(split, {}).get(method, 0)
            if current != target:
                all_met = False
                quota_details.append(f"{split}/{method}: {current}/{target}")
    checks.append({
        "check": "Quota targets met",
        "status": "pass" if all_met else "warn",
        "detail": "All cells match targets" if all_met else "; ".join(quota_details),
    })

    # 2. Total count
    total = sum(counts.values())
    checks.append({
        "check": "Total assigned = 500",
        "status": "pass" if total == 500 else "warn",
        "detail": f"{total}/500 assigned",
    })

    # 3. Group integrity
    violations = validate_split_groups(scenes_data, groups)
    checks.append({
        "check": "No cross-split group leaks",
        "status": "pass" if not violations else "fail",
        "detail": "All groups intact" if not violations else f"{len(violations)} group(s) split across sets",
    })

    # 4. Missing images
    missing = []
    for sc, data in scenes_data.items():
        status = str(data.get("status", "")).lower()
        if status not in ["include", "included"]:
            continue
        if data.get("split", "Unassigned") == "Unassigned":
            continue
        scene_dir = resolve_scene_dir(source_root, sc)
        for img in ["ois_blur.jpg", "ois_sharp.jpg", "nonois_blur.jpg", "nonois_sharp.jpg"]:
            if not (scene_dir / img).exists():
                missing.append(f"{sc}/{img}")
    checks.append({
        "check": "No missing image files",
        "status": "pass" if not missing else "fail",
        "detail": "All images present" if not missing else f"{len(missing)} file(s) missing",
    })

    return checks





import random

def shuffle_fill_all(scenes_data, groups):
    """Auto-distribute all unassigned+included scenes to meet quotas, treating groups atomically. Returns assignment dict."""
    assignments = {}
    scene_to_group = {}
    for gid, members in groups.items():
        for m in members:
            scene_to_group[m] = gid

    counts = compute_quota_status(scenes_data)
    
    # Collect all unassigned scenes into atomic units
    unassigned_units = [] # list of lists of scenes
    seen = set()
    for sc, d in scenes_data.items():
        if sc in seen:
            continue
        status = str(d.get("status", "")).lower()
        if status not in ["include", "included"]:
            continue
        if d.get("split", "Unassigned") != "Unassigned":
            continue
            
        gid = scene_to_group.get(sc)
        if gid:
            unit = [m for m in groups[gid] if m in scenes_data and scenes_data[m].get("split", "Unassigned") == "Unassigned" and str(scenes_data[m].get("status", "")).lower() in ["include", "included"]]
            for m in unit:
                seen.add(m)
            if unit:
                unassigned_units.append(unit)
        else:
            seen.add(sc)
            unassigned_units.append([sc])
            
    random.shuffle(unassigned_units)
    unassigned_units.sort(key=len, reverse=True) # Place larger groups first
    splits_order = ["Training", "Validation", "Testing"]
    methods = ["Handshake", "Vibration", "Sliding"]
    
    for unit in unassigned_units:
        # Find the best split to put this unit in
        # The best split is the one that needs these methods the most, WITHOUT overshooting
        best_split = None
        best_score = -9999
        
        for split in splits_order:
            fits_strictly = True
            for sc in unit:
                m = parse_scene(sc)["method"]
                if m in methods:
                    target = STRATIFIED_QUOTAS.get(split, {}).get(m, 0)
                    current = counts.get((split, m), 0)
                    if current + 1 > target:
                        fits_strictly = False
                        break
            
            if fits_strictly:
                score = 0
                for sc in unit:
                    m = parse_scene(sc)["method"]
                    if m in methods:
                        target = STRATIFIED_QUOTAS.get(split, {}).get(m, 0)
                        current = counts.get((split, m), 0)
                        needed = target - current
                        score += needed # Higher needed = better fit
                
                if score > best_score:
                    best_score = score
                    best_split = split
                
        if best_split:
            for sc in unit:
                assignments[sc] = best_split
                m = parse_scene(sc)["method"]
                counts[(best_split, m)] = counts.get((best_split, m), 0) + 1

    return assignments


def render_quota_dashboard(scenes_data):
    """Render the stratified distribution tracker as an HTML table."""
    counts = compute_quota_status(scenes_data)
    unassigned = compute_unassigned_by_method(scenes_data)
    methods = ["Handshake", "Vibration", "Sliding"]
    splits = ["Training", "Validation", "Testing"]

    rows_html = []
    for split in splits:
        cells = []
        total_c, total_t = 0, 0
        for method in methods:
            c = counts.get((split, method), 0)
            t = STRATIFIED_QUOTAS.get(split, {}).get(method, 0)
            total_c += c
            total_t += t
            pct = min(100, int(c / t * 100)) if t > 0 else 0
            if c > t:
                color, bar = "#ef4444", "#ef4444"
                extra = ' <span style="color:#ef4444;font-size:0.65rem;">OVER</span>'
            elif c == t:
                color, bar, extra = "#22c55e", "#22c55e", ""
            elif c >= t * 0.5:
                color, bar, extra = "#f59e0b", "#f59e0b", ""
            else:
                color = "#ef4444" if c > 0 else "var(--textColor)"
                bar = "rgba(128,128,128,0.3)"
                extra = ""
            cells.append(
                f'<td style="padding:0.5rem 0.6rem;text-align:center;">'
                f'<div style="font-weight:700;font-size:1rem;color:{color};">{c}<span style="opacity:0.4;font-weight:400;">/{t}</span>{extra}</div>'
                f'<div style="background:rgba(128,128,128,0.15);border-radius:999px;height:5px;margin-top:0.2rem;overflow:hidden;">'
                f'<div style="background:{bar};height:100%;width:{pct}%;border-radius:999px;"></div></div></td>'
            )
        tot_color = "#22c55e" if total_c >= total_t else "#f59e0b" if total_c >= total_t * 0.5 else "var(--textColor)"
        split_colors = {"Training": "#3b82f6", "Validation": "#8b5cf6", "Testing": "#f59e0b"}
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(128,128,128,0.08);">'
            f'<td style="padding:0.5rem 0.6rem;font-weight:700;color:{split_colors.get(split, "var(--textColor)")};">{split}</td>'
            + "".join(cells)
            + f'<td style="padding:0.5rem 0.6rem;text-align:center;font-weight:700;color:{tot_color};">{total_c}/{total_t}</td></tr>'
        )

    # Unassigned row
    u_cells = []
    total_u = 0
    for method in methods:
        cnt = unassigned.get(method, 0)
        total_u += cnt
        u_cells.append(f'<td style="padding:0.5rem 0.6rem;text-align:center;opacity:0.5;">{cnt}</td>')
    rows_html.append(
        f'<tr><td style="padding:0.5rem 0.6rem;font-weight:600;opacity:0.5;">Unassigned</td>'
        + "".join(u_cells)
        + f'<td style="padding:0.5rem 0.6rem;text-align:center;opacity:0.5;">{total_u}</td></tr>'
    )

    th_style = 'style="text-align:center;padding:0.4rem 0.6rem;font-size:0.7rem;text-transform:uppercase;opacity:0.4;letter-spacing:0.5px;"'
    st.markdown(
        '<div style="margin-bottom:1.2rem;padding:0.8rem 1rem;background:var(--secondaryBackgroundColor);'
        'border-radius:10px;border:1px solid rgba(128,128,128,0.15);">'
        '<div style="font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.5px;'
        'opacity:0.5;margin-bottom:0.6rem;">Stratified Distribution Tracker</div>'
        '<table style="width:100%;border-collapse:collapse;">'
        '<tr style="border-bottom:2px solid rgba(128,128,128,0.15);">'
        f'<th style="text-align:left;padding:0.4rem 0.6rem;font-size:0.7rem;text-transform:uppercase;opacity:0.4;">Split</th>'
        f'<th {th_style}>Handshake</th><th {th_style}>Vibration</th><th {th_style}>Sliding</th>'
        f'<th {th_style}>Total</th></tr>'
        + "".join(rows_html)
        + '</table></div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------------------
# Parsing, scoring, and UI helpers
# -------------------------------------------------------------------------------------

def section_header(title, level="h3", top_margin="2rem"):
    """Renders a clean, unified section header with a bottom border."""
    st.markdown(
        f"""
        <{level} style="
            border-bottom: 1px solid rgba(128,128,128,0.2); 
            padding-bottom: 0.4rem; 
            margin-top: {top_margin}; 
            margin-bottom: 1.2rem; 
            color: var(--textColor);
            font-weight: 600;
        ">{html.escape(title)}</{level}>
        """, 
        unsafe_allow_html=True
    )


def parse_failed_images_detail(failed_images_str):
    if failed_images_str is None or pd.isna(failed_images_str) or not str(failed_images_str).strip():
        return []
    results = []
    for entry in str(failed_images_str).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "|" in entry:
            image, phase_info = entry.split("|", 1)
            detail = ""
            if ":" in phase_info:
                phase_info, detail = phase_info.split(":", 1)
            for phase in phase_info.split("&"):
                results.append({"image": image.strip(), "phase": phase.strip(), "detail": detail.strip()})
        else:
            results.append({"image": entry, "phase": "unknown", "detail": ""})
    return results


def get_scene_review_map(review_df):
    if review_df.empty or "scene" not in review_df.columns:
        return {}
        
    df = review_df.copy()
    if "reviewer" not in df.columns:
        df["reviewer"] = "Unknown"
    else:
        df["reviewer"] = df["reviewer"].fillna("Unknown").replace("", "Unknown")
        
    latest_per_reviewer = df.drop_duplicates(subset=["scene", "reviewer"], keep="last")
    
    review_map = {}
    for scene, group in latest_per_reviewer.groupby("scene"):
        labels = group["label"].dropna().astype(str).str.upper().unique()
        if len(labels) == 0:
            continue
        elif len(labels) == 1:
            review_map[str(scene)] = labels[0]
        else:
            review_map[str(scene)] = "CONFLICT"
            
    return review_map


def compute_laplacian_deltas(sel_row, ois_df, nonois_df):
    deltas = {}
    for cam, df in [("ois", ois_df), ("nonois", nonois_df)]:
        if sel_row is None or df is None:
            deltas[cam] = {"sharp": None, "blur": None, "delta": None}
            continue
        sharp_frame = frame_value(sel_row, f"{cam}_sharp")
        blur_frame = frame_value(sel_row, f"{cam}_blur")
        sharp_score = metric_value(df, sharp_frame)
        blur_score = metric_value(df, blur_frame)
        delta = None
        if sharp_score is not None and blur_score is not None:
            delta = float(sharp_score) - float(blur_score)
        deltas[cam] = {"sharp": sharp_score, "blur": blur_score, "delta": delta}
    return deltas


def collect_failed_phases(fail_df):
    phases = set()
    if fail_df.empty or "failed_phases" not in fail_df.columns:
        return sorted(phases)
    for val in fail_df["failed_phases"].dropna().astype(str):
        for part in val.split(";"):
            part = part.strip()
            if part:
                phases.add(part)
    return sorted(phases)


def collect_failed_image_types(fail_df):
    images = set()
    if fail_df.empty or "failed_images" not in fail_df.columns:
        return sorted(images)
    for val in fail_df["failed_images"].dropna().astype(str):
        for detail in parse_failed_images_detail(val):
            base = detail["image"].replace(".dng", "").replace(".jpg", "")
            if base:
                images.add(base)
    return sorted(images)


def export_dataset(keep_scenes_df, source_root, export_root):
    manifest = []

    # Only create folders for splits that have assigned scenes
    assigned = keep_scenes_df[keep_scenes_df["split"] != "Unassigned"]
    if assigned.empty:
        return pd.DataFrame()

    for ds_type in ["ois", "nonois"]:
        for split_label in assigned["split"].unique():
            split_key = SPLIT_EXPORT_MAP.get(split_label, split_label.lower())
            (export_root / f"dataset_{ds_type}" / split_key / "input").mkdir(parents=True, exist_ok=True)
            (export_root / f"dataset_{ds_type}" / split_key / "target").mkdir(parents=True, exist_ok=True)

    for split_label in sorted(assigned["split"].unique()):
        split_key = SPLIT_EXPORT_MAP.get(split_label, split_label.lower())
        split_scenes = assigned[assigned["split"] == split_label].sort_values("scene_number")

        for idx, (_, row) in enumerate(split_scenes.iterrows(), start=1):
            scene = row["scene"]
            number = f"{idx:03d}"
            scene_dir = resolve_scene_dir(source_root, scene)

            for ds_type, blur_file, sharp_file in [
                ("ois", "ois_blur.jpg", "ois_sharp.jpg"),
                ("nonois", "nonois_blur.jpg", "nonois_sharp.jpg"),
            ]:
                input_src = scene_dir / blur_file
                target_src = scene_dir / sharp_file
                input_dst = export_root / f"dataset_{ds_type}" / split_key / "input" / f"{number}.jpg"
                target_dst = export_root / f"dataset_{ds_type}" / split_key / "target" / f"{number}.jpg"

                input_ok = input_src.exists()
                target_ok = target_src.exists()
                if input_ok:
                    shutil.copy2(str(input_src), str(input_dst))
                if target_ok:
                    shutil.copy2(str(target_src), str(target_dst))

                manifest.append({
                    "scene": scene, "split": split_key, "dataset": ds_type,
                    "number": number, "input_exists": input_ok, "target_exists": target_ok,
                })

    manifest_df = pd.DataFrame(manifest)
    manifest_path = LOG_DIR / "dataset_export_log.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def scene_image_notes(selection_row):
    if selection_row is None:
        return {}
    return {
        "ois_sharp.jpg": str(selection_row.get("ois_sharp", "")),
        "ois_blur.jpg": str(selection_row.get("ois_blur", "")),
        "nonois_sharp.jpg": str(selection_row.get("nonois_sharp", "")),
        "nonois_blur.jpg": str(selection_row.get("nonois_blur", "")),
    }


def render_scene_image_grid(title, root, scene, selection_row=None):
    section_header(title, level="h4", top_margin="1rem")
    scene_dir = resolve_scene_dir(root, scene)
    notes = scene_image_notes(selection_row)
    cols = st.columns(4)

    for col, name in zip(cols, IMAGE_ORDER):
        path = scene_dir / name
        with col:
            st.markdown(f"**{IMAGE_LABELS[name]}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
                note = notes.get(name)
                if note and note != "nan":
                    st.caption(note)
            else:
                st.warning(f"Missing: {name}")


def tag_html(text, tone):
    tone_classes = {
        "ok": "badge-ok",
        "info": "badge-info",
        "warn": "badge-warn",
        "bad": "badge-bad",
        "muted": "badge-muted",
        "neutral": "badge-neutral",
        "keep": "badge-keep",
        "reject": "badge-reject",
        "flag": "badge-flag",
        "conflict": "badge-bad",
    }
    cls = tone_classes.get(tone, "badge-neutral")
    return f'<span class="dashboard-badge {cls}">{html.escape(str(text))}</span>'


def render_scene_header(scene_row, filtered_count, filtered_index, latest_review=None):
    scene = scene_row["scene"]
    badges = []
    if scene_row["split"] != "Unassigned":
        badges.append(tag_html(scene_row["split"], "info"))
    badges.append(tag_html(scene_row["method"], "muted"))

    # Prepend prominent review status if it exists
    if latest_review == "KEEP":
        badges.insert(0, tag_html("REVIEW: KEEP", "keep"))
    elif latest_review == "REJECT":
        badges.insert(0, tag_html("REVIEW: REJECT", "reject"))
    elif latest_review == "FLAG":
        badges.insert(0, tag_html("REVIEW: FLAG", "flag"))
    elif latest_review == "CONFLICT":
        badges.insert(0, tag_html("REVIEW: CONFLICT", "conflict"))

    if above(scene_row.get("max_flow"), FLOW_THRESHOLD):
        badges.append(tag_html("Bad Alignment", "bad"))
    if below(scene_row.get("min_ssim"), SSIM_THRESHOLD):
        badges.append(tag_html("Low SSIM", "warn"))
    if above(scene_row.get("failed_count"), 0):
        badges.append(tag_html("Failed Checks", "bad"))

    st.markdown(
        (
            '<div class="dashboard-card">'
            '<div class="muted">Current Scene</div>'
            f'<div style="font-size:2.2rem;font-weight:800;margin-bottom:0.2rem;line-height:1.2;">{html.escape(scene_row["scene_name"])}</div>'
            f'<div class="muted" style="margin-bottom:1rem;font-size:0.85rem;text-transform:none;letter-spacing:normal;">{html.escape(scene)}</div>'
            f'<div>{"".join(badges)}</div>'
            "</div>"
            '<br>'
        ),
        unsafe_allow_html=True,
    )


def render_asset_metrics(asset_status):
    section_header("Asset Coverage", level="h4", top_margin="1rem")
    items = list(asset_status.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (label, available) in zip(cols, items[start:start + 3]):
            color = "#22c55e" if available else "#ef4444"
            status_text = "Ready" if available else "Missing"
            
            # Custom styled cards for asset status
            col.markdown(
                f"""
                <div style="border: 1px solid rgba(128,128,128,0.15); border-radius: 8px; padding: 1rem; background: var(--secondaryBackgroundColor); margin-bottom: 0.5rem;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: var(--textColor); opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem;">{label}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {color};">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def metric_value(df, frame):
    if df is None or frame is None or pd.isna(frame):
        return None
    match = df[df["frame"] == str(frame)]
    if match.empty:
        return None
    return match["score"].iloc[0]


def frame_value(row, *keys):
    if row is None:
        return None
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            text = str(value).strip()
            # Ignore empty or nan representations
            if text.lower() not in ["nan", "none", ""]:
                return text
    return None


def resolve_frame_path(capture, cam_type, frame):
    if capture is None or pd.isna(capture) or frame is None:
        return None
    capture_dir = resolve_scene_dir(DECODED_FRAMES, capture)
    return capture_dir / cam_type / frame


def render_selected_frames(capture, cam_type, lap_df, row):
    # Enforce exactly 4 uniform columns so containers are the exact same size for both sets of frames
    cols = st.columns(4)
    cam_label = "OIS" if cam_type == "ois" else "Non-OIS"

    # HTML-based placeholder that natively adapts to light and dark themes
    def _placeholder(text="NO FRAME"):
        st.markdown(
            f"""
            <div style="aspect-ratio: 3/2; width: 100%; display: flex; align-items: center; justify-content: center; background-color: rgba(128,128,128,0.1); border-radius: 6px; border: 1px dashed rgba(128,128,128,0.3); color: var(--textColor); opacity: 0.6; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px; margin-bottom: 1rem;">
                {text}
            </div>
            """, 
            unsafe_allow_html=True
        )

    for col, spec in zip(cols, FRAME_GROUPS[cam_type]):
        label = spec[0]
        frame_key = spec[1]
        
        with col:
            st.markdown(f"**{cam_label} {label}**")

            frame = frame_value(row, frame_key)
            path = resolve_frame_path(capture, cam_type, frame)
            score = metric_value(lap_df, frame)

            if path is not None and path.exists():
                image = load_display_image(str(path))
                if image is not None:
                    st.image(image, use_container_width=True)
                else:
                    _placeholder(text="CORRUPT")
            else:
                _placeholder()

            if frame:
                st.caption(frame)
            else:
                st.caption("No frame")
            
            st.caption(f"Laplacian score: {format_num(score)}")


def plot_laplacian(df, title, markers):
    if df is None or df.empty:
        st.warning(f"{title} log missing")
        return

    # normalize dataframe for plotting
    df_plot = df.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})

    # Aggregate labels for markers that fall on the exact same frame (prevents circles from hiding each other)
    frame_to_labels = {}
    frame_to_color = {}
    for label, frame, color in markers:
        if frame is None or str(frame).lower() in ["nan", "none", ""]:
            continue
        frame_str = str(frame)
        if frame_str in frame_to_labels:
            frame_to_labels[frame_str].append(label)
            # Give overlapped points a highly visible distinct amber color
            frame_to_color[frame_str] = "#d97706" 
        else:
            frame_to_labels[frame_str] = [label]
            frame_to_color[frame_str] = color

    marker_data = []
    frame_to_idx = {str(f): int(i) for i, f in enumerate(df_plot["frame"]) }
    
    for frame_str, labels in frame_to_labels.items():
        if frame_str in frame_to_idx:
            idx = frame_to_idx[frame_str]
            score = float(df_plot.iloc[idx]["score"])
            
            # Format combined labels cleanly
            if len(labels) > 1:
                if set(labels) == {"Drop Actual", "Drop Fallback"}:
                    combined_label = "Drop (Both)"
                else:
                    combined_label = " & ".join(labels)
            else:
                combined_label = labels[0]
                
            marker_data.append({
                "idx": idx, 
                "score": score, 
                "frame": frame_str, 
                "label": combined_label,
                "color": frame_to_color[frame_str]
            })

    # Build dynamic domain and color scale based on the actual present (and merged) markers
    domain = ["Laplacian"]
    range_colors = ["#8B949E"] # A clean slate color that works dynamically in both Light and Dark mode
    for m in marker_data:
        if m["label"] not in domain:
            domain.append(m["label"])
            range_colors.append(m["color"])

    try:
        import altair as alt

        df_plot = df_plot.copy()
        df_plot["label"] = "Laplacian"

        shared_scale = alt.Scale(domain=domain, range=range_colors)
        shared_legend = alt.Legend(
            title=None, 
            orient="bottom", 
            direction="horizontal",
            labelFontSize=12,
            symbolSize=100
        )

        line = (
            alt.Chart(df_plot)
            .mark_line(strokeWidth=1.5, opacity=0.8)
            .encode(
                x=alt.X("idx:Q", title="Frame Index"),
                y=alt.Y("score:Q", title="Laplacian Score"),
                color=alt.Color("label:N", scale=shared_scale, legend=shared_legend),
                tooltip=[alt.Tooltip("frame:N", title="Frame"), alt.Tooltip("score:Q", format=".3f", title="Score"), alt.Tooltip("idx:Q", title="Index")],
            )
        )

        points = (
            alt.Chart(df_plot)
            .mark_circle(size=40, opacity=0.4)
            .encode(
                x="idx:Q",
                y="score:Q",
                color=alt.Color("label:N", scale=shared_scale, legend=None),
                tooltip=["frame:N", alt.Tooltip("score:Q", format=".3f")]
            )
        )

        layers = [line, points]

        if marker_data:
            mdf = pd.DataFrame(marker_data)
            markers_chart = (
                alt.Chart(mdf)
                .mark_point(size=200, filled=True, opacity=1)
                .encode(
                    x="idx:Q",
                    y="score:Q",
                    color=alt.Color("label:N", scale=shared_scale, legend=shared_legend),
                    tooltip=[alt.Tooltip("label:N", title="Marker"), alt.Tooltip("frame:N", title="Frame"), alt.Tooltip("score:Q", format=".3f", title="Score")],
                )
            )
            layers.append(markers_chart)

        layered = alt.layer(*layers).properties(height=360, title=title).interactive()
        st.altair_chart(layered, use_container_width=True)

    except Exception:
        df_m = df.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        
        # Ensures matplotlib plot responds gracefully in Dark Mode environments
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.tick_params(colors='#8B949E')
        ax.xaxis.label.set_color('#8B949E')
        ax.yaxis.label.set_color('#8B949E')
        ax.title.set_color('#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#444444')

        ax.plot(range(len(df_m)), df_m["score"], linewidth=1.5, color="#8B949E", label="Laplacian", alpha=0.8)
        ax.grid(alpha=0.15)

        # Plot the aggregated markers
        for m in marker_data:
            ax.scatter(m["idx"], m["score"], s=120, color=m["color"], label=m["label"], zorder=3, edgecolors='none', linewidth=1)
            ax.annotate(
                m["label"],
                (m["idx"], m["score"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=m["color"] # Matches label color to the marker to be perfectly readable in both modes
            )

        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Laplacian Score")
        
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            # We set legend text color to adapt properly in dark mode
            legend = ax.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=min(4, len(by_label)))
            for text in legend.get_texts():
                text.set_color('#8B949E')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def build_selection_table(row):
    if row is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Camera": "OIS",
                "Sharp": frame_value(row, "ois_sharp"),
                "Drop (Actual)": frame_value(row, "ois_drop_frame_actual"),
                "Drop (Fallback)": "-",
                "Blur": frame_value(row, "ois_blur"),
            },
            {
                "Camera": "Non-OIS",
                "Sharp": frame_value(row, "nonois_sharp"),
                "Drop (Actual)": frame_value(row, "nonois_drop_frame_actual"),
                "Drop (Fallback)": frame_value(row, "nonois_drop_frame"),
                "Blur": frame_value(row, "nonois_blur"),
            },
        ]
    )


def build_laplacian_score_table(row, ois_df, nonois_df):
    if row is None:
        return pd.DataFrame()

    rows = []
    for cam_type, label in (("ois", "OIS"), ("nonois", "Non-OIS")):
        df = ois_df if cam_type == "ois" else nonois_df
        for spec in FRAME_GROUPS[cam_type]:
            stage = spec[0]
            frame = frame_value(row, spec[1])
            rows.append(
                {
                    "Camera": label,
                    "Stage": stage,
                    "Frame": frame,
                    "Score": metric_value(df, frame),
                }
            )
    return pd.DataFrame(rows)


def render_dataframe(df, empty_message):
    if df.empty:
        st.info(empty_message)
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def save_review(scene, label, note="", reviewer=""):
    review_path = LOG_DIR / "manual_review.csv"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[scene, label, note, reviewer, timestamp]], columns=["scene", "label", "note", "reviewer", "timestamp"])

    if review_path.exists():
        try:
            df = pd.read_csv(review_path)
            if "reviewer" not in df.columns or "timestamp" not in df.columns:
                if "reviewer" not in df.columns: df["reviewer"] = ""
                if "timestamp" not in df.columns: df["timestamp"] = ""
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(review_path, index=False)
            else:
                new_row.to_csv(review_path, mode="a", header=False, index=False)
        except pd.errors.EmptyDataError:
            new_row.to_csv(review_path, index=False)
    else:
        new_row.to_csv(review_path, index=False)


def delete_review_row(scene, global_idx):
    """Safely delete a single review record utilizing its global index."""
    review_path = LOG_DIR / "manual_review.csv"
    if review_path.exists():
        df = pd.read_csv(review_path)
        if global_idx in df.index:
            df = df.drop(index=global_idx)
            df.to_csv(review_path, index=False)


def clear_scene_review(scene):
    review_path = LOG_DIR / "manual_review.csv"
    if review_path.exists():
        df = pd.read_csv(review_path)
        # Filter out records associated with this specific scene
        df = df[df["scene"] != scene]
        df.to_csv(review_path, index=False)


def clear_all_global_reviews():
    review_path = LOG_DIR / "manual_review.csv"
    if review_path.exists():
        review_path.unlink()


logs = load_all_logs(log_signature())
summary = build_scene_summary(logs)

if summary.empty:
    st.title("Scene Dashboard")
    st.warning("No scenes found in workspace logs.")
    st.stop()

# ---- Sidebar Setup ----
status_container = st.sidebar.container()
selector_container = st.sidebar.container()
st.sidebar.markdown("---")
stats_container = st.sidebar.container()

# ---- User Settings ----
st.sidebar.header("User Settings")
reviewer_name = st.sidebar.selectbox("Reviewer", ["JC", "JJ", "JP"], key="reviewer_name", help="Select your name before reviewing or modifying scenes.")
st.sidebar.markdown("---")

# Build global review map for statistics
review_map = get_scene_review_map(logs["review"])
summary["global_review_status"] = summary["scene"].map(lambda s: review_map.get(s, "NOT REVIEWED"))

# Build active reviewer's personal map
review_df_safe = logs.get("review", pd.DataFrame())
my_review_map = {}
if not review_df_safe.empty and "scene" in review_df_safe.columns:
    rev_col = review_df_safe.get("reviewer", pd.Series(dtype=str)).fillna("Unknown").replace("", "Unknown")
    my_latest = review_df_safe[rev_col == reviewer_name].drop_duplicates(subset=["scene"], keep="last")
    if "label" in my_latest.columns:
        my_review_map = dict(zip(my_latest["scene"], my_latest["label"].str.upper()))

summary["my_review_status"] = summary["scene"].map(lambda s: my_review_map.get(s, "NOT REVIEWED"))

# Forward compatibility for global stats
summary["review_status"] = summary["global_review_status"]

# ---- Dashboard Statistics Header ----
total_scenes = len(summary)
reviewed_count = summary["review_status"].ne("NOT REVIEWED").sum()
keep_count = summary["review_status"].eq("KEEP").sum()
reject_count = summary["review_status"].eq("REJECT").sum()
flag_count = summary["review_status"].eq("FLAG").sum()
conflict_count = summary["review_status"].eq("CONFLICT").sum()
failed_count_total = logs["fail"]["scene"].dropna().astype(str).nunique() if "scene" in logs["fail"].columns else 0
progress_pct = int(reviewed_count / total_scenes * 100) if total_scenes > 0 else 0

st.markdown(
    f"""
    <br>
    <div style="margin-top: 1rem; margin-bottom: 2rem; padding: 1.6rem; background: var(--secondaryBackgroundColor); border-radius: 12px; border: 1px solid rgba(128,128,128,0.15);">
        <h1 style="font-size: 2.4rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: -0.5px; color: var(--textColor); line-height: 1.2;">Scene Dashboard</h1>
        <p style="font-size: 1.05rem; color: var(--textColor); opacity: 0.7; margin-bottom: 1rem;">Visual inspection & data cleaning workstation for pipeline outputs.</p>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 0.8rem;">
            <div><span style="font-size: 1.6rem; font-weight: 800; color: var(--textColor);">{total_scenes}</span> <span style="opacity: 0.6; font-size: 0.8rem; text-transform: uppercase;">Total</span></div>
            <div><span style="font-size: 1.6rem; font-weight: 800; color: #22c55e;">{keep_count}</span> <span style="opacity: 0.6; font-size: 0.8rem; text-transform: uppercase;">Keep</span></div>
            <div><span style="font-size: 1.6rem; font-weight: 800; color: #ef4444;">{reject_count}</span> <span style="opacity: 0.6; font-size: 0.8rem; text-transform: uppercase;">Reject</span></div>
            <div><span style="font-size: 1.6rem; font-weight: 800; color: #f59e0b;">{flag_count}</span> <span style="opacity: 0.6; font-size: 0.8rem; text-transform: uppercase;">Flag</span></div>
            <div><span style="font-size: 1.6rem; font-weight: 800; color: #ec4899;">{conflict_count}</span> <span style="opacity: 0.6; font-size: 0.8rem; text-transform: uppercase;">Conflict</span></div>
            <div><span style="font-size: 1.6rem; font-weight: 800; color: #8b5cf6;">{total_scenes - int(reviewed_count)}</span> <span style="opacity: 0.6; font-size: 0.8rem; text-transform: uppercase;">Unreviewed</span></div>
        </div>
        <div style="background: rgba(128,128,128,0.15); border-radius: 999px; height: 8px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #22c55e, #3b82f6); height: 100%; width: {progress_pct}%; border-radius: 999px; transition: width 0.3s;"></div>
        </div>
        <div style="font-size: 0.75rem; opacity: 0.5; margin-top: 0.3rem;">Review progress: {reviewed_count}/{total_scenes} ({progress_pct}%)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Setup/User Settings moved up

method_options = ["Sliding", "Vibration", "Handshake"]
st.sidebar.header("Filters")
selected_methods = st.sidebar.multiselect("Method", method_options, default=method_options)
scene_search = st.sidebar.text_input("Find scene", placeholder="scene_045 or vibration")
sort_mode = st.sidebar.selectbox(
    "Sort scenes by",
    [
        "Scene order",
        "Worst quality",
        "Highest flow",
        "Lowest SSIM",
        "Most failures",
    ],
)

# Review status filter
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Filters")
my_review_options = sorted(summary["my_review_status"].unique())
selected_my_reviews = st.sidebar.multiselect("My Review Status", my_review_options, default=my_review_options, help="Filter by your personal review decisions.")

global_review_options = sorted(summary["global_review_status"].unique())
selected_global_reviews = st.sidebar.multiselect("Global Review Status", global_review_options, default=global_review_options, help="Filter by the combined team decisions (including Conflicts).")

# Failed phase filter
PHASE_LABELS = {
    "geo": "Geometric Alignment (Flow)",
    "ssim": "Structural Similarity (SSIM)",
    "photo": "Photometric Alignment",
    "color": "Color Transfer",
    "interpolation": "Frame Interpolation",
}
all_phases = collect_failed_phases(logs["fail"])
if all_phases:
    selected_phases = st.sidebar.multiselect(
        "Show scenes that failed in:", 
        all_phases, 
        default=[],
        format_func=lambda x: PHASE_LABELS.get(x, str(x).title().replace("_", " ")),
        help="Filters the list to only show scenes that failed the selected pipeline validation checks."
    )


# ---- Scoring Criteria Explainer ----
st.sidebar.markdown("---")
with st.sidebar.expander("How Scoring Works", expanded=False):
    st.markdown("""
### Laplacian Score

Measures image sharpness using the **variance of the Laplacian** operator. The Laplacian highlights edges and fine detail — a sharp image produces high variance (lots of strong edges), while a blurry image produces low variance (edges are smoothed out).

**How frames are selected:** The pipeline computes the Laplacian score for every decoded frame in a capture sequence, producing a sharpness curve over time. During a motion event (e.g. handshake, vibration), the curve drops as the scene blurs.

- **Sharp frame (target)** — the frame with the highest Laplacian score, selected as the ground truth
- **Blur frame (input)** — the frame with the lowest Laplacian score during the motion event, representing natural motion blur
- **Drop frame** — the frame where sharpness first starts falling, marking the onset of motion

**Laplacian Delta** shows `Sharp − Blur`. A large positive delta means the pipeline found a clear distinction between sharp and blurry frames. A small or negative delta is a warning — the "blur" may not be blurry enough for meaningful deblurring training.

---

### Flow ROI p90

After geometric alignment (homography warp), this measures **residual misalignment** between the paired images. Optical flow is computed in the center region of interest, and the 90th-percentile magnitude is reported.

- **PASS**: ≤ 15.0 — the alignment successfully registered the pair
- **FAIL**: > 15.0 — significant residual motion remains; the homography could not fully correct the viewpoint difference (parallax, rotation, or extreme blur)

p90 is used instead of the mean because large static regions would dilute the average. The 90th percentile captures worst-case misalignment while ignoring the top 10% of outliers (moving objects, occlusion boundaries).

---

### Alignment Confidence (Inlier Ratio)

During geometric alignment, RANSAC fits a homography model to feature matches between the paired images. The **inlier ratio** is the fraction of feature matches that agreed with the final homography.

- **High ratio** (e.g. >50%) — the model is well-supported; many features consistently agree on the transformation
- **Low ratio** — few features agreed; the homography may be unreliable even if the flow metric looks acceptable

When alignment fails entirely, the system falls back to an identity transform (no warp), logged as `geo:IDENTITY_FALLBACK` in the fail log.

---

### SSIM (Structural Similarity)

Computed after color alignment between each pair of images. SSIM measures structural resemblance — edges, textures, luminance patterns — on a 0 to 1 scale.

- **PASS**: ≥ 0.70 — the pair is structurally consistent
- **FAIL**: < 0.70 — significant structural differences remain (parallax, occlusion, or failed color transfer)

The scorecard shows the **minimum SSIM** across all image pairs in the scene. If even one pair has poor similarity, the scene is flagged.

---

### Delta E (Color Difference)

CIELAB color difference between paired images after color alignment. Measured in perceptual units where **lower = more color-consistent**.

No hard pass/fail threshold — some color variation is expected between OIS and non-OIS camera modules. This metric is informational and contributes to the composite quality score.

---

### Quality Score (Composite)

A weighted combination for **ranking scenes** by overall quality *(higher = worse)*:

`0.5 × flow + 0.3 × (1 − ssim) × 100 + 0.2 × deltaE`

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Flow p90 | 50% | Geometric alignment is the most critical factor |
| (1−SSIM)×100 | 30% | Structural mismatch, scaled to ~0-100 range |
| Delta E | 20% | Color fidelity after alignment |

This is not a pass/fail — it is a **sorting tool**. Use "Sort by: Worst quality" to review the most problematic scenes first.

---

### Scene Status Rules

- **FAIL** — at least one image failed the `geo` or `ssim` phase during pipeline processing
- **NEEDS ATTENTION** — metrics exceed thresholds (flow > 15 or SSIM < 0.70) but no hard pipeline failure
- **CLEAN** — zero failures and all metrics within thresholds
    """)

# ---- Apply All Filters ----
filtered = summary.copy()
filtered = filtered[
    filtered["method"].isin(selected_methods)
]
filtered = filtered[
    filtered["my_review_status"].isin(selected_my_reviews) &
    filtered["global_review_status"].isin(selected_global_reviews)
]

# Apply phase filter (show scenes that have ANY of the selected phases)
if all_phases and selected_phases:
    phase_scenes = set()
    fail_df = logs["fail"]
    if not fail_df.empty and "failed_phases" in fail_df.columns:
        for _, row in fail_df.iterrows():
            phases_str = str(row.get("failed_phases", ""))
            scene_phases = {p.strip() for p in phases_str.split(";") if p.strip()}
            if scene_phases & set(selected_phases):
                phase_scenes.add(str(row["scene"]))
    filtered = filtered[filtered["scene"].isin(phase_scenes)]


# Apply text search
if scene_search:
    search_lower = scene_search.lower()
    filtered = filtered[filtered["scene"].str.lower().str.contains(search_lower, na=False)]

if sort_mode == "Scene order":
    filtered = filtered.sort_values(["scene_number", "scene"])
elif sort_mode == "Worst quality":
    filtered = filtered.sort_values(["quality_score", "scene"], ascending=[False, True], na_position="last")
elif sort_mode == "Highest flow":
    filtered = filtered.sort_values(["max_flow", "scene"], ascending=[False, True], na_position="last")
elif sort_mode == "Lowest SSIM":
    filtered = filtered.sort_values(["min_ssim", "scene"], ascending=[True, True], na_position="last")
elif sort_mode == "Most failures":
    filtered = filtered.sort_values(["failed_count", "scene"], ascending=[False, True], na_position="last")

filtered = filtered.reset_index(drop=True)

if filtered.empty:
    st.warning("No scenes match the current filters.")
    st.stop()

scene_labels = {
    row["scene"]: (f"{row['scene_name']} | {row['method']}" if row["split"] == "Unassigned" else f"{row['scene_name']} | {row['split']} | {row['method']}")
    for _, row in filtered.iterrows()
}

filtered_scenes = filtered["scene"].tolist()

if "nav_scene" not in st.session_state or st.session_state["nav_scene"] not in filtered_scenes:
    if filtered_scenes:
        st.session_state["nav_scene"] = filtered_scenes[0]

scene = selector_container.selectbox(
    "Select scene",
    filtered_scenes,
    format_func=lambda value: scene_labels.get(value, value),
    key="nav_scene"
)

def go_prev():
    cur_idx = filtered_scenes.index(st.session_state["nav_scene"]) if st.session_state["nav_scene"] in filtered_scenes else 0
    st.session_state["nav_scene"] = filtered_scenes[max(0, cur_idx - 1)]

def go_next():
    cur_idx = filtered_scenes.index(st.session_state["nav_scene"]) if st.session_state["nav_scene"] in filtered_scenes else 0
    st.session_state["nav_scene"] = filtered_scenes[min(len(filtered_scenes)-1, cur_idx + 1)]

nav_col1, nav_col2 = selector_container.columns(2)
nav_col1.button("< Previous", on_click=go_prev, use_container_width=True, help="Shortcut to jump to the previous scene in your filtered list")
nav_col2.button("Next >", on_click=go_next, use_container_width=True, help="Shortcut to jump to the next scene in your filtered list")

scene_row = filtered[filtered["scene"] == scene].iloc[0]
scene_position = filtered.index[filtered["scene"] == scene][0] + 1

selection_row = load_scene_row(logs["selection"], scene)
geo_scene = logs["geo"][logs["geo"]["scene"] == scene] if "scene" in logs["geo"].columns else pd.DataFrame()
photo_scene = logs["photo"][logs["photo"]["scene"] == scene] if "scene" in logs["photo"].columns else pd.DataFrame()
color_scene = logs["color"][logs["color"]["scene"] == scene] if "scene" in logs["color"].columns else pd.DataFrame()
fail_scene = logs["fail"][logs["fail"]["scene"] == scene] if "scene" in logs["fail"].columns else pd.DataFrame()
interp_scene = (
    logs["interpolation"][logs["interpolation"]["scene"] == scene]
    if "scene" in logs["interpolation"].columns
    else pd.DataFrame()
)
review_scene = (
    logs["review"][logs["review"]["scene"] == scene]
    if "scene" in logs["review"].columns
    else pd.DataFrame()
)

# Extract current review status if available
current_review_status = scene_row.get("review_status", "NOT REVIEWED")
if current_review_status == "NOT REVIEWED":
    current_review_status = None

capture = None if selection_row is None else selection_row.get("capture")
ois_df = load_laplacian(capture, "ois")
nonois_df = load_laplacian(capture, "nonois")
capture_label = "Missing" if capture is None or pd.isna(capture) else str(capture)

# Render Status Sidebar
status_container.header("Active Scene Status")
status_container.metric("Current Review", current_review_status if current_review_status else "Not Reviewed")


asset_status = {
    "Interpolated Images": bool(scene_row["has_dataset_target"]),
    "Aligned Images": bool(scene_row["has_aligned_color"]),
    "OIS Laplacian Scores": bool(scene_row["has_ois_laplacian"]),
    "Non-OIS Laplacian Scores": bool(scene_row["has_nonois_laplacian"]),
}

# --------------------------------------------------------------------------------------
# GLOBAL DYNAMIC COLOR CODED REVIEW NOTICES
# --------------------------------------------------------------------------------------
review_notice = st.session_state.pop("review_notice", None)
if review_notice:
    if review_notice["type"] == "keep":
        st.success(review_notice["msg"])
    elif review_notice["type"] == "reject":
        st.error(review_notice["msg"])
    elif review_notice["type"] == "flag":
        st.warning(review_notice["msg"])
    else:
        st.info(review_notice["msg"])

# --------------------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# --------------------------------------------------------------------------------------
main_tabs = st.tabs(["Scene Inspector", "Split Assignment & Export", "History & Management"])

# ======================================================================================
# TAB 1: SCENE INSPECTOR (Localized View)
# ======================================================================================
with main_tabs[0]:
    # Render main header with badges
    render_scene_header(scene_row, len(filtered), scene_position, current_review_status)

    # Prominent Global Action Panel for Reviews
    section_header("Scene Review Panel", level="h4", top_margin="0rem")

    base_predef_choices = [
        "3D issue (random background objects appearing during motion blur)",
        "Has light rays",
        "Scene is dark",
        "Has uneven lighting (light spots or patches of light)",
        "Has reflections",
        "Has ghost edge"
    ]
    custom_predef_choices = st.session_state.setdefault("review_note_custom_choices", [])
    predef_choices = base_predef_choices + [x for x in custom_predef_choices if x not in base_predef_choices]
    selected_notes = st.multiselect("Predefined Notes", predef_choices, key="review_note_predef", placeholder="Select encountered problems (optional)...")

    # Injecting an invisible marker to guarantee perfect flex-end alignment
    st.markdown('<div class="review-action-panel-marker" style="display:none;"></div>', unsafe_allow_html=True)
    def remember_custom_note(note: str):
        note = note.strip()
        if not note:
            return
        note_key = note.casefold()
        if note_key in {c.casefold() for c in base_predef_choices}:
            return
        if note_key not in {c.casefold() for c in st.session_state["review_note_custom_choices"]}:
            st.session_state["review_note_custom_choices"].append(note)

    rev_cols = st.columns([3, 1, 1, 1])
    custom_note = rev_cols[0].text_input(
        "Note",
        key="review_note",
        label_visibility="collapsed",
        placeholder="Type additional custom note...",
        on_change=lambda: remember_custom_note(st.session_state.get("review_note", "")),
    )

    final_note = " | ".join(selected_notes + ([custom_note.strip()] if custom_note.strip() else []))

    with rev_cols[1]:
        if st.button("KEEP", use_container_width=True):
            save_review(scene, "KEEP", final_note, reviewer_name)
            load_all_logs.clear()
            st.session_state["review_notice"] = {"msg": f"Saved KEEP for {scene_row['scene_name']}.", "type": "keep"}
            st.rerun()

    with rev_cols[2]:
        if st.button("REJECT", use_container_width=True):
            save_review(scene, "REJECT", final_note, reviewer_name)
            load_all_logs.clear()
            st.session_state["review_notice"] = {"msg": f"Saved REJECT for {scene_row['scene_name']}.", "type": "reject"}
            st.rerun()

    with rev_cols[3]:
        if st.button("FLAG", use_container_width=True):
            save_review(scene, "FLAG", final_note, reviewer_name)
            load_all_logs.clear()
            st.session_state["review_notice"] = {"msg": f"Saved FLAG for {scene_row['scene_name']}.", "type": "flag"}
            st.rerun()

    # JavaScript Injection for specific scene review panel buttons 
    components.html(
        """
        <script>
        const styleButtonsAndAlign = () => {
            const buttons = window.parent.document.querySelectorAll('button');
            buttons.forEach(btn => {
                const text = btn.innerText.trim();
                if (text === 'KEEP') {
                    btn.style.backgroundColor = 'rgba(34, 197, 94, 0.15)';
                    btn.style.borderColor = 'rgba(34, 197, 94, 0.5)';
                    btn.style.color = '#22c55e';
                } else if (text === 'REJECT') {
                    btn.style.backgroundColor = 'rgba(239, 68, 68, 0.15)';
                    btn.style.borderColor = 'rgba(239, 68, 68, 0.5)';
                    btn.style.color = '#ef4444';
                } else if (text === 'FLAG') {
                    btn.style.backgroundColor = 'rgba(245, 158, 11, 0.15)';
                    btn.style.borderColor = 'rgba(245, 158, 11, 0.5)';
                    btn.style.color = '#f59e0b';
                }
                
                if (text === 'KEEP' || text === 'REJECT' || text === 'FLAG') {
                    btn.onmouseover = function() {
                        if(text === 'KEEP') btn.style.backgroundColor = 'rgba(34, 197, 94, 0.25)';
                        if(text === 'REJECT') btn.style.backgroundColor = 'rgba(239, 68, 68, 0.25)';
                        if(text === 'FLAG') btn.style.backgroundColor = 'rgba(245, 158, 11, 0.25)';
                    }
                    btn.onmouseout = function() {
                        if(text === 'KEEP') btn.style.backgroundColor = 'rgba(34, 197, 94, 0.15)';
                        if(text === 'REJECT') btn.style.backgroundColor = 'rgba(239, 68, 68, 0.15)';
                        if(text === 'FLAG') btn.style.backgroundColor = 'rgba(245, 158, 11, 0.15)';
                    }
                }
            });
        };
        styleButtonsAndAlign();
        setInterval(styleButtonsAndAlign, 150);
        </script>
        """,
        height=0,
        width=0,
    )

    # ---- Quality Scorecard (Upgraded Visuals) ----
    section_header("Quality Scorecard", level="h3", top_margin="1rem")

    # Flow pass/fail
    flow_val = scene_row.get("max_flow")
    flow_pass = not above(flow_val, FLOW_THRESHOLD)
    flow_color = "#22c55e" if flow_pass else "#ef4444"
    flow_label = "PASS" if flow_pass else "FAIL"
    if flow_val is None or pd.isna(flow_val):
        flow_color = "var(--textColor)"
        flow_label = "N/A"

    # SSIM pass/fail
    ssim_val = scene_row.get("min_ssim")
    ssim_pass = not below(ssim_val, SSIM_THRESHOLD)
    ssim_color = "#22c55e" if ssim_pass else "#ef4444"
    ssim_label = "PASS" if ssim_pass else "FAIL"
    if ssim_val is None or pd.isna(ssim_val):
        ssim_color = "var(--textColor)"
        ssim_label = "N/A"

    # Delta E
    de_val = scene_row.get("mean_deltaE_after")

    # Quality score
    qs_val = scene_row.get("quality_score")
    
    scorecard_html = (
        f'<div class="metric-grid-4">'
        f'<div class="scorecard-card">'
        f'<div class="scorecard-title">Flow ROI p90</div>'
        f'<div class="scorecard-value">{format_num(flow_val)}</div>'
        f'<div class="scorecard-status" style="color: {flow_color};">{flow_label}</div>'
        f'</div>'
        f'<div class="scorecard-card">'
        f'<div class="scorecard-title">Min SSIM</div>'
        f'<div class="scorecard-value">{format_num(ssim_val, 3)}</div>'
        f'<div class="scorecard-status" style="color: {ssim_color};">{ssim_label}</div>'
        f'</div>'
        f'<div class="scorecard-card">'
        f'<div class="scorecard-title">Mean Delta E</div>'
        f'<div class="scorecard-value">{format_num(de_val)}</div>'
        f'<div class="scorecard-status" style="color: var(--textColor); opacity: 0.5;">LOWER IS BETTER</div>'
        f'</div>'
        f'<div class="scorecard-card">'
        f'<div class="scorecard-title">Quality Score</div>'
        f'<div class="scorecard-value">{format_num(qs_val, 1)}</div>'
        f'<div class="scorecard-status" style="color: var(--textColor); opacity: 0.5;">HIGHER IS WORSE</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(scorecard_html, unsafe_allow_html=True)

    # ---- Laplacian Deltas ----
    lap_deltas = compute_laplacian_deltas(selection_row, ois_df, nonois_df)
    delta_html_parts = []
    for cam_key, cam_label in [("ois", "OIS"), ("nonois", "Non-OIS")]:
        d = lap_deltas[cam_key]
        sharp_s = format_num(d["sharp"], 1) if d["sharp"] is not None else "-"
        blur_s = format_num(d["blur"], 1) if d["blur"] is not None else "-"
        delta_s = format_num(d["delta"], 1) if d["delta"] is not None else "-"
        delta_color = "#22c55e" if d["delta"] is not None and d["delta"] > 0 else "#ef4444" if d["delta"] is not None else "var(--textColor)"
        
        delta_html_parts.append(
            f'<div class="scorecard-card">'
            f'<div class="scorecard-title">{cam_label} Laplacian Delta</div>'
            f'<div style="font-size: 0.95rem; margin-top: 0.2rem; color: var(--textColor);">'
            f'Sharp: <b>{sharp_s}</b> &rarr; Blur: <b>{blur_s}</b> = <span style="color: {delta_color}; font-weight: 800;">&Delta;{delta_s}</span>'
            f'</div>'
            f'</div>'
        )
    st.markdown(f'<div class="metric-grid-2">{"".join(delta_html_parts)}</div>', unsafe_allow_html=True)

    # ---- Alignment Confidence ----
    geo_scene_data = logs["geo"][logs["geo"]["scene"] == scene] if "scene" in logs["geo"].columns else pd.DataFrame()
    if not geo_scene_data.empty and "inlier_ratio" in geo_scene_data.columns:
        avg_inlier = geo_scene_data["inlier_ratio"].mean()
        inlier_pct = f"{avg_inlier * 100:.1f}%" if not pd.isna(avg_inlier) else "-"
        inlier_color = "#22c55e" if avg_inlier and avg_inlier > 0.5 else "#f59e0b"
        st.markdown(
            f'<div class="scorecard-card" style="margin-bottom: 1rem;">'
            f'<div class="scorecard-title">Alignment Confidence</div>'
            f'<div style="font-size: 1.1rem; font-weight: 700; color: {inlier_color};">{inlier_pct} avg inlier ratio</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    scene_tabs = st.tabs(["Overview", "Laplacian", "Alignment + Interpolation", "Scene Logs & History", "Manual Frame Selection"])

    with scene_tabs[0]:
        render_asset_metrics(asset_status)
        
        capture_num = extract_capture_number(capture)

        info_html = (
            f'<div class="metric-grid-4">'
            f'<div class="scorecard-card">'
            f'<div class="scorecard-title">Capture #</div>'
            f'<div class="scorecard-value" style="font-size: 1.4rem;">{capture_num}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(info_html, unsafe_allow_html=True)
        
        selection_table = build_selection_table(selection_row)
        if not selection_table.empty:
            section_header("Selected Frames", level="h4")
            st.dataframe(selection_table, use_container_width=True, hide_index=True)

        render_scene_image_grid("Final Dataset", DATASET_TARGET, scene, selection_row)

    with scene_tabs[2]:
        render_scene_image_grid("Aligned Outputs", ALIGNED_COLOR, scene, selection_row)
        render_scene_image_grid("Interpolated Target Outputs", DATASET_TARGET, scene, selection_row)

    with scene_tabs[1]:
        selection_df = logs.get("selection", pd.DataFrame())

        captures = []
        for name in ("selection", "geo", "photo", "color", "interpolation"):
            df = logs.get(name, pd.DataFrame())
            if not df.empty and "scene" in df.columns and "capture" in df.columns:
                caps = df[df["scene"] == scene]["capture"].dropna().astype(str).unique().tolist()
                captures.extend(caps)

        if not captures and capture is not None and not pd.isna(capture):
            captures = [str(capture)]

        captures = sorted({str(c) for c in captures})

        if not captures:
            st.info("No laplacian captures found for this scene.")
        else:
            section_header("Capture Summary", level="h4", top_margin="0rem")

            summary_rows = []
            for cap in captures:
                sel_rows = (
                    selection_df[(selection_df.get("scene") == scene) & (selection_df.get("capture") == cap)]
                    if not selection_df.empty
                    else pd.DataFrame()
                )
                sel = sel_rows.iloc[0] if not sel_rows.empty else None
                ois = load_laplacian(cap, "ois")
                nonois = load_laplacian(cap, "nonois")

                summary_rows.append(
                    {
                        "Capture": extract_capture_number(cap),
                        "Ready": "Yes" if ois is not None and nonois is not None else "Partial",
                        "OIS Sharp Frame": frame_value(sel, "ois_sharp") if sel is not None else "",
                        "OIS Drop (Actual)": frame_value(sel, "ois_drop_frame_actual") if sel is not None else "",
                        "OIS Blur Frame": frame_value(sel, "ois_blur") if sel is not None else "",
                        "Non-OIS Sharp Frame": frame_value(sel, "nonois_sharp") if sel is not None else "",
                        "Non-OIS Drop (Actual)": frame_value(sel, "nonois_drop_frame_actual") if sel is not None else "",
                        "Non-OIS Drop (Fallback)": frame_value(sel, "nonois_drop_frame") if sel is not None else "",
                        "Non-OIS Blur Frame": frame_value(sel, "nonois_blur") if sel is not None else "",
                    }
                )

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            for cap in captures:
                cap_label = extract_capture_number(cap)
                with st.expander(f"Capture {cap_label}", expanded=(cap == str(capture))):
                    sel_rows = (
                        selection_df[(selection_df.get("scene") == scene) & (selection_df.get("capture") == cap)]
                        if not selection_df.empty
                        else pd.DataFrame()
                    )
                    sel = sel_rows.iloc[0] if not sel_rows.empty else None

                    ois = load_laplacian(cap, "ois")
                    nonois = load_laplacian(cap, "nonois")

                    cols = st.columns(1)
                    cols[0].metric("Selection Ready", "Yes" if ois is not None and nonois is not None else "Partial")

                    frame_tabs = st.tabs(["OIS Frames", "Non-OIS Frames"])
                    with frame_tabs[0]:
                        render_selected_frames(cap, "ois", ois, sel)
                    with frame_tabs[1]:
                        render_selected_frames(cap, "nonois", nonois, sel)

                    # Add clear vertical spacing between frames and graphs
                    st.markdown("<br><br>", unsafe_allow_html=True)

                    plot_cols = st.columns(2)
                    with plot_cols[0]:
                        plot_laplacian(
                            ois,
                            "OIS Laplacian Curve",
                            [
                                ("Sharp", frame_value(sel, "ois_sharp"), "#16a34a"),
                                ("Drop", frame_value(sel, "ois_drop_frame_actual"), "#d97706"),
                                ("Blur", frame_value(sel, "ois_blur"), "#dc2626"),
                            ],
                        )
                    with plot_cols[1]:
                        plot_laplacian(
                            nonois,
                            "Non-OIS Laplacian Curve",
                            [
                                ("Sharp", frame_value(sel, "nonois_sharp"), "#16a34a"),
                                ("Drop Actual", frame_value(sel, "nonois_drop_frame_actual"), "#ea580c"),
                                ("Drop Fallback", frame_value(sel, "nonois_drop_frame"), "#f97316"),
                                ("Blur", frame_value(sel, "nonois_blur"), "#dc2626"),
                            ],
                        )

                    section_header("Selected Frame Scores", level="h4")
                    st.dataframe(
                        build_laplacian_score_table(sel, ois, nonois),
                        use_container_width=True,
                        hide_index=True
                    )

    with scene_tabs[3]:
        log_tabs = st.tabs(["Geo", "Photo", "Color", "Interpolation", "Failures", "Review History"])

        with log_tabs[0]:
            render_dataframe(geo_scene, "No geo log rows for this scene.")
        with log_tabs[1]:
            render_dataframe(photo_scene, "No photo log rows for this scene.")
        with log_tabs[2]:
            render_dataframe(color_scene, "No color log rows for this scene.")
        with log_tabs[3]:
            render_dataframe(interp_scene, "No interpolation log rows for this scene.")
        with log_tabs[4]:
            if fail_scene.empty:
                st.info("No scene failures recorded for this scene.")
            else:
                render_dataframe(fail_scene, "")
                section_header("Parsed Failure Details", level="h4")
                parsed_rows = []
                for _, frow in fail_scene.iterrows():
                    details = parse_failed_images_detail(frow.get("failed_images"))
                    for d in details:
                        parsed_rows.append(d)
                if parsed_rows:
                    parsed_df = pd.DataFrame(parsed_rows)
                    badge_parts = []
                    for _, pr in parsed_df.iterrows():
                        phase = pr["phase"]
                        tone = "bad" if phase == "geo" else "warn" if phase == "ssim" else "muted"
                        detail_str = f' <span style="opacity:0.6;">({html.escape(pr["detail"])})</span>' if pr["detail"] else ""
                        badge_parts.append(
                            f'<tr>'
                            f'<td style="padding: 0.4rem 0.8rem;">{html.escape(pr["image"])}</td>'
                            f'<td style="padding: 0.4rem 0.8rem;">{tag_html(phase, tone)}</td>'
                            f'<td style="padding: 0.4rem 0.8rem;">{detail_str}</td>'
                            f'</tr>'
                        )
                    st.markdown(
                        '<table style="width:100%; border-collapse: collapse;">'
                        '<tr style="border-bottom: 1px solid rgba(128,128,128,0.2);">'
                        '<th style="text-align:left; padding: 0.4rem 0.8rem; font-size: 0.8rem; text-transform: uppercase; opacity: 0.6;">Image</th>'
                        '<th style="text-align:left; padding: 0.4rem 0.8rem; font-size: 0.8rem; text-transform: uppercase; opacity: 0.6;">Phase</th>'
                        '<th style="text-align:left; padding: 0.4rem 0.8rem; font-size: 0.8rem; text-transform: uppercase; opacity: 0.6;">Detail</th>'
                        '</tr>'
                        + "".join(badge_parts) +
                        '</table>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No parseable failure details.")
        
        with log_tabs[5]:
            section_header("Scene Review History", level="h4")
            
            if review_scene.empty:
                st.info("No manual review history for this scene yet.")
            else:
                st.markdown('<div style="margin-bottom:1rem;"></div>', unsafe_allow_html=True)
                
                hcols = st.columns([2, 5, 2, 2])
                hcols[0].markdown("<div style='color: var(--sb-muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Decision</div>", unsafe_allow_html=True)
                hcols[1].markdown("<div style='color: var(--sb-muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Note</div>", unsafe_allow_html=True)
                hcols[2].markdown("<div style='color: var(--sb-muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Reviewer</div>", unsafe_allow_html=True)
                hcols[3].markdown("<div style='color: var(--sb-muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Action</div>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 0.5rem 0 0.5rem 0; border-color: rgba(128,128,128,0.2);'>", unsafe_allow_html=True)
                
                for global_idx, row in review_scene.iterrows():
                    row_cols = st.columns([2, 5, 2, 2], vertical_alignment="center")
                    
                    with row_cols[0]:
                        st.markdown(f"<div style='font-weight: 700; font-size: 0.95rem;'>{row['label']}</div>", unsafe_allow_html=True)
                    
                    with row_cols[1]:
                        note_text = row.get("note", "")
                        if pd.notna(note_text) and note_text:
                            st.markdown(f"<div style='font-size: 0.95rem;'>{html.escape(str(note_text))}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='opacity: 0.5; font-style: italic; font-size: 0.95rem;'>No note provided</div>", unsafe_allow_html=True)
                            
                    with row_cols[2]:
                        rev_text = row.get("reviewer", "")
                        if pd.notna(rev_text) and rev_text:
                            st.markdown(f"<div style='font-size: 0.9rem; color: var(--textColor); opacity: 0.8;'>User: {html.escape(str(rev_text))}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='opacity: 0.5; font-style: italic; font-size: 0.85rem;'>-</div>", unsafe_allow_html=True)
                    
                    with row_cols[3]:
                        if st.button("Delete", key=f"del_row_{global_idx}", use_container_width=True):
                            delete_review_row(scene, global_idx)
                            load_all_logs.clear()
                            st.session_state["review_notice"] = {"msg": "Deleted a review record.", "type": "info"}
                            st.rerun()
                            
                    st.markdown("<hr style='margin: 0.2rem 0; border-color: rgba(128,128,128,0.1);'>", unsafe_allow_html=True)
                
                st.write("")
                if st.button("Clear Scene History", type="secondary"):
                    clear_scene_review(scene)
                    load_all_logs.clear()
                    st.session_state["review_notice"] = {"msg": f"Cleared all review history for {scene_row['scene_name']}.", "type": "info"}
                    st.rerun()

    with scene_tabs[4]:
        section_header("Manual Frame Selection", level="h3", top_margin="0rem")
        st.markdown(
            "Override the automatically selected frames for this scene. "
            "Pick replacement frames from the decoded capture using Laplacian scores as a guide, "
            "then optionally re-run alignment and interpolation for this scene only."
        )

        if capture is None or pd.isna(capture):
            st.warning("No capture associated with this scene — cannot browse frames.")
        else:
            # ── Current frame assignments ──────────────────────────────────────
            section_header("Current Frame Assignments", level="h4")
            FRAME_TYPES = [
                ("ois_sharp",    "OIS Sharp",    "ois"),
                ("ois_blur",     "OIS Blur",     "ois"),
                ("nonois_sharp", "Non-OIS Sharp","nonois"),
                ("nonois_blur",  "Non-OIS Blur", "nonois"),
            ]

            cur_frames = {
                ft: frame_value(selection_row, ft) for ft, _, _ in FRAME_TYPES
            }

            cur_cols = st.columns(4)
            for col, (ft, label, cam) in zip(cur_cols, FRAME_TYPES):
                with col:
                    st.markdown(f"**{label}**")
                    cur_f = cur_frames.get(ft)
                    if cur_f:
                        st.caption(cur_f)
                        img_path = resolve_scene_dir(DATASET_TARGET, scene) / f"{ft}.jpg"
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.markdown(
                                '<div style="aspect-ratio:1;display:flex;align-items:center;'
                                'justify-content:center;background:rgba(128,128,128,0.08);'
                                'border:1px dashed rgba(128,128,128,0.25);border-radius:6px;'
                                'font-size:0.75rem;opacity:0.5;">No interpolated output</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown('<div style="opacity:0.5;font-style:italic;font-size:0.85rem;">Not set</div>', unsafe_allow_html=True)

            st.markdown("<hr style='margin:1.5rem 0;border-color:rgba(128,128,128,0.15);'>", unsafe_allow_html=True)

            # ── Frame selection dropdowns ──────────────────────────────────────
            section_header("Select Replacement Frames", level="h4")
            st.markdown(
                "Each dropdown shows available `.dng` frames from the decoded capture with their Laplacian score. "
                "Only change the frames you want to override — leave the others on their current selection."
            )

            selected_overrides = {}
            for ft, label, cam in FRAME_TYPES:
                frames = list_capture_frames(capture, cam)
                if not frames:
                    st.warning(f"No decoded frames found for {cam} in capture `{capture}`.")
                    continue

                lap_df = load_laplacian(capture, cam)
                def lap_score(fname):
                    if lap_df is None:
                        return None
                    row_match = lap_df[lap_df["frame"] == fname]
                    if row_match.empty:
                        return None
                    return row_match["score"].iloc[0]

                def lap_direction(fname):
                    if lap_df is None or "direction" not in lap_df.columns:
                        return None
                    row_match = lap_df[lap_df["frame"] == fname]
                    if row_match.empty:
                        return None
                    val = row_match["direction"].iloc[0]
                    if pd.isna(val):
                        return None
                    return str(val)

                def fmt_option(fname, frames=frames, cur_frame=cur_f, method=scene_row.get("method")):
                    score = lap_score(fname)
                    score_str = f"  [Lap: {score:.1f}]" if score is not None else ""

                    dir_str = ""
                    try:
                        if method == "Sliding":
                            direction = lap_direction(fname)
                            if direction:
                                dir_str = f"  [{direction.upper()}]"
                    except Exception:
                        dir_str = ""

                    return f"{fname}{dir_str}{score_str}"

                cur_f = cur_frames.get(ft)
                default_idx = frames.index(cur_f) if cur_f in frames else 0

                with st.expander(f"{label}  —  current: `{cur_f or 'None'}`", expanded=False):
                    sel_cols = st.columns([3, 1])
                    chosen = sel_cols[0].selectbox(
                        f"New frame for {label}",
                        options=frames,
                        index=default_idx,
                        format_func=fmt_option,
                        key=f"frame_sel_{scene}_{ft}",
                        label_visibility="collapsed",
                    )
                    if chosen != cur_f:
                        selected_overrides[ft] = chosen

                    # Preview chosen frame
                    chosen_path = resolve_scene_dir(DECODED_FRAMES, capture) / cam / chosen
                    prev_img = load_display_image(str(chosen_path))
                    if prev_img is not None:
                        score = lap_score(chosen)
                        caption = f"Laplacian: {score:.1f}" if score is not None else "No score"
                        sel_cols[1].image(prev_img, caption=caption, use_container_width=True)
                    else:
                        sel_cols[1].caption("Preview unavailable")

            # ── Apply overrides ────────────────────────────────────────────────
            if selected_overrides:
                st.write("")
                st.info(f"**{len(selected_overrides)} frame(s) changed** — click Apply to commit.")
                if st.button("Apply Frame Overrides", type="primary", use_container_width=True):
                    for ft, new_frame in selected_overrides.items():
                        old_frame = cur_frames.get(ft)
                        # Copy DNG to dataset folder
                        cam = next(c for f, _, c in FRAME_TYPES if f == ft)
                        src_dng = resolve_scene_dir(DECODED_FRAMES, capture) / cam / new_frame
                        scene_dataset_dir = resolve_scene_dir(DATASET_ROOT, scene)
                        scene_dataset_dir.mkdir(parents=True, exist_ok=True)
                        dst_dng = scene_dataset_dir / f"{ft}.dng"
                        if src_dng.exists():
                            import shutil as _sh
                            _sh.copy2(str(src_dng), str(dst_dng))
                        # Update selection log
                        update_selection_log(scene, {ft: new_frame})
                        # Write override log entry
                        save_frame_override_log(scene, ft, old_frame or "", new_frame, reviewer_name)

                    load_all_logs.clear()
                    st.success(f"Applied {len(selected_overrides)} frame override(s). Re-run the pipeline below to regenerate outputs.")
                    st.rerun()
            else:
                st.write("")
                st.caption("Change at least one frame above to enable the Apply button.")

            st.markdown("<hr style='margin:1.5rem 0;border-color:rgba(128,128,128,0.15);'>", unsafe_allow_html=True)

            # ── Re-run pipeline ────────────────────────────────────────────────
            section_header("Re-run Pipeline for This Scene", level="h4")
            st.markdown(
                "After applying frame overrides, re-run alignment and/or interpolation "
                "to regenerate the aligned JPGs and 1080px outputs for this scene."
            )
            pipeline_cols = st.columns(3)
            run_stage = None
            with pipeline_cols[0]:
                if st.button("Re-run Alignment Only", use_container_width=True):
                    run_stage = "align"
            with pipeline_cols[1]:
                if st.button("Re-run Interpolation Only", use_container_width=True):
                    run_stage = "interp"
            with pipeline_cols[2]:
                if st.button("Re-run Alignment + Interpolation", use_container_width=True):
                    run_stage = "both"

            if run_stage:
                stage_label = {"align": "Alignment", "interp": "Interpolation", "both": "Alignment + Interpolation"}[run_stage]
                with st.spinner(f"Running {stage_label} for {scene_row['scene_name']}..."):
                    outputs, rcode = run_scene_pipeline(scene, run_stage)
                for stage_name, stdout, stderr in outputs:
                    with st.expander(f"{stage_name} Output", expanded=True):
                        if stdout.strip():
                            st.code(stdout, language="text")
                        if stderr.strip():
                            st.warning(stderr[:3000])
                if rcode == 0:
                    save_frame_override_log(scene, "", "", "", reviewer_name, action=f"rerun_{run_stage}")
                    load_all_logs.clear()
                    st.success(f"{stage_label} completed successfully.")
                    st.rerun()
                else:
                    st.error(f"{stage_label} exited with code {rcode}. Check the output above.")

            st.markdown("<hr style='margin:1.5rem 0;border-color:rgba(128,128,128,0.15);'>", unsafe_allow_html=True)

            # ── Override history ───────────────────────────────────────────────
            section_header("Frame Override History", level="h4")
            override_log_df = logs.get("frame_override", pd.DataFrame())
            if not override_log_df.empty and "scene" in override_log_df.columns:
                scene_overrides = override_log_df[override_log_df["scene"] == scene].sort_values("timestamp", ascending=False)
            else:
                scene_overrides = pd.DataFrame()

            if scene_overrides.empty:
                st.info("No frame overrides recorded for this scene yet.")
            else:
                st.dataframe(scene_overrides, use_container_width=True, hide_index=True)



    # ======================================================================================
    # TAB 2: SPLIT ASSIGNMENT & EXPORT
    # ======================================================================================
    with main_tabs[1]:
        sa_tabs = st.tabs(["Scene Set Builder", "Export Dataset"])

        # ------------------------------------------------------------------
        # SUB-TAB A: SCENE SET BUILDER
        # ------------------------------------------------------------------
        with sa_tabs[0]:
            section_header("Scene Set Builder", level="h3", top_margin="0rem")
            st.markdown(
                "Designate **scene sets** by curating collections of KEEP scenes. "
                "Use this interface to manage which scenes are included in specific dataset versions "
                "and define their split assignments for streamlined data organization."
            )

            all_keep_scenes = summary[summary["review_status"] == "KEEP"].copy()

            if all_keep_scenes.empty:
                st.info("No scenes marked as **KEEP** yet. Review scenes in the **Scene Inspector** first.")
            else:
                existing_sets = list_scene_sets()

                # ── Create / Load controls ─────────────────────────────────
                col_create, col_load = st.columns(2)
                with col_create:
                    with st.container(border=True):
                        st.markdown("**Create New Set**")
                        c1, c2 = st.columns([3, 1])
                        new_set_name = c1.text_input("New set name", placeholder="e.g. experiment_v1", key="new_set_name_input", label_visibility="collapsed")
                        if c2.button("Create", use_container_width=True, type="primary"):
                            name = new_set_name.strip()
                            if not name:
                                st.session_state["review_notice"] = {"msg": "Enter a name for the new set.", "type": "flag"}
                            elif name in existing_sets:
                                st.session_state["review_notice"] = {"msg": f"Set '{name}' already exists.", "type": "flag"}
                            else:
                                new_set = create_new_scene_set(name, all_keep_scenes, reviewer_name)
                                save_scene_set(name, new_set)
                                st.session_state["active_set"] = new_set
                                st.session_state["active_set_name"] = name
                                st.session_state["review_notice"] = {"msg": f"Created set '{name}' with {len(new_set['scenes'])} scenes.", "type": "keep"}
                            st.rerun()

                with col_load:
                    with st.container(border=True):
                        st.markdown("**Load Existing Set**")
                        l1, l2 = st.columns([3, 1])
                        load_options = ["— None —"] + existing_sets
                        selected_load = l1.selectbox("Load existing", load_options, key="load_set_sel", label_visibility="collapsed")
                        if l2.button("Load", use_container_width=True):
                            if selected_load != "— None —":
                                loaded = load_scene_set(selected_load)
                                if loaded:
                                    st.session_state["active_set"] = loaded
                                    st.session_state["active_set_name"] = selected_load
                                    st.session_state["review_notice"] = {"msg": f"Loaded set '{selected_load}'.", "type": "info"}
                                st.rerun()

                st.markdown("<hr style='margin:0.8rem 0;border-color:rgba(128,128,128,0.15);'>", unsafe_allow_html=True)

                # ── Active set editor ──────────────────────────────────────
                if "active_set" not in st.session_state:
                    st.info("Create a new set or load an existing one to begin editing.")
                else:
                    active_set = st.session_state["active_set"]
                    set_name = st.session_state.get("active_set_name", "Untitled")

                    # Sync new KEEP scenes into the set
                    keep_scene_ids = set(all_keep_scenes["scene"].astype(str))
                    for _, r in all_keep_scenes.iterrows():
                        if r["scene"] not in active_set["scenes"]:
                            active_set["scenes"][r["scene"]] = {"status": "Included", "split": "Unassigned"}
                            
                    # Clean up ghost scenes: if a scene is no longer KEEP globally, unassign it
                    changed = False
                    for sc in list(active_set["scenes"].keys()):
                        if str(sc) not in keep_scene_ids:
                            if active_set["scenes"][sc].get("status", "").lower() in ["include", "included"] or active_set["scenes"][sc].get("split", "Unassigned") != "Unassigned":
                                active_set["scenes"][sc]["status"] = "Unassigned"
                                active_set["scenes"][sc]["split"] = "Unassigned"
                                changed = True
                                
                    if changed:
                        save_scene_set(set_name, active_set)

                    scenes_data = active_set["scenes"]
                    inc_count = sum(1 for v in scenes_data.values() if v.get("status") == "Included")
                    exc_count = sum(1 for v in scenes_data.values() if v.get("status") == "Excluded")
                    flg_count = sum(1 for v in scenes_data.values() if v.get("status") == "Flagged")

                    # Stats banner
                    st.markdown(
                        f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1rem;'
                        f'padding:1rem 1.4rem;background:var(--secondaryBackgroundColor);'
                        f'border-radius:10px;border:1px solid rgba(128,128,128,0.15);">'
                        f'<div style="font-size:1.1rem;font-weight:700;">Set: {html.escape(set_name)}</div>'
                        f'<div><span style="font-size:1.4rem;font-weight:800;color:#22c55e;">{inc_count}</span>'
                        f' <span style="opacity:0.6;font-size:0.75rem;text-transform:uppercase;">Included</span></div>'
                        f'<div><span style="font-size:1.4rem;font-weight:800;color:#ef4444;">{exc_count}</span>'
                        f' <span style="opacity:0.6;font-size:0.75rem;text-transform:uppercase;">Excluded</span></div>'
                        f'<div><span style="font-size:1.4rem;font-weight:800;color:#f59e0b;">{flg_count}</span>'
                        f' <span style="opacity:0.6;font-size:0.75rem;text-transform:uppercase;">Flagged</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Internal sub-tabs: Scene Status | Split Assignment | Scene Groups ──────
                    ssb_tabs = st.tabs(["Scene Status", "Split Assignment", "Split Preview", "Scene Groups"])

                    # ══════════════════════════════════════════════════════
                    # INTERNAL TAB 1: SCENE STATUS (Include / Exclude / Flag)
                    # ══════════════════════════════════════════════════════
                    with ssb_tabs[0]:
                        # ── Bulk action bar ────────────────────────────────────
                        with st.container(border=True):
                            st.markdown("<div style='font-weight:600;margin-bottom:0.8rem;'>Bulk Actions for Selected Scenes</div>", unsafe_allow_html=True)
                            b_cols = st.columns(2)

                            with b_cols[0]:
                                st.caption("Selection")
                                sel_1, sel_2 = st.columns(2)
                                if sel_1.button("Select All", use_container_width=True, key="act_sel_all"):
                                    for sc in scenes_data:
                                        st.session_state[f"setsel_{set_name}_{sc}"] = True
                                    st.rerun()
                                if sel_2.button("Deselect All", use_container_width=True, key="act_desel_all"):
                                    for sc in scenes_data:
                                        st.session_state[f"setsel_{set_name}_{sc}"] = False
                                    st.rerun()

                            with b_cols[1]:
                                st.caption("Set Status")
                                stat_1, stat_2 = st.columns([2, 1])
                                bulk_action = stat_1.selectbox("Action", ["Include", "Exclude", "Flag"], key="set_bulk_action", label_visibility="collapsed")
                                if stat_2.button("Apply", key="btn_apply_status", use_container_width=True):
                                    selected = [sc for sc in scenes_data if st.session_state.get(f"setsel_{set_name}_{sc}")]
                                    if selected:
                                        action_map = {"Include": "Included", "Exclude": "Excluded", "Flag": "Flagged"}
                                        new_status_val = action_map[bulk_action]
                                        for sc in selected:
                                            scenes_data[sc]["status"] = new_status_val
                                        save_scene_set(set_name, active_set)
                                        st.session_state["review_notice"] = {"msg": f"Set {len(selected)} scenes to {new_status_val}.", "type": "info"}
                                        for sc in selected:
                                            st.session_state[f"setsel_{set_name}_{sc}"] = False
                                        st.rerun()
                                    else:
                                        st.warning("Select scenes first.")

                        # ── Scene list grouped by method ─────────────────────────
                        keep_method_map = {}
                        for _, r in all_keep_scenes.iterrows():
                            m = r["method"]
                            if m not in keep_method_map:
                                keep_method_map[m] = []
                            keep_method_map[m].append(r)

                        for method in sorted(keep_method_map.keys()):
                            method_scenes = keep_method_map[method]
                            method_inc = sum(1 for r in method_scenes if scenes_data.get(r["scene"], {}).get("status") == "Included")
                            method_label = f"{method}  ·  {method_inc}/{len(method_scenes)} included"

                            with st.expander(method_label, expanded=True):
                                for r in sorted(method_scenes, key=lambda x: x["scene_number"]):
                                    sc = r["scene"]
                                    sc_name = r["scene_name"]
                                    sc_data = scenes_data.get(sc, {"status": "Included", "split": "Unassigned"})
                                    status = sc_data["status"]
                                    if status.lower() == "include": status = "Included"
                                    elif status.lower() == "exclude": status = "Excluded"
                                    elif status.lower() == "flag": status = "Flagged"
                                    status_badge_cls = {"Included": "badge-keep", "Excluded": "badge-reject", "Flagged": "badge-flag"}.get(status, "badge-muted")

                                    st.markdown(
                                        f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">'
                                        f'<div style="font-size:1.05rem;font-weight:700;">{html.escape(sc_name)}</div>'
                                        f'<span class="dashboard-badge {status_badge_cls}">{status.upper()}</span>'
                                        f'<div style="font-size:0.8rem;opacity:0.5;">{html.escape(sc)}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                                    scene_dir_target = resolve_scene_dir(DATASET_TARGET, sc)
                                    img_specs = [
                                        ("OIS Blur", "ois_blur.jpg"),
                                        ("OIS Sharp", "ois_sharp.jpg"),
                                        ("Non-OIS Blur", "nonois_blur.jpg"),
                                        ("Non-OIS Sharp", "nonois_sharp.jpg"),
                                    ]
                                    img_cols = st.columns([1, 4, 4, 4, 4])
                                    with img_cols[0]:
                                        st.checkbox("sel", key=f"setsel_{set_name}_{sc}", label_visibility="collapsed")

                                    any_image = False
                                    for img_col, (img_label, img_file) in zip(img_cols[1:], img_specs):
                                        img_path = scene_dir_target / img_file
                                        with img_col:
                                            if img_path.exists():
                                                st.image(str(img_path), caption=img_label, use_container_width=True)
                                                any_image = True
                                            else:
                                                st.markdown(
                                                    f'<div style="aspect-ratio:1;display:flex;align-items:center;'
                                                    f'justify-content:center;background:rgba(128,128,128,0.08);'
                                                    f'border:1px dashed rgba(128,128,128,0.25);border-radius:6px;'
                                                    f'font-size:0.75rem;opacity:0.5;text-align:center;padding:0.5rem;">'
                                                    f'{img_label}<br>Missing</div>',
                                                    unsafe_allow_html=True,
                                                )

                                    if not any_image:
                                        st.caption("No output images found — the pipeline may not have run yet for this scene.")

                                    st.markdown("<hr style='margin:1rem 0 0.6rem 0;border-color:rgba(128,128,128,0.1);'></hr>", unsafe_allow_html=True)

                    # ══════════════════════════════════════════════════════
                    # INTERNAL TAB 2: SPLIT ASSIGNMENT
                    # ══════════════════════════════════════════════════════
                    with ssb_tabs[1]:
                        section_header("Split Assignment", level="h4", top_margin="0rem")
                        st.markdown(
                            "Assign scenes to **Training / Validation / Testing** splits with stratified quotas. "
                            "Select a target split, check scenes to assign, and track progress against Figure 9 targets."
                        )

                        # ── Quota Dashboard ─────────────────────────────────
                        render_quota_dashboard(scenes_data)

                        # ── Distribution Chart ──────────────────────────────
                        with st.expander("Distribution Chart", expanded=False):
                            try:
                                import altair as alt
                                counts = compute_quota_status(scenes_data)
                                chart_rows = []
                                for split in ["Training", "Validation", "Testing"]:
                                    for method in ["Handshake", "Vibration", "Sliding"]:
                                        chart_rows.append({"Split": split, "Method": method, "Count": counts.get((split, method), 0), "Type": "Current"})
                                        chart_rows.append({"Split": split, "Method": method, "Count": STRATIFIED_QUOTAS.get(split, {}).get(method, 0), "Type": "Target"})
                                cdf = pd.DataFrame(chart_rows)
                                chart = alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                                    x=alt.X("Type:N", title=None, axis=alt.Axis(labelAngle=0)),
                                    y=alt.Y("Count:Q", title="Scenes"),
                                    color=alt.Color("Type:N", scale=alt.Scale(domain=["Current", "Target"], range=["#3b82f6", "rgba(128,128,128,0.3)"])),
                                    column=alt.Column("Split:N", title=None),
                                    row=alt.Row("Method:N", title=None),
                                    tooltip=["Split:N", "Method:N", "Type:N", "Count:Q"],
                                ).properties(width=120, height=80)
                                st.altair_chart(chart)
                            except Exception:
                                st.info("Install altair for distribution charts.")

                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Target Split Selector ───────────────────────────
                        sp_target_cols = st.columns([2, 1, 1])
                        with sp_target_cols[0]:
                            target_split = st.radio(
                                "Target Split",
                                ["Training", "Validation", "Testing", "Unassigned"],
                                horizontal=True,
                                key="sp_target_split",
                            )
                        with sp_target_cols[1]:
                            show_assigned = st.checkbox("Show already-assigned", value=False, key="sp_show_assigned")
                        with sp_target_cols[2]:
                            st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

                        groups = active_set.get("groups", {})
                        # Build reverse group map for badge display
                        scene_to_group = {}
                        for gid, members in groups.items():
                            for member in members:
                                scene_to_group[member] = gid

                        # ── Quick-fill Buttons ──────────────────────────────
                        with st.container(border=True):
                            st.markdown("<div style='font-weight:600;margin-bottom:0.5rem;'>Quick Actions</div>", unsafe_allow_html=True)
                            qf_cols = st.columns(5)
                            with qf_cols[0]:
                                if st.button("Auto-fill to Quota", use_container_width=True, key="sp_autofill"):
                                    # Collect unassigned atomic units
                                    unassigned_units = []
                                    seen = set()
                                    for sc, d in scenes_data.items():
                                        if sc in seen: continue
                                        if str(d.get("status", "")).lower() not in ["include", "included"]: continue
                                        if d.get("split", "Unassigned") != "Unassigned": continue
                                        gid = scene_to_group.get(sc)
                                        if gid:
                                            unit = [m for m in groups[gid] if m in scenes_data and scenes_data[m].get("split", "Unassigned") == "Unassigned" and str(scenes_data[m].get("status", "")).lower() in ["include", "included"]]
                                            for m in unit: seen.add(m)
                                            if unit: unassigned_units.append(unit)
                                        else:
                                            seen.add(sc)
                                            unassigned_units.append([sc])
                                    
                                    import random
                                    random.shuffle(unassigned_units)
                                    unassigned_units.sort(key=len, reverse=True) # Place larger groups first
                                    counts = compute_quota_status(scenes_data)
                                    filled = 0
                                    
                                    for unit in unassigned_units:
                                        # Strictly ensure we don't overshoot any method target
                                        can_add = True
                                        for sc in unit:
                                            m = parse_scene(sc)["method"]
                                            target = STRATIFIED_QUOTAS.get(target_split, {}).get(m, 0)
                                            current = counts.get((target_split, m), 0)
                                            if current + 1 > target:
                                                can_add = False
                                                break
                                                
                                        if can_add:
                                            for sc in unit:
                                                scenes_data[sc]["split"] = target_split
                                                m = parse_scene(sc)["method"]
                                                counts[(target_split, m)] = counts.get((target_split, m), 0) + 1
                                                filled += 1
                                                
                                    if filled:
                                        save_scene_set(set_name, active_set)
                                        st.session_state["review_notice"] = {"msg": f"Auto-filled {filled} scene(s) into {target_split}.", "type": "keep"}
                                    else:
                                        st.session_state["review_notice"] = {"msg": f"{target_split} quota already met.", "type": "info"}
                                    st.rerun()
                            with qf_cols[1]:
                                if st.button("Shuffle & Fill All", use_container_width=True, key="sp_shuffle"):
                                    assignments = shuffle_fill_all(scenes_data, groups)
                                    if assignments:
                                        for sc, sp in assignments.items():
                                            scenes_data[sc]["split"] = sp
                                        save_scene_set(set_name, active_set)
                                        st.session_state["review_notice"] = {"msg": f"Shuffled {len(assignments)} scene(s) across all splits.", "type": "keep"}
                                    else:
                                        st.session_state["review_notice"] = {"msg": "No unassigned scenes to distribute.", "type": "info"}
                                    st.rerun()
                            with qf_cols[2]:
                                if st.button("Check All Visible", use_container_width=True, key="sp_check_all"):
                                    for sc in scenes_data:
                                        st.session_state[f"sp_sel_{set_name}_{sc}"] = True
                                    st.rerun()
                            with qf_cols[3]:
                                if st.button("Uncheck All", use_container_width=True, key="sp_uncheck_all"):
                                    for sc in scenes_data:
                                        st.session_state[f"sp_sel_{set_name}_{sc}"] = False
                                    st.rerun()
                            with qf_cols[4]:
                                if st.button("Unassign All", use_container_width=True, key="sp_unassign_all"):
                                    count = 0
                                    for sc, d in scenes_data.items():
                                        if d.get("split", "Unassigned") != "Unassigned":
                                            d["split"] = "Unassigned"
                                            count += 1
                                    if count > 0:
                                        save_scene_set(set_name, active_set)
                                        st.session_state["review_notice"] = {"msg": f"Unassigned {count} scene(s).", "type": "info"}
                                    else:
                                        st.session_state["review_notice"] = {"msg": "All scenes are already unassigned.", "type": "info"}
                                    st.rerun()

                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Build method-grouped scene lists ────────────────
                        METHOD_ORDER = ["Handshake", "Sliding", "Vibration"]
                        sp_method_map = {m: [] for m in METHOD_ORDER}
                        for _, r in all_keep_scenes.iterrows():
                            m = r["method"]
                            if m in sp_method_map:
                                sp_method_map[m].append(r)
                        for m in METHOD_ORDER:
                            sp_method_map[m] = sorted(sp_method_map[m], key=lambda x: x["scene_number"])

                        METHOD_COLORS = {
                            "Handshake": "#3b82f6",
                            "Sliding": "#8b5cf6",
                            "Vibration": "#f59e0b",
                        }

                        # ── 3-column method headers with progress ───────────
                        counts = compute_quota_status(scenes_data)
                        hdr_cols = st.columns(3)
                        for hdr_col, method in zip(hdr_cols, METHOD_ORDER):
                            current = counts.get((target_split, method), 0)
                            target = STRATIFIED_QUOTAS.get(target_split, {}).get(method, 0)
                            pct = min(100, int(current / target * 100)) if target > 0 else 0
                            color = METHOD_COLORS.get(method, "var(--textColor)")
                            fill_color = "#22c55e" if current >= target else color
                            hdr_col.markdown(
                                f'<div style="text-align:center;padding:0.6rem 0.8rem;'
                                f'background:var(--secondaryBackgroundColor);border-radius:10px;'
                                f'border:2px solid {color}33;margin-bottom:0.8rem;">'
                                f'<div style="font-size:1rem;font-weight:800;color:{color};">{method}</div>'
                                f'<div style="font-size:1.3rem;font-weight:800;color:{fill_color};margin:0.2rem 0;">{current}/{target}</div>'
                                f'<div style="background:rgba(128,128,128,0.15);border-radius:999px;height:5px;overflow:hidden;">'
                                f'<div style="background:{fill_color};height:100%;width:{pct}%;border-radius:999px;"></div></div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        # ── Row-by-row scene cards ──────────────────────────
                        sp_filtered_map = {}
                        for method in METHOD_ORDER:
                            scenes_list = []
                            for r in sp_method_map[method]:
                                sc = r["scene"]
                                sc_data = scenes_data.get(sc, {"status": "Included", "split": "Unassigned"})
                                status = str(sc_data.get("status", "")).lower()
                                if status not in ["include", "included"]:
                                    continue
                                split_val = sc_data.get("split", "Unassigned")
                                if not show_assigned and split_val != "Unassigned":
                                    continue
                                scenes_list.append(r)
                            sp_filtered_map[method] = scenes_list

                        max_scenes = max((len(v) for v in sp_filtered_map.values()), default=0)

                        for row_i in range(max_scenes):
                            row_cols = st.columns(3)
                            for col_widget, method in zip(row_cols, METHOD_ORDER):
                                scenes_in_method = sp_filtered_map[method]
                                if row_i >= len(scenes_in_method):
                                    with col_widget:
                                        st.markdown("<div style='min-height:50px;'></div>", unsafe_allow_html=True)
                                    continue

                                r = scenes_in_method[row_i]
                                sc = r["scene"]
                                sc_name = r["scene_name"]
                                sc_data = scenes_data.get(sc, {"status": "Included", "split": "Unassigned"})
                                split_val = sc_data.get("split", "Unassigned")
                                split_badge_cls = "badge-info" if split_val != "Unassigned" else "badge-muted"
                                group_id = scene_to_group.get(sc)

                                with col_widget:
                                    with st.container(border=True):
                                        chk_col, info_col = st.columns([1, 5])
                                        with chk_col:
                                            st.checkbox("sel", key=f"sp_sel_{set_name}_{sc}", label_visibility="collapsed")
                                        with info_col:
                                            badges = f'<span class="dashboard-badge {split_badge_cls}" style="font-size:0.65rem;">{split_val}</span>'
                                            if group_id:
                                                badges += f' <span class="dashboard-badge badge-warn" style="font-size:0.6rem;">[Group] {group_id}</span>'
                                            st.markdown(
                                                f'<div style="font-size:0.9rem;font-weight:700;line-height:1.2;">{html.escape(sc_name)}</div>'
                                                f'<div style="margin-top:0.2rem;">{badges}</div>',
                                                unsafe_allow_html=True,
                                            )

                                        scene_dir_target = resolve_scene_dir(DATASET_TARGET, sc)
                                        ois_sharp_path = scene_dir_target / "ois_sharp.jpg"
                                        if ois_sharp_path.exists():
                                            st.image(str(ois_sharp_path), caption="OIS Sharp", use_container_width=True)
                                        else:
                                            st.markdown(
                                                '<div style="aspect-ratio:4/3;display:flex;align-items:center;'
                                                'justify-content:center;background:rgba(128,128,128,0.08);'
                                                'border:1px dashed rgba(128,128,128,0.25);border-radius:6px;'
                                                'font-size:0.75rem;opacity:0.5;">Missing</div>',
                                                unsafe_allow_html=True,
                                            )

                            st.markdown("<hr style='margin:0.3rem 0;border-color:rgba(128,128,128,0.06);'></hr>", unsafe_allow_html=True)

                        # ── Assign Button ───────────────────────────────────
                        sp_selected = [sc for sc in scenes_data if st.session_state.get(f"sp_sel_{set_name}_{sc}")]
                        if sp_selected:
                            group_warnings = []
                            for sc in sp_selected:
                                gid = scene_to_group.get(sc)
                                if gid:
                                    for member in groups[gid]:
                                        if member not in sp_selected:
                                            existing_split = scenes_data.get(member, {}).get("split", "Unassigned")
                                            if existing_split not in ["Unassigned", target_split]:
                                                group_warnings.append(f"{parse_scene(member)['scene_name']} ({gid}) is in {existing_split}")

                            if group_warnings:
                                st.warning(f"[Warning] Group conflict: " + "; ".join(group_warnings[:5]))

                            st.info(f"**{len(sp_selected)} scene(s) selected** — assign to **{target_split}**")
                            def do_assign():
                                for sc in sp_selected:
                                    scenes_data[sc]["split"] = target_split
                                    gid = scene_to_group.get(sc)
                                    if gid:
                                        for member in groups[gid]:
                                            if member in scenes_data:
                                                scenes_data[member]["split"] = target_split
                                save_scene_set(set_name, active_set)
                                for sc in sp_selected:
                                    if f"sp_sel_{set_name}_{sc}" in st.session_state:
                                        st.session_state[f"sp_sel_{set_name}_{sc}"] = False
                                st.session_state["review_notice"] = {"msg": f"Assigned {len(sp_selected)} scene(s) to {target_split}.", "type": "keep"}

                            st.button(f"Assign {len(sp_selected)} Scene(s) to {target_split}", type="primary", use_container_width=True, key="sp_assign_btn", on_click=do_assign)
                        else:
                            st.caption("Check scenes above then click Assign, or use Auto-fill / Shuffle & Fill.")

                        # ── Pre-export Validation ───────────────────────────
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("Pre-export Validation Checklist", expanded=False):
                            if st.button("Run Validation", use_container_width=True, key="sp_validate"):
                                checks = validate_pre_export(scenes_data, groups, DATASET_TARGET)
                                for chk in checks:
                                    icon = "[OK]" if chk["status"] == "pass" else "[Warning]" if chk["status"] == "warn" else "[Fail]"
                                    color = "#22c55e" if chk["status"] == "pass" else "#f59e0b" if chk["status"] == "warn" else "#ef4444"
                                    st.markdown(
                                        f'<div style="padding:0.5rem 0.8rem;margin-bottom:0.4rem;border-radius:8px;'
                                        f'border:1px solid {color}33;background:{color}11;">'
                                        f'{icon} <b>{chk["check"]}</b><br>'
                                        f'<span style="font-size:0.85rem;opacity:0.7;">{chk["detail"]}</span></div>',
                                        unsafe_allow_html=True,
                                    )

                    # ══════════════════════════════════════════════════════
                    # INTERNAL TAB 3: SPLIT PREVIEW
                    # ══════════════════════════════════════════════════════
                    with ssb_tabs[2]:
                        section_header("Split Preview", level="h4", top_margin="0rem")
                        st.markdown("Preview the assigned scenes grouped by their visual similarity to verify no identical scenes cross splits.")
                        
                        preview_split = st.selectbox("Select Split to Preview", ["Training", "Validation", "Testing", "Unassigned"], key="prev_split")
                        
                        prev_scenes = []
                        for sc, d in scenes_data.items():
                            if str(d.get("status", "")).lower() not in ["include", "included"]: continue
                            if d.get("split", "Unassigned") == preview_split:
                                prev_scenes.append(sc)
                                
                        if not prev_scenes:
                            st.info(f"No included scenes currently assigned to {preview_split}.")
                        else:
                            st.caption(f"Showing {len(prev_scenes)} scenes in {preview_split}. Identical scenes (same group) are placed side-by-side.")
                            
                            scene_to_group = {}
                            groups = active_set.get("groups", {})
                            for gid, members in groups.items():
                                for m in members: scene_to_group[m] = gid
                                
                            def sort_key(s):
                                return (scene_to_group.get(s, f"Z_{s}"), s)
                            prev_scenes.sort(key=sort_key)
                            
                            p_cols = st.columns(6)
                            for idx, sc in enumerate(prev_scenes):
                                with p_cols[idx % 6]:
                                    scene_dir = resolve_scene_dir(DATASET_TARGET, sc)
                                    img_path = scene_dir / "ois_sharp.jpg"
                                    
                                    if img_path.exists():
                                        img_arr = load_display_image(str(img_path))
                                        st.image(img_arr, use_container_width=True)
                                    else:
                                        st.markdown(f"<div style='aspect-ratio:1;background:rgba(128,128,128,0.1);border-radius:6px;display:flex;align-items:center;justify-content:center;color:gray;font-size:0.7rem;margin-bottom:0.2rem;'>No Image</div>", unsafe_allow_html=True)
                                        
                                    gid = scene_to_group.get(sc)
                                    badge = f'<span style="background:rgba(245,158,11,0.2);color:#f59e0b;padding:0.1rem 0.3rem;border-radius:4px;font-size:0.6rem;font-weight:700;">[Group] {gid}</span>' if gid else ""
                                    st.markdown(f"<div style='text-align:center;font-size:0.75rem;margin-bottom:1rem;'>{parse_scene(sc)['scene_name']}<br>{badge}</div>", unsafe_allow_html=True)


                    # ══════════════════════════════════════════════════════
                    # INTERNAL TAB 4: SCENE GROUPS
                    # ══════════════════════════════════════════════════════
                    with ssb_tabs[3]:
                        section_header("Scene Groups (Duplicate Detection)", level="h4", top_margin="0rem")
                        st.markdown(
                            "Detect visually similar scenes using **perceptual hashing** to prevent data leakage. "
                            "Grouped scenes are automatically assigned to the **same split** during assignment."
                        )

                        groups = active_set.get("groups", {})

                        # ── Scan Controls ───────────────────────────────────
                        with st.container(border=True):
                            scan_cols = st.columns([2, 1, 1])
                            with scan_cols[0]:
                                threshold = st.slider("Similarity Threshold (lower = stricter)", 6, 20, 14, key="phash_threshold",
                                                      help="Hamming distance threshold. 6=near-exact, 14=loose, 20=very loose")
                            with scan_cols[1]:
                                st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
                                scan_btn = st.button("Scan for Similar Scenes", type="primary", use_container_width=True, key="scan_phash")
                            with scan_cols[2]:
                                st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
                                clear_groups_btn = st.button("Clear All Groups", use_container_width=True, key="clear_groups")

                        if clear_groups_btn:
                            active_set["groups"] = {}
                            save_scene_set(set_name, active_set)
                            st.session_state.pop("detected_pairs", None)
                            st.session_state["review_notice"] = {"msg": "Cleared all scene groups.", "type": "info"}
                            st.rerun()

                        if scan_btn:
                            included_scenes = tuple(
                                sc for sc, d in scenes_data.items()
                                if str(d.get("status", "")).lower() in ["include", "included"]
                            )
                            with st.spinner(f"Scanning {len(included_scenes)} scenes..."):
                                hashes = compute_all_phashes(included_scenes, str(DATASET_TARGET))
                                pairs = find_similar_pairs(hashes, threshold=threshold)
                                detected = cluster_pairs_into_groups(pairs)
                            st.session_state["detected_pairs"] = pairs
                            st.session_state["detected_groups"] = detected
                            if detected:
                                st.session_state["review_notice"] = {"msg": f"Found {len(detected)} group(s) with {len(pairs)} similar pair(s).", "type": "info"}
                            else:
                                st.session_state["review_notice"] = {"msg": "No similar scenes detected at this threshold.", "type": "info"}
                            st.rerun()

                        # ── Show Detected Pairs ─────────────────────────────
                        detected_groups = st.session_state.get("detected_groups", {})
                        if detected_groups:
                            section_header("Detected Similar Groups", level="h4")
                            st.markdown(f"**{len(detected_groups)} group(s)** detected. Review and confirm to protect against data leakage.")

                            for gid, members in detected_groups.items():
                                with st.expander(f"{gid}: {len(members)} scenes", expanded=True):
                                    thumb_cols = st.columns(min(len(members), 5))
                                    for tc, sc in zip(thumb_cols, members[:5]):
                                        with tc:
                                            img_path = resolve_scene_dir(DATASET_TARGET, sc) / "ois_sharp.jpg"
                                            if img_path.exists():
                                                st.image(str(img_path), caption=parse_scene(sc)["scene_name"], use_container_width=True)
                                            else:
                                                st.caption(parse_scene(sc)["scene_name"])
                                    if len(members) > 5:
                                        st.caption(f"... and {len(members) - 5} more")

                            if st.button("Confirm All Detected Groups", type="primary", use_container_width=True, key="confirm_groups"):
                                existing = active_set.get("groups", {})
                                next_idx = len(existing) + 1
                                for _, members in detected_groups.items():
                                    gid = f"group_{next_idx:03d}"
                                    existing[gid] = members
                                    next_idx += 1
                                active_set["groups"] = existing
                                save_scene_set(set_name, active_set)
                                st.session_state.pop("detected_groups", None)
                                st.session_state.pop("detected_pairs", None)
                                st.session_state["review_notice"] = {"msg": f"Confirmed {len(detected_groups)} group(s).", "type": "keep"}
                                st.rerun()

                        # ── Manual Group Management ──────────────────────────────
                        with st.expander("Manual Group Management", expanded=False):
                            st.markdown("Manually create a new group or add scenes to an existing group. Useful for forcing scenes to stay together if auto-detection missed them.")
                            all_included = [sc for sc, d in scenes_data.items() if str(d.get("status", "")).lower() in ["include", "included"]]
                            
                            man_scenes = st.multiselect("Select Scenes", all_included, format_func=lambda x: f"{parse_scene(x)['scene_name']} ({x.split('_', 1)[0]})", key="man_grp_sc")
                            grp_options = ["Create New Group"] + list(groups.keys())
                            man_tgt = st.selectbox("Target Group", grp_options, key="man_grp_tgt")
                            
                            if st.button("Apply Manual Grouping", type="primary", disabled=not man_scenes):
                                # Remove from existing groups first to prevent duplicates
                                for sc in man_scenes:
                                    for g, m in list(groups.items()):
                                        if sc in m:
                                            m.remove(sc)
                                            if not m: del active_set["groups"][g]
                                            
                                if man_tgt == "Create New Group":
                                    next_idx = 1
                                    while f"group_{next_idx:03d}" in active_set.get("groups", {}):
                                        next_idx += 1
                                    gid = f"group_{next_idx:03d}"
                                    active_set.setdefault("groups", {})[gid] = man_scenes
                                else:
                                    active_set["groups"][man_tgt].extend(man_scenes)
                                    # Deduplicate
                                    active_set["groups"][man_tgt] = list(set(active_set["groups"][man_tgt]))
                                
                                save_scene_set(set_name, active_set)
                                st.session_state["review_notice"] = {"msg": f"Manually updated {man_tgt if man_tgt != 'Create New Group' else gid}.", "type": "keep"}
                                st.rerun()

                        # ── Show Confirmed Groups ───────────────────────────
                        if groups:
                            section_header("Confirmed Groups", level="h4")
                            for gid, members in groups.items():
                                with st.expander(f"{gid}: {len(members)} scenes"):
                                    num_cols = 8
                                    for i in range(0, len(members), num_cols):
                                        cols = st.columns(num_cols)
                                        for j, sc in enumerate(members[i:i+num_cols]):
                                            with cols[j]:
                                                img_path = resolve_scene_dir(DATASET_TARGET, sc) / "ois_sharp.jpg"
                                                sc_split = scenes_data.get(sc, {}).get("split", "Unassigned")
                                                split_short = sc_split[:3].upper() if sc_split != "Unassigned" else "---"
                                                sc_method = parse_scene(sc).get("method", "Unknown")
                                                m_short = sc_method[:3].upper() if sc_method != "Unknown" else "---"
                                                label = f"{parse_scene(sc)['scene_name']} [{m_short} | {split_short}]"
                                                if img_path.exists():
                                                    st.image(str(img_path), caption=label, use_container_width=True)
                                                else:
                                                    st.caption(label)
                                        
                                    rc1, rc2, rc3 = st.columns([2, 2, 1])
                                    with rc1:
                                        all_included = [s for s, d in scenes_data.items() if str(d.get("status", "")).lower() in ["include", "included"] and s not in members]
                                        to_add = st.multiselect("Add scenes", options=all_included, format_func=lambda x: f"{parse_scene(x)['scene_name']} ({x.split('_', 1)[0]})", key=f"add_sc_{gid}")
                                        if st.button("Add Selected", key=f"btn_add_sc_{gid}", disabled=not to_add):
                                            active_set["groups"][gid].extend(to_add)
                                            active_set["groups"][gid] = list(set(active_set["groups"][gid]))
                                            # Also remove these scenes from any other group they might be in
                                            for g, m in list(active_set["groups"].items()):
                                                if g == gid: continue
                                                for a in to_add:
                                                    if a in m: m.remove(a)
                                                if not m: del active_set["groups"][g]
                                            save_scene_set(set_name, active_set)
                                            st.session_state["review_notice"] = {"msg": f"Added {len(to_add)} scene(s) to {gid}.", "type": "keep"}
                                            st.rerun()
                                    with rc2:
                                        to_remove = st.multiselect("Remove scenes", options=members, format_func=lambda x: parse_scene(x)['scene_name'], key=f"rm_sc_{gid}")
                                        if st.button("Remove Selected", key=f"btn_rm_sc_{gid}", disabled=not to_remove):
                                            new_members = [m for m in members if m not in to_remove]
                                            if new_members:
                                                active_set["groups"][gid] = new_members
                                            else:
                                                del active_set["groups"][gid]
                                            save_scene_set(set_name, active_set)
                                            st.session_state["review_notice"] = {"msg": f"Removed {len(to_remove)} scene(s) from {gid}.", "type": "info"}
                                            st.rerun()
                                    with rc3:
                                        st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
                                        if st.button("Delete Group", key=f"rm_grp_{gid}", type="secondary"):
                                            del active_set["groups"][gid]
                                            save_scene_set(set_name, active_set)
                                            st.session_state["review_notice"] = {"msg": f"Removed {gid}.", "type": "info"}
                                            st.rerun()

                            violations = validate_split_groups(scenes_data, groups)
                            if violations:
                                st.error(f"[Warning] {len(violations)} group(s) have members in different splits!")
                                for v in violations:
                                    st.warning(f"**{v['group']}**: split across {', '.join(v['splits'])}")
                            else:
                                st.success("[OK] All groups are intact — no cross-split leaks.")
                        elif not detected_groups:
                            st.info("No groups defined yet. Click **Scan for Similar Scenes** to auto-detect.")

                    # ── Set management footer (shared, outside internal tabs) ────────

        # ------------------------------------------------------------------
        # SUB-TAB B: EXPORT DATASET
        # ------------------------------------------------------------------
        with sa_tabs[1]:
            section_header("Export Dataset", level="h3", top_margin="0rem")
            st.markdown(
                "Export a **scene set** into the final folder structure for model training. "
                "Each export creates a versioned dataset folder under `workspace/data/final_dataset/`."
            )

            available_sets = list_scene_sets()
            if not available_sets:
                st.info("No scene sets found. Create one in the **Scene Set Builder** tab first.")
            else:
                exp_cols_top = st.columns([2, 2, 2])
                with exp_cols_top[0]:
                    export_set_name = st.selectbox("Scene Set", available_sets, key="export_set_select")
                with exp_cols_top[1]:
                    existing_ds = list_final_datasets()
                    next_num = len(existing_ds) + 1
                    default_ds_name = f"dataset_{next_num}"
                    dataset_name = st.text_input("Dataset folder name", value=default_ds_name, key="export_ds_name")
                with exp_cols_top[2]:
                    st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)

                # Load and preview the set
                export_set = load_scene_set(export_set_name)
                if export_set is None:
                    st.error("Could not load the selected scene set.")
                else:
                    # Cross-reference with global review status: only keep scenes globally approved
                    global_keep_scenes = set(summary[summary["review_status"] == "KEEP"]["scene"])
                    export_set["scenes"] = {
                        k: v for k, v in export_set["scenes"].items() 
                        if k in global_keep_scenes
                    }

                    inc_scenes = {k: v for k, v in export_set["scenes"].items() if str(v.get("status", "")).lower() in ["include", "included"]}
                    assigned = {k: v for k, v in inc_scenes.items() if v.get("split") in SPLIT_EXPORT_MAP}
                    unassigned_inc = len(inc_scenes) - len(assigned)

                    # Stats banner
                    split_counts = {}
                    for v in assigned.values():
                        sp = v["split"]
                        split_counts[sp] = split_counts.get(sp, 0) + 1

                    banner_parts = []
                    for sp in ["Training", "Validation", "Testing"]:
                        cnt = split_counts.get(sp, 0)
                        if cnt > 0:
                            color = {"Training": "#3b82f6", "Validation": "#8b5cf6", "Testing": "#f59e0b"}.get(sp, "var(--textColor)")
                            banner_parts.append(
                                f'<div><span style="font-size:1.4rem;font-weight:800;color:{color};">{cnt}</span>'
                                f' <span style="opacity:0.6;font-size:0.75rem;text-transform:uppercase;">{sp}</span></div>'
                            )
                    if unassigned_inc > 0:
                        banner_parts.append(
                            f'<div><span style="font-size:1.4rem;font-weight:800;color:#ef4444;">{unassigned_inc}</span>'
                            f' <span style="opacity:0.6;font-size:0.75rem;text-transform:uppercase;">Unassigned (skipped)</span></div>'
                        )

                    st.markdown(
                        f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.5rem;'
                        f'padding:1rem 1.4rem;background:var(--secondaryBackgroundColor);'
                        f'border-radius:10px;border:1px solid rgba(128,128,128,0.15);">'
                        f'<div><span style="font-size:1.4rem;font-weight:800;color:#22c55e;">{len(assigned)}</span>'
                        f' <span style="opacity:0.6;font-size:0.75rem;text-transform:uppercase;">Exportable</span></div>'
                        + "".join(banner_parts) + '</div>',
                        unsafe_allow_html=True,
                    )

                    if assigned:
                        # Breakdown by Method
                        method_counts = {}
                        for sc in assigned.keys():
                            m = parse_scene(sc)["method"]
                            method_counts[m] = method_counts.get(m, 0) + 1
                        
                        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; opacity: 0.6; text-transform: uppercase;'>Dataset Composition by Method</div>", unsafe_allow_html=True)
                        
                        method_cards = []
                        for m, cnt in sorted(method_counts.items()):
                            method_cards.append(
                                f'<div class="scorecard-card">'
                                f'<div class="scorecard-title" style="margin-bottom: 0.2rem;">{m}</div>'
                                f'<div class="scorecard-value" style="font-size: 1.6rem;">{cnt} <span style="font-size: 0.8rem; font-weight: 600; opacity: 0.5; text-transform: uppercase;">Scenes</span></div>'
                                f'</div>'
                            )
                        
                        st.markdown(
                            f'<div class="metric-grid-4" style="margin-bottom: 1.5rem;">{"".join(method_cards)}</div>',
                            unsafe_allow_html=True
                        )

                    if not assigned:
                        st.warning("No included scenes have a split assigned. Go to **Scene Set Builder** and assign splits.")
                    else:
                        # Output structure preview
                        section_header("Output Folder Structure", level="h4")
                        ds_n = html.escape(dataset_name.strip() or default_ds_name)
                        st.code(
                            f"""final_dataset/{ds_n}/
  ois/
    train/
      input/   <- ois_blur.jpg  (001.jpg, 002.jpg, ...)
      target/  <- ois_sharp.jpg (001.jpg, 002.jpg, ...)
    val/  ...  test/
  nonois/
    train/
      input/   <- nonois_blur.jpg
      target/  <- nonois_sharp.jpg
    val/  ...  test/""",
                            language="text",
                        )
                        st.caption(f"Export location: `{(FINAL_DATASET_ROOT / (dataset_name.strip() or default_ds_name)).resolve()}`")

                        # Existing datasets
                        if existing_ds:
                            section_header("Existing Datasets", level="h4")
                            for ds in existing_ds:
                                ds_path = FINAL_DATASET_ROOT / ds
                                manifest_path = ds_path / "manifest.csv"
                                file_count = sum(1 for _ in ds_path.rglob("*.jpg"))
                                detail = f"{file_count} images"
                                if manifest_path.exists():
                                    try:
                                        mdf = pd.read_csv(manifest_path)
                                        detail = f"{len(mdf)} entries, {file_count} images"
                                    except Exception:
                                        pass
                                ds_cols = st.columns([3, 2, 1])
                                ds_cols[0].markdown(f"**{ds}**")
                                ds_cols[1].caption(detail)
                                with ds_cols[2]:
                                    if st.button("Delete", key=f"del_ds_{ds}", use_container_width=True):
                                        shutil.rmtree(str(ds_path), ignore_errors=True)
                                        st.session_state["review_notice"] = {"msg": f"Deleted dataset '{ds}'.", "type": "reject"}
                                        st.rerun()

                        # Check for overwrite
                        final_name = dataset_name.strip() or default_ds_name
                        if (FINAL_DATASET_ROOT / final_name).exists():
                            st.warning(f"A dataset named **{final_name}** already exists. Exporting will overwrite it.")

                        st.write("")
                        if st.button("Export Dataset", use_container_width=True, type="primary"):
                            with st.spinner("Exporting dataset..."):
                                try:
                                    manifest_df = export_dataset_v2(export_set, DATASET_TARGET, final_name)
                                    if manifest_df.empty:
                                        st.warning("No files were exported. Check that scenes have assigned splits.")
                                    else:
                                        st.success(
                                            f"Exported {len(manifest_df)} files to "
                                            f"`final_dataset/{final_name}/` successfully!"
                                        )
                                except Exception as e:
                                    st.error(f"Export failed: {e}")


# ======================================================================================
# TAB 3: HISTORY & MANAGEMENT
# ======================================================================================
with main_tabs[2]:
    hm_tabs = st.tabs(["Review Analytics", "Review Log", "Frame Overrides", "Auto-Actions", "Audit Trail"])

    # ------------------------------------------------------------------
    # REVIEW ANALYTICS
    # ------------------------------------------------------------------
    with hm_tabs[0]:
        section_header("Reviewer Productivity Analytics", level="h3", top_margin="0rem")
        st.markdown(
            "Leaderboard showing the latest unique review decision per scene, per reviewer. "
            "Multiple reviews on the same scene by the same reviewer only count once (last wins)."
        )

        full_review_df = logs.get("review", pd.DataFrame())

        if full_review_df.empty:
            st.info("No global manual review history found.")
        else:
            clean_df = full_review_df.copy()
            if "reviewer" not in clean_df.columns:
                clean_df["reviewer"] = "Unknown"
            else:
                clean_df["reviewer"] = clean_df["reviewer"].fillna("Unknown").replace("", "Unknown")

            uniq_reviews = clean_df.drop_duplicates(subset=["scene", "reviewer"], keep="last")

            stats_data = []
            for rev, group in uniq_reviews.groupby("reviewer"):
                counts = group["label"].str.upper().value_counts()
                stats_data.append(
                    {
                        "Reviewer": "User: " + str(rev),
                        "Total Scenes": len(group),
                        "KEEP": counts.get("KEEP", 0),
                        "REJECT": counts.get("REJECT", 0),
                        "FLAG": counts.get("FLAG", 0),
                    }
                )

            stats_df = pd.DataFrame(stats_data).sort_values("Total Scenes", ascending=False)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # REVIEW EVENT LOG
    # ------------------------------------------------------------------
    with hm_tabs[1]:
        section_header("Raw Review Event Log", level="h3", top_margin="0rem")
        st.markdown(
            "Chronological, un-deduplicated log of every review action as it was recorded. "
            "Each row is a single click of KEEP / REJECT / FLAG."
        )

        full_review_df = logs.get("review", pd.DataFrame())

        if full_review_df.empty:
            st.info("No history to display.")
        else:
            st.dataframe(full_review_df, use_container_width=True, hide_index=True)
            st.write("")
            st.warning("Warning: This permanently deletes ALL review decisions across the entire workspace.")
            if st.button("Clear ALL Global Review History", type="primary"):
                clear_all_global_reviews()
                load_all_logs.clear()
                st.session_state["review_notice"] = {
                    "msg": "Completely cleared all global review history.",
                    "type": "info",
                }
                st.rerun()

    # ------------------------------------------------------------------
    # FRAME OVERRIDES LOG
    # ------------------------------------------------------------------
    with hm_tabs[2]:
        section_header("Global Frame Overrides History", level="h3", top_margin="0rem")
        st.markdown(
            "Every frame replacement and pipeline re-run recorded across all scenes. "
            "Newest entries appear at the top."
        )
        fo_df = logs.get("frame_override", pd.DataFrame())
        if fo_df.empty:
            st.info("No frame overrides have been recorded yet.")
        else:
            st.dataframe(
                fo_df.sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True
            )
            if st.button("Clear Frame Override Log", type="secondary"):
                if FRAME_OVERRIDE_LOG.exists():
                    FRAME_OVERRIDE_LOG.unlink()
                    load_all_logs.clear()
                    st.success("Cleared the frame override log.")
                    st.rerun()

    # ------------------------------------------------------------------
    # AUTO-ACTIONS
    # ------------------------------------------------------------------
    with hm_tabs[3]:
        section_header("Auto-Reject Failed Alignments", level="h3", top_margin="0rem")
        st.markdown(
            "Automatically marks scenes as **REJECT** if they appear in the scene failure log "
            "(i.e., they failed the geometric or SSIM alignment checks). "
            "Only applies to scenes not already set to REJECT."
        )
        if st.button("Auto-Reject All Alignment Failures", type="primary"):
            rejected_count = 0

            review_df = logs.get("review", pd.DataFrame())
            my_reviews = {}
            if not review_df.empty and "scene" in review_df.columns:
                rev_col = (
                    review_df.get("reviewer", pd.Series(dtype=str)).fillna("Unknown").replace("", "Unknown")
                )
                my_latest = review_df[rev_col == reviewer_name].drop_duplicates(
                    subset=["scene"], keep="last"
                )
                if "label" in my_latest.columns:
                    my_reviews = dict(zip(my_latest["scene"], my_latest["label"].str.upper()))

            for _, row in summary.iterrows():
                my_status = my_reviews.get(str(row["scene"]), "NOT REVIEWED")
                if row.get("failed_count", 0) > 0 and my_status != "REJECT":
                    save_review(
                        row["scene"],
                        "REJECT",
                        "Auto-rejected due to pipeline alignment failure",
                        reviewer_name,
                    )
                    rejected_count += 1

            if rejected_count > 0:
                load_all_logs.clear()
                st.success(f"Successfully auto-rejected {rejected_count} failed alignment scenes!")
                st.rerun()
            else:
                st.info(
                    "No unreviewed failed alignment scenes found. "
                    "They either don't exist or have already been reviewed."
                )

    # ------------------------------------------------------------------
    # AUDIT TRAIL — split assignments only
    # ------------------------------------------------------------------
    with hm_tabs[4]:
        section_header("Split Assignment History", level="h3", top_margin="0rem")
        st.markdown(
            "A chronological record of every manual split assignment made from the **Split Assignment** tab. "
            "These entries are what the dashboard reads to place scenes into Training, Validation, or Testing — "
            "the underlying pipeline logs are never modified."
        )

        overrides_df = load_overrides()
        split_overrides = pd.DataFrame()
        if not overrides_df.empty:
            overrides_df_sorted = overrides_df.sort_values("timestamp", ascending=False)
            split_overrides = overrides_df_sorted[overrides_df_sorted["override_type"] == "split"]

        if split_overrides.empty:
            st.info("No split assignments have been saved yet. Go to **Split Assignment & Export** to assign scenes.")
        else:
            st.dataframe(split_overrides, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset All Split Assignments", type="secondary"):
                df = load_overrides()
                df = df[df["override_type"] != "split"]
                if df.empty:
                    if OVERRIDES_CSV.exists():
                        OVERRIDES_CSV.unlink()
                else:
                    df.to_csv(OVERRIDES_CSV, index=False)
                load_all_logs.clear()
                st.success("All split assignments have been reset. Scenes are now Unassigned.")
                st.rerun()
