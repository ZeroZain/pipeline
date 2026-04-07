import html
import re
from pathlib import Path
import shutil
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

DATASET_256 = ROOT / "data" / "dataset_256" / "gt_ois"
ALIGNED_COLOR = ROOT / "data" / "aligned" / "gt_ois" / "color"
DECODED_FRAMES = ROOT / "data" / "decoded_frames"

EXPORT_ROOT = ROOT / "data"
SPLIT_OPTIONS = ["Unassigned", "Training", "Validation", "Testing"]
SPLIT_EXPORT_MAP = {
    "Training": "train",
    "Validation": "val",
    "Testing": "test",
}

LOG_DIR = ROOT / "logs"
LAPLACIAN_DIR = LOG_DIR / "laplacian"
OVERRIDES_CSV = LOG_DIR / "scene_overrides.csv"

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
        "method": parts[0] if len(parts) > 0 else "Unknown",
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
    }


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
    summary["has_dataset_256"] = summary["scene"].map(lambda s: has_named_images(DATASET_256, s))
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
        summary["has_dataset_256"]
        & summary["has_aligned_color"]
        & summary["has_ois_laplacian"]
        & summary["has_nonois_laplacian"]
    )
    return summary


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
    badges = [
        tag_html(scene_row["split"], "info"),
        tag_html(scene_row["method"], "muted"),
        tag_html(f"Scene {scene_row['scene_name']}", "muted"),
        tag_html(f"{filtered_index}/{filtered_count} in filter", "muted"),
    ]

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
    if as_bool(scene_row.get("nonois_used_fallback")):
        badges.append(tag_html("Fallback Used", "warn"))
    if as_bool(scene_row.get("is_linear_scene")) is True:
        badges.append(tag_html("Linear Motion", "info"))
    if as_bool(scene_row.get("is_linear_scene")) is False:
        badges.append(tag_html("Non-Linear Motion", "ok"))

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

split_options = sorted(summary["split"].dropna().unique())
method_options = sorted(summary["method"].dropna().unique())

st.sidebar.header("Filters")
selected_splits = st.sidebar.multiselect("Split", split_options, default=split_options)
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
    filtered["split"].isin(selected_splits) & filtered["method"].isin(selected_methods)
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
    row["scene"]: f"{row['scene_name']} | {row['split']} | {row['method']}"
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
nav_col1.button("◀ Previous", on_click=go_prev, use_container_width=True, help="Shortcut to jump to the previous scene in your filtered list")
nav_col2.button("Next ▶", on_click=go_next, use_container_width=True, help="Shortcut to jump to the next scene in your filtered list")

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
    "Interpolated Images": bool(scene_row["has_dataset_256"]),
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
main_tabs = st.tabs(["Scene Inspector", "Data Management & History", "Assigned Splits Preview", "Export Dataset"])

# ======================================================================================
# TAB 1: SCENE INSPECTOR (Localized View)
# ======================================================================================
with main_tabs[0]:
    # Render main header with badges
    render_scene_header(scene_row, len(filtered), scene_position, current_review_status)

    # Prominent Global Action Panel for Reviews
    section_header("Scene Review Panel", level="h4", top_margin="0rem")

    # Injecting an invisible marker to guarantee perfect flex-end alignment
    st.markdown('<div class="review-action-panel-marker" style="display:none;"></div>', unsafe_allow_html=True)
    rev_cols = st.columns([3, 1, 1, 1])
    review_note = rev_cols[0].text_input("Note", key="review_note", label_visibility="collapsed", placeholder="Optional Review Note...")

    with rev_cols[1]:
        if st.button("KEEP", use_container_width=True):
            save_review(scene, "KEEP", review_note, reviewer_name)
            load_all_logs.clear()
            st.session_state["review_notice"] = {"msg": f"Saved KEEP for {scene_row['scene_name']}.", "type": "keep"}
            st.rerun()

    with rev_cols[2]:
        if st.button("REJECT", use_container_width=True):
            save_review(scene, "REJECT", review_note, reviewer_name)
            load_all_logs.clear()
            st.session_state["review_notice"] = {"msg": f"Saved REJECT for {scene_row['scene_name']}.", "type": "reject"}
            st.rerun()

    with rev_cols[3]:
        if st.button("FLAG", use_container_width=True):
            save_review(scene, "FLAG", review_note, reviewer_name)
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

    scene_tabs = st.tabs(["Overview", "Laplacian", "Alignment + Interpolation", "Scene Logs & History"])

    with scene_tabs[0]:
        render_asset_metrics(asset_status)
        
        capture_num = extract_capture_number(capture)
        fallback_used = bool_label(scene_row.get("nonois_used_fallback"))
        linear_scene = bool_label(scene_row.get("is_linear_scene"))

        info_html = (
            f'<div class="metric-grid-4">'
            f'<div class="scorecard-card">'
            f'<div class="scorecard-title">Capture #</div>'
            f'<div class="scorecard-value" style="font-size: 1.4rem;">{capture_num}</div>'
            f'</div>'
            f'<div class="scorecard-card">'
            f'<div class="scorecard-title">Fallback Used</div>'
            f'<div class="scorecard-value" style="font-size: 1.4rem;">{fallback_used}</div>'
            f'</div>'
            f'<div class="scorecard-card">'
            f'<div class="scorecard-title">Linear Scene</div>'
            f'<div class="scorecard-value" style="font-size: 1.4rem;">{linear_scene}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(info_html, unsafe_allow_html=True)
        
        # ---- Scene Data Management (Overrides) ----
        section_header("Manage Scene Data", level="h4")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            current_split = scene_row["split"]
            split_idx = SPLIT_OPTIONS.index(current_split) if current_split in SPLIT_OPTIONS else 0
            new_split = st.selectbox(
                "Assign to Split", 
                options=SPLIT_OPTIONS, 
                index=split_idx,
                key=f"split_move_{scene}"
            )
            if st.button("Apply Split Change", key=f"btn_split_{scene}"):
                if new_split != scene_row["split"]:
                    save_override(scene, "split", scene_row["split"], new_split, reviewer_name)
                    load_all_logs.clear()
                    st.session_state["review_notice"] = {"msg": f"Assigned {scene_row['scene_name']} to {new_split}!", "type": "info"}
                    st.rerun()

        with m_col2:
            method_opts = ["HandShake Method", "Sliding Method", "Vibration Method"]
            
            # Gracefully map existing raw methods to the clean UI options
            raw_method = str(scene_row["method"])
            curr_mapped = raw_method
            if "handshake" in raw_method.lower():
                curr_mapped = "HandShake Method"
            elif "sliding" in raw_method.lower():
                curr_mapped = "Sliding Method"
            elif "vibration" in raw_method.lower():
                curr_mapped = "Vibration Method"
                
            new_method = st.selectbox(
                "Change Method", 
                options=method_opts, 
                index=method_opts.index(curr_mapped) if curr_mapped in method_opts else 0,
                key=f"method_move_{scene}"
            )
            if st.button("Apply Method Change", key=f"btn_method_{scene}"):
                if new_method != curr_mapped:
                    save_override(scene, "method", raw_method, new_method, reviewer_name)
                    load_all_logs.clear()
                    st.session_state["review_notice"] = {"msg": f"Changed {scene_row['scene_name']} method to {new_method}!", "type": "info"}
                    st.rerun()

        selection_table = build_selection_table(selection_row)
        if not selection_table.empty:
            section_header("Selected Frames", level="h4")
            st.dataframe(selection_table, use_container_width=True, hide_index=True)

        render_scene_image_grid("Final Dataset", DATASET_256, scene, selection_row)

    with scene_tabs[2]:
        render_scene_image_grid("Aligned Outputs", ALIGNED_COLOR, scene, selection_row)
        render_scene_image_grid("Interpolated 256 Outputs", DATASET_256, scene, selection_row)

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
                        "Fallback": bool_label(sel.get("nonois_used_fallback")) if sel is not None else "-",
                        "Linear": bool_label(sel.get("is_linear_scene")) if sel is not None else "-",
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

                    cols = st.columns(3)
                    cols[0].metric("Fallback Used", bool_label(sel.get("nonois_used_fallback")) if sel is not None else "-")
                    cols[1].metric("Linear Scene", bool_label(sel.get("is_linear_scene")) if sel is not None else "-")
                    cols[2].metric("Selection Ready", "Yes" if ois is not None and nonois is not None else "Partial")

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
                            st.markdown(f"<div style='font-size: 0.9rem; color: var(--textColor); opacity: 0.8;'>👤 {html.escape(str(rev_text))}</div>", unsafe_allow_html=True)
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

# ======================================================================================
# TAB 2: DATA MANAGEMENT & HISTORY
# ======================================================================================
with main_tabs[1]:
    section_header("Reviewer Productivity Analytics", level="h3", top_margin="0rem")
    st.markdown("A leaderboard tracking the latest exact review decisions strictly scoped per unique reviewer.")

    full_review_df = logs.get("review", pd.DataFrame())
    
    if full_review_df.empty:
        st.info("No global manual review history found.")
    else:
        # Produce deduplicated analytics table
        clean_df = full_review_df.copy()
        if "reviewer" not in clean_df.columns:
            clean_df["reviewer"] = "Unknown"
        else:
            clean_df["reviewer"] = clean_df["reviewer"].fillna("Unknown").replace("", "Unknown")
        
        # We drop duplicates so a reviewer hammering 'KEEP' multiple times on a single scene only counts once
        uniq_reviews = clean_df.drop_duplicates(subset=["scene", "reviewer"], keep="last")
        
        stats_data = []
        for rev, group in uniq_reviews.groupby("reviewer"):
            counts = group["label"].str.upper().value_counts()
            stats_data.append({
                "Reviewer": "👤 " + str(rev),
                "Total Scenes": len(group),
                "KEEP": counts.get("KEEP", 0),
                "REJECT": counts.get("REJECT", 0),
                "FLAG": counts.get("FLAG", 0),
            })
            
        stats_df = pd.DataFrame(stats_data).sort_values("Total Scenes", ascending=False)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
    section_header("Raw Review Event Log", level="h3")
    st.markdown("Chronological un-deduplicated log of every review action exactly as it was recorded.")

    if full_review_df.empty:
        st.info("No history to display.")
    else:
        # Display the global dataframe
        st.dataframe(full_review_df, use_container_width=True, hide_index=True)
        
        st.write("")
        st.warning("Warning: This action deletes all review decisions across the entire workspace.")
        if st.button("Clear ALL Global Review History", type="primary"):
            clear_all_global_reviews()
            load_all_logs.clear()
            st.session_state["review_notice"] = {"msg": "Completely cleared all global review history.", "type": "info"}
            st.rerun()

    section_header("Scene Overrides & Data Management", level="h3")
    st.markdown("Use this section to view or automatically fix scene categorizations (splits & methods) without touching the original workspace files.")

    section_header("Auto-Fix Methods Based on Linear Motion", level="h4")
    st.markdown(
        "Scans all scenes. If a `HandShake Method` scene HAS linear motion, it will be logically moved to `Sliding Method`. "
        "If a `Sliding Method` scene has NO linear motion, it will be logically moved to `HandShake Method`."
    )
    if st.button("Auto-Fix Misclassified Methods", type="primary"):
        fixes_applied = 0
        for _, row in summary.iterrows():
            method = str(row["method"]).lower()
            is_linear = as_bool(row.get("is_linear_scene"))
            scene_id = row["scene"]
            
            # Using 'in' instead of '==' to catch variations like "handshake_1" or "sliding_test"
            if is_linear is True and "handshake" in method:
                save_override(scene_id, "method", row["method"], "Sliding Method", reviewer_name)
                fixes_applied += 1
            elif is_linear is False and "sliding" in method:
                save_override(scene_id, "method", row["method"], "HandShake Method", reviewer_name)
                fixes_applied += 1
                
        if fixes_applied > 0:
            load_all_logs.clear()
            st.success(f"Successfully auto-fixed and logically moved {fixes_applied} scenes to their correct methods!")
            st.rerun()
        else:
            st.info("All scenes are already correctly mapped based on their linear motion characteristic.")
    st.write("")
    section_header("Auto-Reject Failed Alignments", level="h4")
    st.markdown(
        "Automatically marks any scene as **REJECT** if it failed the automated alignment checks (i.e., appears in the scene fail logs). "
        "These scenes generally have heavily distorted or poorly matched images and are unsafe for dataset export."
    )
    if st.button("Auto-Reject All Alignment Failures", type="primary"):
        rejected_count = 0
        
        review_df = logs.get("review", pd.DataFrame())
        my_reviews = {}
        if not review_df.empty and "scene" in review_df.columns:
            rev_col = review_df.get("reviewer", pd.Series(dtype=str)).fillna("Unknown").replace("", "Unknown")
            my_latest = review_df[rev_col == reviewer_name].drop_duplicates(subset=["scene"], keep="last")
            if "label" in my_latest.columns:
                my_reviews = dict(zip(my_latest["scene"], my_latest["label"].str.upper()))

        for _, row in summary.iterrows():
            my_status = my_reviews.get(str(row["scene"]), "NOT REVIEWED")
            if row.get("failed_count", 0) > 0 and my_status != "REJECT":
                save_review(row["scene"], "REJECT", "Auto-rejected due to pipeline alignment failure", reviewer_name)
                rejected_count += 1
                
        if rejected_count > 0:
            load_all_logs.clear()
            st.success(f"Successfully auto-rejected {rejected_count} failed alignment scenes!")
            st.rerun()
        else:
            st.info("No unreviewed failed alignment scenes found. They either don't exist or have already been reviewed.")

    st.write("")
    section_header("Dataset Modifications Log", level="h4")
    st.markdown("A persistent audit trail of all manual adjustments made to scene categorizations. These modifications dynamically alter the final exported dataset without modifying the underlying raw logs.")
    
    overrides_df = load_overrides()
    if overrides_df.empty:
        st.info("No manual dataset modifications applied yet.")
    else:
        overrides_df_sorted = overrides_df.sort_values("timestamp", ascending=False)
        split_overrides = overrides_df_sorted[overrides_df_sorted["override_type"] == "split"]
        method_overrides = overrides_df_sorted[overrides_df_sorted["override_type"] == "method"]
        
        hist_tabs = st.tabs(["Split Overrides", "Method Overrides"])
        
        with hist_tabs[0]:
            if split_overrides.empty:
                st.info("No split overrides applied.")
            else:
                st.dataframe(split_overrides, use_container_width=True, hide_index=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Clear Split Modifications", type="secondary"):
                    df = load_overrides()
                    df = df[df["override_type"] != "split"]
                    if df.empty:
                        if OVERRIDES_CSV.exists():
                            OVERRIDES_CSV.unlink()
                    else:
                        df.to_csv(OVERRIDES_CSV, index=False)
                    load_all_logs.clear()
                    st.success("Cleared all manual split modifications.")
                    st.rerun()
                
        with hist_tabs[1]:
            if method_overrides.empty:
                st.info("No method overrides applied.")
            else:
                st.dataframe(method_overrides, use_container_width=True, hide_index=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Clear Method Modifications", type="secondary"):
                    df = load_overrides()
                    df = df[df["override_type"] != "method"]
                    if df.empty:
                        if OVERRIDES_CSV.exists():
                            OVERRIDES_CSV.unlink()
                    else:
                        df.to_csv(OVERRIDES_CSV, index=False)
                    load_all_logs.clear()
                    st.success("Cleared all manual method modifications.")
                    st.rerun()


# ======================================================================================
# TAB 3: ASSIGNED SPLITS PREVIEW
# ======================================================================================
with main_tabs[2]:
    section_header("Assigned Splits Preview", level="h3", top_margin="0rem")
    st.markdown("Visually review the images comprising each dataset split before you export.")

    keep_scenes = summary[summary["review_status"] == "KEEP"].copy()
    total_keep = len(keep_scenes)
    exportable = keep_scenes[keep_scenes["split"] != "Unassigned"]
    total_exportable = len(exportable)

    if total_exportable == 0:
        st.info("No KEEP scenes are assigned to an active split (Training, Validation, or Testing).")
    else:
        unique_splits = sorted(exportable["split"].unique())
        split_tabs_ui = st.tabs(unique_splits)
        for sp_idx, sp_name in enumerate(unique_splits):
            with split_tabs_ui[sp_idx]:
                sp_exportable = exportable[exportable["split"] == sp_name].sort_values("scene_number")
                st.metric("Total Assigned Scenes", len(sp_exportable))
                
                ds_tabs = st.tabs(["OIS Images", "Non-OIS Images"])
                
                # OIS
                with ds_tabs[0]:
                    for _, row in sp_exportable.iterrows():
                        scene_dir = resolve_scene_dir(DATASET_256, row["scene"])
                        ois_blur = scene_dir / "ois_blur.jpg"
                        ois_sharp = scene_dir / "ois_sharp.jpg"
                        if ois_blur.exists() or ois_sharp.exists():
                            orig_m = split_rel_path(str(row.get("scene", "")))[0] if str(row.get("scene", "")) else "Unknown"
                            upd_m = str(row.get("method", "Unknown"))
                            m_disp = f"`{upd_m}` (moved from `{orig_m}`)" if orig_m != upd_m else f"`{upd_m}`"
                            st.markdown(f"**Scene:** `{row['scene_name']}` &nbsp;&nbsp;|&nbsp;&nbsp; **Capture:** `{extract_capture_number(row.get('capture'))}` &nbsp;&nbsp;|&nbsp;&nbsp; **Method:** {m_disp}")
                            c1, c2 = st.columns(2)
                            with c1:
                                if ois_blur.exists(): st.image(str(ois_blur), caption="Input (ois_blur)")
                            with c2:
                                if ois_sharp.exists(): st.image(str(ois_sharp), caption="Target (ois_sharp)")
                            st.divider()
                
                # Non-OIS
                with ds_tabs[1]:
                    for _, row in sp_exportable.iterrows():
                        scene_dir = resolve_scene_dir(DATASET_256, row["scene"])
                        nonois_blur = scene_dir / "nonois_blur.jpg"
                        nonois_sharp = scene_dir / "nonois_sharp.jpg"
                        if nonois_blur.exists() or nonois_sharp.exists():
                            orig_m = split_rel_path(str(row.get("scene", "")))[0] if str(row.get("scene", "")) else "Unknown"
                            upd_m = str(row.get("method", "Unknown"))
                            m_disp = f"`{upd_m}` (moved from `{orig_m}`)" if orig_m != upd_m else f"`{upd_m}`"
                            st.markdown(f"**Scene:** `{row['scene_name']}` &nbsp;&nbsp;|&nbsp;&nbsp; **Capture:** `{extract_capture_number(row.get('capture'))}` &nbsp;&nbsp;|&nbsp;&nbsp; **Method:** {m_disp}")
                            c1, c2 = st.columns(2)
                            with c1:
                                if nonois_blur.exists(): st.image(str(nonois_blur), caption="Input (nonois_blur)")
                            with c2:
                                if nonois_sharp.exists(): st.image(str(nonois_sharp), caption="Target (nonois_sharp)")
                            st.divider()

# ======================================================================================
# TAB 4: EXPORT DATASET
# ======================================================================================
with main_tabs[3]:
    section_header("Export Reviewed Dataset", level="h3", top_margin="0rem")
    st.markdown(
        "Export all **KEEP**-reviewed scenes that have an assigned split (Training, Validation, or Testing) "
        "into the final folder structure. Source images come from 256×256 interpolated final output.\n\n"
        "Scenes marked as **Unassigned** will not be exported. Use the Scene Inspector to assign splits first."
    )

    keep_scenes = summary[summary["review_status"] == "KEEP"].copy()
    total_keep = len(keep_scenes)
    exportable = keep_scenes[keep_scenes["split"] != "Unassigned"]
    total_exportable = len(exportable)
    unassigned_keep = total_keep - total_exportable

    if total_keep == 0:
        st.warning("No scenes have been marked as KEEP yet. Review scenes first.")
    elif total_exportable == 0:
        st.warning(f"{total_keep} KEEP scenes found, but none have an assigned split. Use the Scene Inspector to assign Training/Validation/Testing.")
    else:
        # Summary by split and method
        section_header("Scenes to Export", level="h4")
        st.markdown("Breakdown by folder structure and internal scene methods:")
        
        split_method_summary = exportable.groupby(["split", "method"]).size().reset_index(name="Scenes")
        split_method_summary.columns = ["Split", "Method", "Count"]
        
        unique_splits = sorted(exportable["split"].unique())
        export_tabs = st.tabs(unique_splits)
        for i, split_name in enumerate(unique_splits):
            with export_tabs[i]:
                split_folder = SPLIT_EXPORT_MAP.get(split_name, split_name.lower())
                st.caption(f"**Target Paths:** `dataset_ois/{split_folder}/` & `dataset_nonois/{split_folder}/`")
                
                sp_data = split_method_summary[split_method_summary["Split"] == split_name]
                st.metric("Total Output Pairs (per dataset)", sp_data["Count"].sum())
                st.dataframe(sp_data[["Method", "Count"]], use_container_width=True, hide_index=True)

        # Preview table
        if unassigned_keep > 0:
            st.info(f"{unassigned_keep} KEEP scene(s) have no split assigned and will be skipped during export.")

        with st.expander(f"Preview all {total_exportable} exportable KEEP scenes", expanded=False):
            preview_rows = []
            for split_label in sorted(exportable["split"].unique()):
                split_key = SPLIT_EXPORT_MAP.get(split_label, split_label.lower())
                split_scenes_sorted = exportable[exportable["split"] == split_label].sort_values("scene_number")
                for idx, (_, row) in enumerate(split_scenes_sorted.iterrows(), start=1):
                    preview_rows.append({
                        "Export Filename": f"{idx:03d}.jpg",
                        "Original Scene": row["scene_name"],
                        "Split Target": split_key,
                        "Exported Method": row["method"],
                    })
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        section_header("Output Structure", level="h4")
        st.code(
            "dataset_ois/\n"
            "├── train/\n"
            "│   ├── input/   ← ois_blur.jpg (001.jpg, 002.jpg, ...)\n"
            "│   └── target/  ← ois_sharp.jpg (001.jpg, 002.jpg, ...)\n"
            "├── val/\n"
            "└── test/\n"
            "\n"
            "dataset_nonois/\n"
            "├── train/\n"
            "│   ├── input/   ← nonois_blur.jpg\n"
            "│   └── target/  ← nonois_sharp.jpg\n"
            "├── val/\n"
            "└── test/",
            language="text",
        )

        export_path = EXPORT_ROOT
        st.caption(f"Export location: `{export_path.resolve()}`")

        # Check for existing export
        ois_exists = (export_path / "dataset_ois").exists()
        nonois_exists = (export_path / "dataset_nonois").exists()
        if ois_exists or nonois_exists:
            st.warning("Previous export detected. Exporting again will overwrite existing files.")

        exp_cols = st.columns([1, 1, 2])
        with exp_cols[0]:
            if st.button("Export Dataset", use_container_width=True, type="primary", disabled=total_exportable == 0):
                with st.spinner("Exporting dataset..."):
                    progress = st.progress(0, text="Starting export...")
                    try:
                        manifest_df = export_dataset(exportable, DATASET_256, export_path)
                        progress.progress(100, text="Export complete!")
                        st.session_state["last_export_manifest"] = manifest_df
                        st.session_state["last_export_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success(f"Exported {total_exportable} scenes ({len(manifest_df)} files) successfully! Log saved to logs/dataset_export_log.csv")
                    except Exception as e:
                        st.error(f"Export failed: {e}")

