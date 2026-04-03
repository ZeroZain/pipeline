import html
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rawpy
import streamlit as st

st.set_page_config(page_title="Scene Dashboard", layout="wide")

ROOT = Path("workspace")

DATASET_256 = ROOT / "data" / "dataset_256" / "gt_ois"
ALIGNED_COLOR = ROOT / "data" / "aligned" / "gt_ois" / "color"
DECODED_FRAMES = ROOT / "data" / "decoded_frames"

DEBUG_ROOT = ROOT / "debug"
ALIGN_DEBUG = DEBUG_ROOT / "alignment"
INTERP_DEBUG = DEBUG_ROOT / "interpolation"
LAPLACIAN_DEBUG = DEBUG_ROOT / "laplacian"

LOG_DIR = ROOT / "logs"
LAPLACIAN_DIR = LOG_DIR / "laplacian"

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
FRAME_GROUPS = {
    "ois": [
        ("Sharp", "ois_sharp"),
        ("Drop", "ois_drop_frame_actual", "ois_drop_frame"),
        ("Blur", "ois_blur"),
    ],
    "nonois": [
        ("Sharp", "nonois_sharp"),
        ("Drop", "nonois_drop_frame_actual", "nonois_drop_frame"),
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
        "split": parts[0] if len(parts) > 0 else "Unknown",
        "method": parts[1] if len(parts) > 1 else "Unknown",
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
st.markdown(
    """
    <style>
    /* Theme-aware dashboard styles: use Streamlit theme variables */
    :root {
        --sb-bg: var(--backgroundColor);
        --sb-fg: var(--textColor);
        --sb-card: var(--secondaryBackgroundColor);
        --sb-muted: rgba(115,128,150,0.6);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        color: var(--sb-fg);
        background: var(--sb-bg);
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.10);
        border-radius: 12px;
        padding: 0.35rem 0.6rem;
        background: var(--sb-card);
        color: var(--sb-fg);
    }
    .dashboard-card {
        padding:1.1rem 1.2rem;
        border:1px solid rgba(100,110,120,0.06);
        border-radius:16px;
        background:var(--sb-card);
        color:var(--sb-fg);
    }
    .dashboard-card .muted { color:var(--sb-muted); font-weight:600; font-size:0.85rem; }
    .dashboard-badge { display:inline-block; margin:0 0.35rem 0.35rem 0; padding:0.28rem 0.7rem; border-radius:999px; font-size:0.78rem; font-weight:600; }
    .stImage { border-radius:8px; }
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


def resolve_debug_image(scene, folder):
    slug = slugify_rel_path(scene)
    return first_existing_path(
        [
            folder / f"{slug}.jpg",
            folder / f"{slug}.jpeg",
            folder / f"{slug}.png",
        ]
    )


def resolve_laplacian_debug(scene):
    scene_name = parse_scene(scene)["scene_name"]
    return first_existing_path(
        [
            LAPLACIAN_DEBUG / f"{scene_name}.jpg",
            LAPLACIAN_DEBUG / f"{scene_name}.jpeg",
            LAPLACIAN_DEBUG / f"{scene_name}.png",
        ]
    )


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


@st.cache_data(show_spinner=False)
def load_all_logs():
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

    defaults = {
        "max_flow": pd.NA,
        "min_ssim": pd.NA,
        "mean_deltaE_after": pd.NA,
        "failed_count": 0,
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
    summary["has_alignment_debug"] = summary["scene"].map(
        lambda s: resolve_debug_image(s, ALIGN_DEBUG) is not None
    )
    summary["has_interpolation_debug"] = summary["scene"].map(
        lambda s: resolve_debug_image(s, INTERP_DEBUG) is not None
    )
    summary["has_laplacian_debug"] = summary["scene"].map(
        lambda s: resolve_laplacian_debug(s) is not None
    )
    summary["has_ois_laplacian"] = summary["capture"].map(
        lambda c: laplacian_csv_path(c, "ois") is not None
    )
    summary["has_nonois_laplacian"] = summary["capture"].map(
        lambda c: laplacian_csv_path(c, "nonois") is not None
    )
    summary["interpolation_complete"] = summary["interpolation_success"].fillna(0).ge(4)
    summary["needs_attention"] = (
        summary["max_flow"].fillna(-1).gt(FLOW_THRESHOLD)
        | summary["min_ssim"].fillna(1).lt(SSIM_THRESHOLD)
        | summary["failed_count"].fillna(0).gt(0)
    )
    summary["missing_assets"] = ~(
        summary["has_dataset_256"]
        & summary["has_aligned_color"]
        & summary["has_alignment_debug"]
        & summary["has_interpolation_debug"]
        & summary["has_ois_laplacian"]
        & summary["has_nonois_laplacian"]
    )
    return summary


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
    st.subheader(title)
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


def render_debug_image(title, path, empty_message):
    st.subheader(title)
    if path is None:
        st.info(empty_message)
        return
    st.image(str(path), use_container_width=True)
    st.caption(path.name)


def tag_html(text, tone):
    colors = {
        "ok": ("#ecfdf3", "#166534"),
        "info": ("#eff6ff", "#1d4ed8"),
        "warn": ("#fff7ed", "#9a3412"),
        "bad": ("#fef2f2", "#991b1b"),
        "muted": ("#f1f5f9", "#334155"),
    }
    bg, fg = colors[tone]
    return (
        f'<span style="display:inline-block;margin:0 0.35rem 0.35rem 0;'
        f"padding:0.28rem 0.7rem;border-radius:999px;"
        f'background:{bg};color:{fg};font-size:0.78rem;font-weight:600;">'
        f"{html.escape(str(text))}</span>"
    )


def render_scene_header(scene_row, filtered_count, filtered_index):
    scene = scene_row["scene"]
    badges = [
        tag_html(scene_row["split"], "info"),
        tag_html(scene_row["method"], "muted"),
        tag_html(f"Scene {scene_row['scene_name']}", "muted"),
        tag_html(f"{filtered_index}/{filtered_count} in filter", "muted"),
    ]

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
            f'<div style="font-size:1.75rem;font-weight:700;">{html.escape(scene_row["scene_name"])}</div>'
            f'<div class="muted" style="margin-top:0.35rem;">{html.escape(scene)}</div>'
            f'<div style="margin-top:0.8rem;">{"".join(badges)}</div>'
            "</div>"
            '<br>'
        ),
        unsafe_allow_html=True,
    )


def render_asset_metrics(asset_status):
    st.subheader("Asset Coverage")
    items = list(asset_status.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (label, available) in zip(cols, items[start:start + 3]):
            col.metric(label, "Ready" if available else "Missing")


def metric_value(df, frame):
    if df is None or frame is None or pd.isna(frame):
        return None
    match = df[df["frame"] == str(frame)]
    if match.empty:
        return None
    return match["score"].iloc[0]


def frame_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            return str(value)
    return None


def resolve_frame_path(capture, cam_type, frame):
    if capture is None or pd.isna(capture) or frame is None:
        return None
    capture_dir = resolve_scene_dir(DECODED_FRAMES, capture)
    return capture_dir / cam_type / frame


def render_selected_frames(capture, cam_type, lap_df, row):
    cols = st.columns(3)
    # display sizes (pixels)
    width_large = 360
    width_small = 160
    height_large = 240
    height_small = 120

    def _placeholder(w, h, text="No frame"):
        img = np.full((h, w, 3), 220, dtype=np.uint8)
        return img
    for col, spec in zip(cols, FRAME_GROUPS[cam_type]):
        label = spec[0]
        with col:
            st.markdown(f"**{cam_type.upper()} {label}**")

            # If this is a Drop stage with an actual + fallback frame, show both
            if label.lower() == "drop" and len(spec) >= 3:
                actual_key = spec[1]
                fallback_key = spec[2]
                actual_frame = frame_value(row, actual_key)
                fallback_frame = frame_value(row, fallback_key)
                left_col, right_col = st.columns(2)

                with left_col:
                    st.markdown("**Actual**")
                    if actual_frame:
                        path_a = resolve_frame_path(capture, cam_type, actual_frame)
                        if path_a is not None and path_a.exists():
                            img = load_display_image(str(path_a))
                            if img is not None:
                                st.image(img, width=width_small)
                            else:
                                st.warning("Could not decode actual drop frame")
                        else:
                            st.warning("Actual drop frame missing")
                        st.caption(actual_frame)
                        st.caption(f"Laplacian score: {format_num(metric_value(lap_df, actual_frame))}")
                    else:
                        st.image(_placeholder(width_small, height_small), width=width_small)
                        st.caption("No actual drop frame")

                with right_col:
                    st.markdown("**Fallback**")
                    if fallback_frame and fallback_frame != actual_frame:
                        path_f = resolve_frame_path(capture, cam_type, fallback_frame)
                        if path_f is not None and path_f.exists():
                            img = load_display_image(str(path_f))
                            if img is not None:
                                st.image(img, width=width_small)
                            else:
                                st.warning("Could not decode fallback drop frame")
                        else:
                            st.warning("Fallback drop frame missing")
                        st.caption(fallback_frame)
                        st.caption(f"Laplacian score: {format_num(metric_value(lap_df, fallback_frame))}")
                    else:
                        # show same-size placeholder to keep layout consistent
                        st.image(_placeholder(width_small, height_small), width=width_small)
                        st.caption("No fallback drop frame")

                continue

            # Default single-frame stages (Sharp / Blur)
            frame = frame_value(row, *spec[1:])
            path = resolve_frame_path(capture, cam_type, frame)
            score = metric_value(lap_df, frame)
            if path is not None and path.exists():
                image = load_display_image(str(path))
                if image is not None:
                    st.image(image, width=width_large)
                else:
                    st.warning("Could not decode frame")
            else:
                st.warning("Frame missing")
            if frame:
                st.caption(frame)
            st.caption(f"Laplacian score: {format_num(score)}")


def plot_laplacian(df, title, markers):
    if df is None or df.empty:
        st.warning(f"{title} log missing")
        return

    # normalize dataframe for plotting
    df_plot = df.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})

    # Try Altair for an interactive chart (hover shows frame and score). If Altair
    # is not installed, fall back to the static matplotlib plot.
    try:
        import altair as alt

        # Prepare color mapping and domain for legend
        marker_labels = [lbl for lbl, _, _ in markers if lbl]
        unique_marker_labels = []
        for lbl in marker_labels:
            if lbl not in unique_marker_labels:
                unique_marker_labels.append(lbl)

        domain = ["Laplacian"] + [l for l in unique_marker_labels if l != "Laplacian"]
        color_map = {
            "Laplacian": "#2563eb",
            "Sharp": "#16a34a",
            "Drop Actual": "#d97706",
            "Drop Fallback": "#f97316",
            "Drop": "#d97706",
            "Blur": "#dc2626",
        }
        range_colors = [color_map.get(k, "#888888") for k in domain]

        # attach a label for the main line series
        df_plot = df_plot.copy()
        df_plot["label"] = "Laplacian"

        color_encoding_main = alt.Color(
            "label:N",
            scale=alt.Scale(domain=domain, range=range_colors),
            legend=alt.Legend(title="Series", orient="top"),
        )

        # base line and points (both use the same label so they share color 'Laplacian')
        line = (
            alt.Chart(df_plot)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("idx:Q", title="Frame Index"),
                y=alt.Y("score:Q", title="Laplacian Score"),
                color=color_encoding_main,
                tooltip=[alt.Tooltip("frame:N", title="Frame"), alt.Tooltip("score:Q", format=".3f", title="Score"), alt.Tooltip("idx:Q", title="Index")],
            )
        )

        points = (
            alt.Chart(df_plot)
            .mark_circle(size=60)
            .encode(x="idx:Q", y="score:Q", color=color_encoding_main, tooltip=["frame:N", alt.Tooltip("score:Q", format=".3f")])
        )

        # add annotated markers as separate layers but hide their individual legends
        marker_layers = []
        frame_to_idx = {str(f): int(i) for i, f in enumerate(df_plot["frame"]) }
        for label, frame, _color in markers:
            if frame is None:
                continue
            frame_str = str(frame)
            if frame_str in frame_to_idx:
                idx = frame_to_idx[frame_str]
                score = float(df_plot.iloc[idx]["score"])
                mdf = pd.DataFrame([{"idx": idx, "score": score, "frame": frame_str, "label": label}])
                marker_layers.append(
                    alt.Chart(mdf)
                    .mark_point(size=120, filled=True)
                    .encode(
                        x="idx:Q",
                        y="score:Q",
                        color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
                        tooltip=[alt.Tooltip("label:N", title="Pick"), alt.Tooltip("frame:N", title="Frame"), alt.Tooltip("score:Q", format=".3f", title="Score")],
                    )
                )

        layered = alt.layer(line, points, *marker_layers).properties(height=360)
        st.altair_chart(layered, use_container_width=True)

    except Exception:
        # Matplotlib fallback (static)
        df_m = df.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(range(len(df_m)), df_m["score"], linewidth=1.8, color="#2563eb", label="Laplacian")
        ax.grid(alpha=0.25)

        frame_to_index = {frame: idx for idx, frame in enumerate(df_m["frame"]) }
        for label, frame, color in markers:
            if frame in frame_to_index:
                idx = frame_to_index[frame]
                score = df_m.iloc[idx]["score"]
                ax.scatter(idx, score, s=85, color=color, label=label, zorder=3)
                ax.annotate(
                    label,
                    (idx, score),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Laplacian Score")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def build_selection_table(row):
    if row is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "camera": "OIS",
                "sharp": frame_value(row, "ois_sharp"),
                "drop_actual": frame_value(row, "ois_drop_frame_actual"),
                "drop_fallback": frame_value(row, "ois_drop_frame"),
                "blur": frame_value(row, "ois_blur"),
            },
            {
                "camera": "Non-OIS",
                "sharp": frame_value(row, "nonois_sharp"),
                "drop_actual": frame_value(row, "nonois_drop_frame_actual"),
                "drop_fallback": frame_value(row, "nonois_drop_frame"),
                "blur": frame_value(row, "nonois_blur"),
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
            # For Drop stage, include actual and fallback as separate rows when available
            if stage.lower() == "drop" and len(spec) >= 3:
                actual = frame_value(row, spec[1])
                fallback = frame_value(row, spec[2])
                rows.append(
                    {
                        "camera": label,
                        "stage": "Drop (actual)",
                        "frame": actual,
                        "score": metric_value(df, actual),
                    }
                )
                if fallback and fallback != actual:
                    rows.append(
                        {
                            "camera": label,
                            "stage": "Drop (fallback)",
                            "frame": fallback,
                            "score": metric_value(df, fallback),
                        }
                    )
            else:
                frame = frame_value(row, *spec[1:])
                rows.append(
                    {
                        "camera": label,
                        "stage": stage,
                        "frame": frame,
                        "score": metric_value(df, frame),
                    }
                )
    return pd.DataFrame(rows)


def render_dataframe(df, empty_message):
    if df.empty:
        st.info(empty_message)
        return
    st.dataframe(df, use_container_width=True)


def save_review(scene, label, note=""):
    review_path = LOG_DIR / "manual_review.csv"
    row = pd.DataFrame([[scene, label, note]], columns=["scene", "label", "note"])

    if review_path.exists():
        row.to_csv(review_path, mode="a", header=False, index=False)
    else:
        row.to_csv(review_path, index=False)


# consolidated theme-aware CSS applied earlier

logs = load_all_logs()
summary = build_scene_summary(logs)

if summary.empty:
    st.title("Scene Dashboard")
    st.warning("No scenes found in workspace logs.")
    st.stop()

st.title("Scene Dashboard")
st.caption("Inspect outputs, debug composites, laplacian picks, and pipeline logs for each scene.")

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

filtered = summary.copy()
filtered = filtered[
    filtered["split"].isin(selected_splits) & filtered["method"].isin(selected_methods)
]


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
    row["scene"]: (
        f"{row['scene_name']} | {row['method']} | "
        f"flow {format_num(row.get('max_flow'))} | "
        f"ssim {format_num(row.get('min_ssim'), 3)}"
    )
    for _, row in filtered.iterrows()
}

scene = st.sidebar.selectbox(
    "Select scene",
    filtered["scene"].tolist(),
    format_func=lambda value: scene_labels.get(value, value),
)

st.sidebar.header("Dataset Stats")
st.sidebar.metric("Total scenes", int(len(summary)))
st.sidebar.metric("Filtered scenes", int(len(filtered)))
st.sidebar.metric("Bad alignment", int(summary["max_flow"].fillna(-1).gt(FLOW_THRESHOLD).sum()))
st.sidebar.metric("Low SSIM", int(summary["min_ssim"].fillna(1).lt(SSIM_THRESHOLD).sum()))
st.sidebar.metric("Failed scenes", int(summary["failed_count"].fillna(0).gt(0).sum()))
st.sidebar.metric("Missing assets", int(summary["missing_assets"].sum()))

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

capture = None if selection_row is None else selection_row.get("capture")
ois_df = load_laplacian(capture, "ois")
nonois_df = load_laplacian(capture, "nonois")
capture_label = "Missing" if capture is None or pd.isna(capture) else str(capture)

asset_status = {
    "256 Images": bool(scene_row["has_dataset_256"]),
    "Aligned Color": bool(scene_row["has_aligned_color"]),
    "Alignment Debug": bool(scene_row["has_alignment_debug"]),
    "Interpolation Debug": bool(scene_row["has_interpolation_debug"]),
    "OIS Laplacian Log": bool(scene_row["has_ois_laplacian"]),
    "Non-OIS Laplacian Log": bool(scene_row["has_nonois_laplacian"]),
    "Laplacian Debug": bool(scene_row["has_laplacian_debug"]),
}

review_notice = st.session_state.pop("review_notice", None)
if review_notice:
    st.success(review_notice)

render_scene_header(scene_row, len(filtered), scene_position)

review_count = scene_row.get("review_count")
summary_cols = st.columns(5)
summary_cols[0].metric("Quality Score", format_num(scene_row.get("quality_score")))
summary_cols[1].metric("Worst Flow ROI P90", format_num(scene_row.get("max_flow")))
summary_cols[2].metric("Lowest SSIM", format_num(scene_row.get("min_ssim"), 3))
summary_cols[3].metric("Mean deltaE After", format_num(scene_row.get("mean_deltaE_after")))
summary_cols[4].metric("Review Count", int(0 if pd.isna(review_count) else review_count))

issues = []
if above(scene_row.get("max_flow"), FLOW_THRESHOLD):
    issues.append(f"flow ROI P90 is above threshold ({format_num(scene_row['max_flow'])} > {FLOW_THRESHOLD:.0f})")
if below(scene_row.get("min_ssim"), SSIM_THRESHOLD):
    issues.append(f"SSIM is below threshold ({format_num(scene_row['min_ssim'], 3)} < {SSIM_THRESHOLD:.2f})")
if above(scene_row.get("failed_count"), 0):
    issues.append(f"scene has {int(scene_row['failed_count'])} failed checks")
if issues:
    st.warning("Attention: " + " | ".join(issues))

tabs = st.tabs(["Overview", "Laplacian", "Alignment + Interpolation", "Logs + Review"])

with tabs[0]:
    render_asset_metrics(asset_status)

    info_cols = st.columns(3)
    info_cols[0].metric("Capture #", extract_capture_number(capture))
    info_cols[1].metric("Fallback Used", bool_label(scene_row.get("nonois_used_fallback")))
    info_cols[2].metric("Linear Scene", bool_label(scene_row.get("is_linear_scene")))

    selection_table = build_selection_table(selection_row)
    if not selection_table.empty:
        st.subheader("Selected Frames")
        st.dataframe(selection_table, use_container_width=True)

    render_scene_image_grid("Final Dataset", DATASET_256, scene, selection_row)

with tabs[2]:
    debug_cols = st.columns(2)
    with debug_cols[0]:
        render_debug_image(
            "Alignment Debug",
            resolve_debug_image(scene, ALIGN_DEBUG),
            "No alignment debug image found for this scene.",
        )
    with debug_cols[1]:
        render_debug_image(
            "Interpolation Debug",
            resolve_debug_image(scene, INTERP_DEBUG),
            "No interpolation debug image found for this scene.",
        )

    render_scene_image_grid("Aligned Outputs", ALIGNED_COLOR, scene, selection_row)
    render_scene_image_grid("Interpolated 256 Outputs", DATASET_256, scene, selection_row)

with tabs[1]:
    # Show laplacian logs and selection decisions for every capture associated with this scene
    selection_df = logs.get("selection", pd.DataFrame())

    # Gather captures that reference this scene from available logs
    captures = []
    for name in ("selection", "geo", "photo", "color", "interpolation"):
        df = logs.get(name, pd.DataFrame())
        if not df.empty and "scene" in df.columns and "capture" in df.columns:
            caps = df[df["scene"] == scene]["capture"].dropna().astype(str).unique().tolist()
            captures.extend(caps)

    # Fallback to the one in summary/selection if nothing else
    if not captures and capture is not None and not pd.isna(capture):
        captures = [str(capture)]

    captures = sorted({str(c) for c in captures})

    if not captures:
        st.info("No laplacian captures found for this scene.")
    else:
        st.markdown("**Capture summary:**")

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
                    "capture": extract_capture_number(cap),
                    "fallback": bool_label(sel.get("nonois_used_fallback")) if sel is not None else "-",
                    "linear": bool_label(sel.get("is_linear_scene")) if sel is not None else "-",
                    "ready": "Yes" if ois is not None and nonois is not None else "Partial",
                    "ois_sharp_frame": frame_value(sel, "ois_sharp") if sel is not None else "",
                    "ois_drop_actual": frame_value(sel, "ois_drop_frame_actual") if sel is not None else "",
                    "ois_drop_fallback": frame_value(sel, "ois_drop_frame") if sel is not None else "",
                    "ois_blur_frame": frame_value(sel, "ois_blur") if sel is not None else "",
                    "nonois_sharp_frame": frame_value(sel, "nonois_sharp") if sel is not None else "",
                    "nonois_drop_actual": frame_value(sel, "nonois_drop_frame_actual") if sel is not None else "",
                    "nonois_drop_fallback": frame_value(sel, "nonois_drop_frame") if sel is not None else "",
                    "nonois_blur_frame": frame_value(sel, "nonois_blur") if sel is not None else "",
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        # Detailed per-capture expanders with laplacian plots and selected-frame markers
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

                render_debug_image(
                    "Laplacian Debug Composite",
                    resolve_laplacian_debug(scene),
                    "No laplacian debug image found for this scene.",
                )

                plot_cols = st.columns(2)
                with plot_cols[0]:
                    plot_laplacian(
                        ois,
                        "OIS Laplacian Curve",
                        [
                            ("Sharp", frame_value(sel, "ois_sharp"), "#16a34a"),
                            ("Drop Actual", frame_value(sel, "ois_drop_frame_actual"), "#d97706"),
                            ("Drop Fallback", frame_value(sel, "ois_drop_frame"), "#f97316"),
                            ("Blur", frame_value(sel, "ois_blur"), "#dc2626"),
                        ],
                    )
                with plot_cols[1]:
                    plot_laplacian(
                        nonois,
                        "Non-OIS Laplacian Curve",
                        [
                            ("Sharp", frame_value(sel, "nonois_sharp"), "#16a34a"),
                            ("Drop Actual", frame_value(sel, "nonois_drop_frame_actual"), "#d97706"),
                            ("Drop Fallback", frame_value(sel, "nonois_drop_frame"), "#f97316"),
                            ("Blur", frame_value(sel, "nonois_blur"), "#dc2626"),
                        ],
                    )

                st.subheader("Selected Frame Scores")
                st.dataframe(
                    build_laplacian_score_table(sel, ois, nonois),
                    use_container_width=True,
                )

with tabs[3]:
    log_tabs = st.tabs(["Geo", "Photo", "Color", "Interpolation", "Failures", "Review"])

    with log_tabs[0]:
        render_dataframe(geo_scene, "No geo log rows for this scene.")
    with log_tabs[1]:
        render_dataframe(photo_scene, "No photo log rows for this scene.")
    with log_tabs[2]:
        render_dataframe(color_scene, "No color log rows for this scene.")
    with log_tabs[3]:
        render_dataframe(interp_scene, "No interpolation log rows for this scene.")
    with log_tabs[4]:
        render_dataframe(fail_scene, "No scene failures recorded for this scene.")
    with log_tabs[5]:
        st.subheader("Manual Review")

        note = st.text_input("Optional Note")

        review_cols = st.columns(3)

        if review_cols[0].button("KEEP", use_container_width=True):
            save_review(scene, "KEEP", note)
            load_all_logs.clear()
            st.session_state["review_notice"] = f"Saved KEEP for {scene}."
            st.rerun()

        if review_cols[1].button("REJECT", use_container_width=True):
            save_review(scene, "REJECT", note)
            load_all_logs.clear()
            st.session_state["review_notice"] = f"Saved REJECT for {scene}."
            st.rerun()

        if review_cols[2].button("FLAG", use_container_width=True):
            save_review(scene, "FLAG", note)
            load_all_logs.clear()
            st.session_state["review_notice"] = f"Saved FLAG for {scene}."
            st.rerun()

        render_dataframe(review_scene, "No manual review history for this scene yet.")

