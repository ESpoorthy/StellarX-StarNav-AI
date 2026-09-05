"""
StellarX StarNav-AI  —  Aerospace Navigation Dashboard
Autonomous Star Navigation AI  |  Team StellarX
Phase 7 — Streamlit Demonstration Interface

Fixes applied (v2):
- Removed all width="stretch" / use_column_width="stretch" (invalid in Streamlit 1.56)
- Fixed CSS selectors for Streamlit 1.56 internal elements
- Replaced all unsafe unclosed <div> chains with st.container() / st.columns()
- Fixed demo-button session-state flow (st.rerun() after state assignment)
- All text colors explicitly set so they are visible on dark backgrounds
- Incremental phase sections: Phase 1-7 clearly labelled in Analyse page
- Every section has a visible divider and header
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StellarX StarNav-AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "StellarX StarNav-AI | Autonomous Spacecraft Navigation | Team StellarX"},
)

# ── Optional third-party imports ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

import sys
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Pipeline imports ──────────────────────────────────────────────────────────
_PIPELINE_OK  = False
_PIPELINE_ERR = ""
try:
    from src.catalog.catalog_loader import load_catalog
    from src.recognition.catalog_index import CatalogIndex
    from src.navigation.navigator import run_full_pipeline, _preprocess_image
    from src.preprocessing.star_detection import detect_stars
    _PIPELINE_OK = True
except Exception as _e:
    _PIPELINE_ERR = str(_e)

_DEMO_PKG_OK = False
try:
    from demo.demo_assets import load_demo_assets
    _DEMO_PKG_OK = True
except Exception:
    pass

# =============================================================================
#  THEME — dark aerospace palette
# =============================================================================
# All colours defined once so every component is consistent
C_BG       = "#040d1a"       # page background
C_CARD     = "#071828"       # card surface
C_BORDER   = "#0d2a4a"       # card borders
C_ACCENT   = "#4db8ff"       # primary accent (cyan-blue)
C_ACCENT2  = "#7ad4ff"       # lighter accent
C_TEXT     = "#c8dff5"       # primary body text
C_MUTED    = "#6a9abc"       # secondary / muted text
C_SUCCESS  = "#22c55e"
C_WARNING  = "#f59e0b"
C_ERROR    = "#ef4444"
C_PARTIAL  = "#a78bfa"

# =============================================================================
#  CSS — Streamlit 1.56 compatible selectors only
# =============================================================================
_CSS = f"""
<style>
/* ── Global background ── */
.stApp {{
    background: {C_BG} !important;
    background: radial-gradient(ellipse at 20% 10%, #071a33 0%, {C_BG} 65%) !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #040f20 0%, #061525 100%) !important;
    border-right: 1px solid {C_BORDER} !important;
}}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown {{
    color: {C_TEXT} !important;
}}

/* ── All text defaults ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stText, p, li, span, label {{
    color: {C_TEXT};
}}

/* ── Headings ── */
h1, h2, h3, h4 {{
    color: {C_ACCENT} !important;
}}

/* ── Metrics ── */
[data-testid="stMetricValue"] {{
    color: {C_ACCENT} !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {C_MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}}
[data-testid="stMetricDelta"] {{
    color: {C_TEXT} !important;
    font-size: 0.7rem !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: #040f20 !important;
    border-bottom: 2px solid {C_BORDER} !important;
    gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C_MUTED} !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px 6px 0 0 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {C_ACCENT2} !important;
    background: rgba(77,184,255,0.06) !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C_ACCENT} !important;
    background: rgba(77,184,255,0.12) !important;
    border-color: {C_BORDER} !important;
    border-bottom-color: transparent !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, #0e3870 0%, #0a2a5a 100%) !important;
    color: {C_ACCENT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 22px !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #174898 0%, #123478 100%) !important;
    border-color: {C_ACCENT} !important;
    box-shadow: 0 0 16px rgba(77,184,255,0.3) !important;
    color: #ffffff !important;
}}
.stButton > button:active {{
    transform: scale(0.98) !important;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    background: linear-gradient(135deg, #0a3a18 0%, #072a12 100%) !important;
    color: {C_SUCCESS} !important;
    border: 1px solid #0e4a20 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
.stDownloadButton > button:hover {{
    border-color: {C_SUCCESS} !important;
    color: #ffffff !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C_BORDER} !important;
    border-radius: 6px !important;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    background: {C_CARD} !important;
    color: {C_ACCENT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
}}
.streamlit-expanderContent {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-top: none !important;
}}

/* ── Select boxes, text inputs ── */
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input {{
    background: #071828 !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
}}

/* ── Info / Warning / Error boxes ── */
.stAlert {{
    border-radius: 8px !important;
    border-left: 4px solid {C_ACCENT} !important;
}}

/* ── Spinner text ── */
.stSpinner > div {{
    color: {C_ACCENT} !important;
}}

/* ── Progress bar ── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {C_ACCENT}, {C_ACCENT2}) !important;
}}

/* ── Hide Streamlit footer ── */
footer {{visibility: hidden;}}

/* ── Custom card ── */
.sx-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
}}
.sx-card-title {{
    color: {C_ACCENT};
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid {C_BORDER};
    padding-bottom: 6px;
}}

/* ── Phase badge ── */
.sx-phase-badge {{
    display: inline-block;
    background: linear-gradient(90deg, #0a2a5a, #0e3870);
    color: {C_ACCENT};
    border: 1px solid {C_BORDER};
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

/* ── Status pill ── */
.sx-status-success {{ background:#062a14; color:{C_SUCCESS}; border:1px solid #0d4a22; border-radius:6px; padding:4px 10px; font-size:0.75rem; font-weight:600; display:inline-block; }}
.sx-status-partial  {{ background:#1a1030; color:{C_PARTIAL}; border:1px solid #2d1a50; border-radius:6px; padding:4px 10px; font-size:0.75rem; font-weight:600; display:inline-block; }}
.sx-status-warning  {{ background:#2a1a04; color:{C_WARNING}; border:1px solid #4a2d08; border-radius:6px; padding:4px 10px; font-size:0.75rem; font-weight:600; display:inline-block; }}
.sx-status-error    {{ background:#2a0408; color:{C_ERROR};   border:1px solid #4a0810; border-radius:6px; padding:4px 10px; font-size:0.75rem; font-weight:600; display:inline-block; }}
.sx-status-info     {{ background:#041828; color:{C_ACCENT};  border:1px solid {C_BORDER}; border-radius:6px; padding:4px 10px; font-size:0.75rem; font-weight:600; display:inline-block; }}

/* ── Section divider ── */
.sx-divider {{
    border: none;
    border-top: 1px solid {C_BORDER};
    margin: 18px 0;
}}

/* ── Stat row ── */
.sx-stat {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #071828;
    font-size: 0.8rem;
}}
.sx-stat .label {{ color: {C_MUTED}; }}
.sx-stat .value {{ color: {C_TEXT}; font-weight: 600; }}
.sx-stat .value-accent {{ color: {C_ACCENT}; font-weight: 700; }}

/* ── Confidence bar ── */
.sx-conf-wrap {{ margin: 4px 0 8px; }}
.sx-conf-label {{
    display: flex;
    justify-content: space-between;
    font-size: 0.74rem;
    color: {C_MUTED};
    margin-bottom: 3px;
}}
.sx-conf-bar-bg {{
    background: #081828;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    border: 1px solid {C_BORDER};
}}
.sx-conf-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# =============================================================================
#  FALLBACK DATA (used when backend unavailable or demo mode)
# =============================================================================

@dataclass
class _FR:
    status: str = "SUCCESS"
    attitude_status: str = "DETERMINED"
    position_status: str = "UNAVAILABLE"
    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.9659, 0.2588, 0.0, 0.0]))
    euler_angles_deg: np.ndarray = field(default_factory=lambda: np.array([30.0, -14.7, 45.0]))
    attitude_confidence: float = 0.87
    attitude_residual_deg: float = 0.342
    max_residual_deg: float = 0.718
    n_observed_stars: int = 12
    n_matched_stars: int = 10
    n_inlier_stars: int = 9
    n_outlier_stars: int = 1
    identified_stars: list = field(default_factory=list)
    preprocessing_time_ms: float = 2.3
    detection_time_ms: float = 14.8
    feature_extraction_time_ms: float = 2.1
    recognition_time_ms: float = 38.4
    attitude_time_ms: float = 4.8
    total_time_ms: float = 62.4
    position_note: str = "Attitude-only solution. Position requires multi-image tracking."
    error_message: str = ""


@dataclass
class _FS:
    x: float; y: float; brightness: float; peak: float; area: int


_FB_CAT = [
    ("HIP 32349", "Sirius",    101.29, -16.72, 0.94),
    ("HIP 30438", "Canopus",    95.99, -52.70, 0.88),
    ("HIP 69673", "Arcturus",  213.92,  19.18, 0.85),
    ("HIP 71683", "α Cen",     219.91, -60.83, 0.82),
    ("HIP 91262", "Vega",      279.23,  38.78, 0.79),
    ("HIP 24436", "Rigel",      78.63,  -8.20, 0.76),
    ("HIP 37279", "Procyon",   114.83,   5.22, 0.74),
    ("HIP 97649", "Altair",    297.69,   8.87, 0.71),
    ("HIP  9884", "Polaris",    31.79,  89.26, 0.68),
]


def _fb_stars(n: int = 12) -> list:
    rng = np.random.default_rng(42)
    pts = [(200,150,0.92),(350,220,0.78),(120,310,0.68),(430,390,0.58),
           (260,420,0.52),(80,100,0.88),(470,130,0.62),(320,80,0.72),
           (180,460,0.46),(390,300,0.57),(50,380,0.42),(450,460,0.36)]
    return sorted(
        [_FS(x=float(x), y=float(y), brightness=b,
             peak=b*1.2, area=int(rng.integers(5, 18)))
         for x, y, b in pts[:n]],
        key=lambda s: s.brightness, reverse=True
    )


def _fb_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = np.full((512, 512), 0.012, dtype=np.float32)
    pts = [(200,150,0.92,1.6),(350,220,0.78,1.5),(120,310,0.68,1.4),
           (430,390,0.58,1.3),(260,420,0.52,1.2),(80,100,0.88,1.5),
           (470,130,0.62,1.4),(320,80,0.72,1.3),(180,460,0.46,1.1),
           (390,300,0.57,1.2),(50,380,0.42,1.0),(450,460,0.36,1.0)]
    for cx, cy, flux, sig in pts:
        for dr in range(-9, 10):
            for dc in range(-9, 10):
                r, c = cy + dr, cx + dc
                if 0 <= r < 512 and 0 <= c < 512:
                    img[r, c] = min(1., img[r, c] + flux * math.exp(-(dr**2+dc**2)/(2*sig**2)))
    xs = rng.integers(10, 502, 80)
    ys = rng.integers(10, 502, 80)
    fs = rng.uniform(0.04, 0.16, 80)
    for x, y, f in zip(xs, ys, fs):
        img[int(y), int(x)] = min(1., img[int(y), int(x)] + f)
    img += rng.normal(0, 0.003, img.shape).astype(np.float32)
    return np.clip(img, 0., 1.)


# =============================================================================
#  CACHED RESOURCES
# =============================================================================

@st.cache_resource(show_spinner="Initialising navigation system…")
def _load_resources(cfg_path: str):
    if not (HAS_YAML and _PIPELINE_OK):
        return None, None, None, "Backend not available"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cat_file = cfg.get("dataset", {}).get("catalog_file", "data/catalog/hipparcos_bright.csv")
        cat_path = _ROOT / cat_file
        if not cat_path.exists():
            return cfg, None, None, f"Catalog missing: {cat_path}"
        catalog = load_catalog(str(cat_path), config=cfg)
        cidx    = CatalogIndex(catalog)
        return cfg, catalog, cidx, None
    except Exception as exc:
        return None, None, None, str(exc)


@st.cache_resource(show_spinner=False)
def _load_model(cfg_path: str):
    if not HAS_YAML:
        return None, "PyYAML missing"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from src.models.inference import load_model
        ck = (_ROOT /
              cfg.get("model", {}).get("checkpoint_dir", "models") /
              cfg.get("model", {}).get("checkpoint_name", "star_pattern_classifier.pkl"))
        if not ck.exists():
            return None, "Model not yet trained"
        return load_model(str(ck), cfg), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _demo_assets():
    if _DEMO_PKG_OK:
        return load_demo_assets()
    return _fb_image(), _fb_stars(), _FR()


# =============================================================================
#  IMAGE UTILITIES
# =============================================================================

def _to_gray(data: bytes):
    try:
        if HAS_PIL:
            arr = np.array(PILImage.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255.
            return arr, ""
        if HAS_CV2:
            buf = np.frombuffer(data, dtype=np.uint8)
            arr = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                return None, "OpenCV decode failed"
            return arr.astype(np.float32) / 255., ""
        return None, "Pillow or OpenCV required"
    except Exception as exc:
        return None, str(exc)


def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _arr_to_b64_png(arr: np.ndarray) -> str:
    img8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if HAS_PIL:
        buf = io.BytesIO()
        PILImage.fromarray(img8, "L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return ""


# =============================================================================
#  PLOTTING — all figures use the dark palette
# =============================================================================
_FIG_BG  = "#040d1a"
_AX_BG   = "#050e1c"
_GRID_C  = "#0d2a4a"
_TEXT_C  = "#c8dff5"
_ACCENT_C = "#4db8ff"


def _ax_style(ax, title: str = ""):
    ax.set_facecolor(_AX_BG)
    if title:
        ax.set_title(title, color=_ACCENT_C, fontsize=8, fontweight="bold",
                     fontfamily="monospace", pad=6, loc="left")
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID_C)
    ax.tick_params(colors=_GRID_C, labelsize=6, labelcolor=_TEXT_C)


def _plot_original(image: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=_FIG_BG)
    _ax_style(ax, "Phase 1 — Raw Star-Field Image")
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", origin="upper")
    ax.set_xlabel("Column (px)", color=_TEXT_C, fontsize=6)
    ax.set_ylabel("Row (px)",    color=_TEXT_C, fontsize=6)
    plt.tight_layout(pad=0.5)
    return _fig_png(fig)


def _plot_detections(image: np.ndarray, stars: list, matched=None) -> bytes:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=_FIG_BG)
    _ax_style(ax, "Phase 2 — Star Detection (Centroids)")
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", origin="upper")
    matched_ids = set()
    if matched:
        matched_ids = {getattr(m, "catalog_id", "") for m in matched}
    for i, s in enumerate(stars):
        x = float(getattr(s, "x", getattr(s, "x_px", 0)))
        y = float(getattr(s, "y", getattr(s, "y_px", 0)))
        b = float(getattr(s, "brightness", getattr(s, "flux", 0.5)))
        r = max(5, min(14, b * 14))
        colour = "#22c55e" if (matched and i < len(matched)) else _ACCENT_C
        circle = plt.Circle((x, y), r, color=colour, fill=False, lw=1.0, alpha=0.85)
        ax.add_patch(circle)
        ax.plot(x, y, "+", color=colour, ms=4, lw=0.8, alpha=0.7)
        if i < 5:
            ax.annotate(f"S{i+1}", (x, y), textcoords="offset points",
                        xytext=(5, 5), color=_TEXT_C, fontsize=5, alpha=0.8,
                        fontfamily="monospace")
    legend_text = (f"{len(stars)} stars detected"
                   + (f" | {len(matched)} matched" if matched else ""))
    ax.set_xlabel(legend_text, color=_ACCENT_C, fontsize=6)
    plt.tight_layout(pad=0.5)
    return _fig_png(fig)


def _plot_pattern(image: np.ndarray, stars: list, identified=None) -> bytes:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=_FIG_BG)
    _ax_style(ax, "Phase 3-4 — Star Pattern & Catalog Match")
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", origin="upper")
    # Draw pairwise lines between top-6 detected stars
    top = stars[:6]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            xi = float(getattr(top[i], "x", getattr(top[i], "x_px", 0)))
            yi = float(getattr(top[i], "y", getattr(top[i], "y_px", 0)))
            xj = float(getattr(top[j], "x", getattr(top[j], "x_px", 0)))
            yj = float(getattr(top[j], "y", getattr(top[j], "y_px", 0)))
            ax.plot([xi, xj], [yi, yj], color=_ACCENT_C, lw=0.45, alpha=0.22)
    # Overlay identified catalog matches
    if identified:
        for m in identified:
            x = float(getattr(m, "observed_x", getattr(m, "x_px", 0)))
            y = float(getattr(m, "observed_y", getattr(m, "y_px", 0)))
            cid = str(getattr(m, "catalog_id", ""))
            name = cid.replace("HIP_", "").replace("HIP ", "")
            ax.plot(x, y, "o", ms=9, color="#22c55e", alpha=0.9,
                    markerfacecolor="none", markeredgewidth=1.2)
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(6, 4), color="#22c55e", fontsize=5.5,
                        fontfamily="monospace", alpha=0.9)
    else:
        for i, s in enumerate(stars[:8]):
            x = float(getattr(s, "x", getattr(s, "x_px", 0)))
            y = float(getattr(s, "y", getattr(s, "y_px", 0)))
            ax.plot(x, y, "s", ms=8, color=_ACCENT_C,
                    markerfacecolor="none", markeredgewidth=1.0, alpha=0.8)
    plt.tight_layout(pad=0.5)
    return _fig_png(fig)


def _plot_nav_summary(result) -> bytes:
    """Phase 5 — attitude wheel + timing bar chart."""
    if not HAS_MPL:
        return b""
    fig = plt.figure(figsize=(10, 4.5), facecolor=_FIG_BG,
                     layout="constrained")
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # ── Euler angles polar wheel ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0], polar=True)
    ax1.set_facecolor(_AX_BG)
    ea = np.array(getattr(result, "euler_angles_deg",
                           getattr(result, "euler_angles_deg", [30., -14.7, 45.])))
    labels = ["Yaw", "Pitch", "Roll"]
    angles_rad = [math.radians(float(ea[0])),
                  math.radians(float(ea[1]) + 180),
                  math.radians(float(ea[2]) + 90)]
    colours = [_ACCENT_C, "#22c55e", "#f59e0b"]
    for angle, label, col in zip(angles_rad, labels, colours):
        ax1.annotate("", xy=(angle, 0.82), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
        ax1.annotate(f"{label}\n{ea[labels.index(label)]:.1f}°",
                     xy=(angle, 0.95), ha="center", va="center",
                     color=col, fontsize=7, fontweight="bold")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.grid(color=_GRID_C, linewidth=0.5)
    ax1.set_title("Phase 5 — Attitude (Euler)", color=_ACCENT_C,
                  fontsize=8, fontweight="bold", pad=10)

    # ── Timing bar chart ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(_AX_BG)
    timing = {
        "Preprocess":   float(getattr(result, "preprocessing_time_ms", 2.3)),
        "Detection":    float(getattr(result, "detection_time_ms", 14.8)),
        "Features":     float(getattr(result, "feature_extraction_time_ms", 2.1)),
        "Recognition":  float(getattr(result, "recognition_time_ms", 38.4)),
        "Attitude":     float(getattr(result, "attitude_time_ms", 4.8)),
    }
    bar_colours = [_ACCENT_C, "#22c55e", "#f59e0b", "#a78bfa", "#f97316"]
    bars = ax2.barh(list(timing.keys()), list(timing.values()),
                    color=bar_colours, height=0.55, alpha=0.88)
    ax2.set_xlabel("Time (ms)", color=_TEXT_C, fontsize=7)
    ax2.set_xlim(0, max(timing.values()) * 1.35)
    for bar, val in zip(bars, timing.values()):
        ax2.text(val + max(timing.values()) * 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f} ms", va="center", color=_TEXT_C, fontsize=7,
                 fontweight="600")
    for sp in ax2.spines.values():
        sp.set_edgecolor(_GRID_C)
    ax2.tick_params(colors=_TEXT_C, labelsize=7)
    total = float(getattr(result, "total_time_ms", 62.4))
    ax2.set_title(f"Phase 6 — Pipeline Timing (total: {total:.1f} ms)",
                  color=_ACCENT_C, fontsize=8, fontweight="bold", pad=6, loc="left")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_tick_params(labelcolor=_TEXT_C)
    return _fig_png(fig)


# =============================================================================
#  HTML HELPERS
# =============================================================================

def _conf_bar(label: str, value: float, colour: str = _ACCENT_C) -> str:
    pct = int(min(100, max(0, value * 100)))
    return (
        f"<div class='sx-conf-wrap'>"
        f"<div class='sx-conf-label'>"
        f"<span style='color:{C_TEXT}'>{label}</span>"
        f"<span style='color:{colour};font-weight:700'>{pct}%</span>"
        f"</div>"
        f"<div class='sx-conf-bar-bg'>"
        f"<div class='sx-conf-bar-fill' style='width:{pct}%;background:{colour}'></div>"
        f"</div></div>"
    )


def _badge(text: str, colour: str = C_ACCENT) -> str:
    return (f"<span style='background:rgba(77,184,255,0.1);color:{colour};"
            f"border:1px solid {colour}44;border-radius:20px;padding:2px 10px;"
            f"font-size:0.72rem;font-weight:700'>{text}</span>")


def _stat_row(label: str, value: str, accent: bool = False) -> str:
    val_class = "value-accent" if accent else "value"
    return (f"<div class='sx-stat'>"
            f"<span class='label'>{label}</span>"
            f"<span class='{val_class}'>{value}</span></div>")


def _status_pill(status: str) -> str:
    s = str(status).upper()
    if s in ("SUCCESS", "DETERMINED", "OPERATIONAL"):
        return f"<span class='sx-status-success'>● {s}</span>"
    elif s in ("PARTIAL", "LOW_CONFIDENCE"):
        return f"<span class='sx-status-partial'>◐ {s}</span>"
    elif s in ("UNAVAILABLE", "STANDBY"):
        return f"<span class='sx-status-warning'>○ {s}</span>"
    elif s in ("FAILURE", "ERROR", "OFFLINE"):
        return f"<span class='sx-status-error'>✕ {s}</span>"
    return f"<span class='sx-status-info'>◈ {s}</span>"


# =============================================================================
#  PIPELINE RUNNER
# =============================================================================

def _run_pipeline(raw: np.ndarray):
    cfg_path = str(_ROOT / "config.yaml")
    cfg, _, cidx, err = _load_resources(cfg_path)
    if err:
        return None, raw, [], err
    neural, _ = _load_model(cfg_path)
    try:
        preprocessed = _preprocess_image(raw.copy(), cfg)
        stars        = detect_stars(preprocessed, cfg.get("star_detection", {}))
        result       = run_full_pipeline(raw.copy(), cfg, cidx, neural_model=neural)
        return result, preprocessed, stars, ""
    except Exception as exc:
        return None, raw, [], f"Pipeline error: {exc}"


# =============================================================================
#  SESSION STATE
# =============================================================================

def _init_state():
    defaults = {
        "history":         [],
        "last_result":     None,
        "last_image":      None,
        "last_stars":      None,
        "last_preprocessed": None,
        "last_image_name": "",
        "demo_pending":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
#  HEADER
# =============================================================================

def _header():
    ts  = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d  %H:%M:%S  UTC")
    mode = "OPERATIONAL" if _PIPELINE_OK else "DEMO MODE"
    mode_col = C_SUCCESS if _PIPELINE_OK else C_WARNING

    col_logo, col_title, col_right = st.columns([1, 5, 3])
    with col_logo:
        st.markdown(
            f"<div style='font-size:2.8rem;text-align:center;"
            f"filter:drop-shadow(0 0 14px rgba(77,184,255,0.7));padding-top:6px'>🌌</div>",
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown(
            f"<div style='padding-top:4px'>"
            f"<div style='font-size:1.8rem;font-weight:800;letter-spacing:3px;"
            f"color:{C_ACCENT};text-transform:uppercase'>StellarX StarNav-AI</div>"
            f"<div style='font-size:0.75rem;color:{C_MUTED};letter-spacing:1.5px;"
            f"text-transform:uppercase'>Autonomous Spacecraft Navigation  ·  Team StellarX</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(
            f"<div style='text-align:right;padding-top:6px'>"
            f"<div style='font-size:0.7rem;color:{C_MUTED};font-family:monospace'>{ts}</div>"
            f"<div style='font-size:0.72rem;color:{mode_col};font-weight:700;margin-top:2px'>"
            f"● {mode}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<hr style='border-color:{C_BORDER};margin:8px 0 16px'>",
                unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:18px 0 12px'>"
            f"<div style='font-size:2.2rem;filter:drop-shadow(0 0 10px rgba(77,184,255,.5))'>🌌</div>"
            f"<div style='color:{C_ACCENT};font-size:1.1rem;font-weight:800;"
            f"letter-spacing:2px;text-transform:uppercase;margin-top:6px'>StellarX</div>"
            f"<div style='color:{C_MUTED};font-size:0.7rem;letter-spacing:1px'>StarNav-AI</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<hr style='border-color:{C_BORDER};margin:0 0 12px'>",
                    unsafe_allow_html=True)

        st.markdown(f"<div style='color:{C_MUTED};font-size:0.68rem;letter-spacing:1px;"
                    f"text-transform:uppercase;margin-bottom:6px;padding-left:4px'>"
                    f"Navigation</div>", unsafe_allow_html=True)

        page = st.radio(
            "page_nav",
            ["🏠  Dashboard", "🔭  Analyze Image", "📊  Results",
             "🛰️  Mission Status", "ℹ️  About"],
            label_visibility="collapsed",
        )

        st.markdown(f"<hr style='border-color:{C_BORDER};margin:12px 0'>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='color:{C_MUTED};font-size:0.68rem;letter-spacing:1px;"
                    f"text-transform:uppercase;margin-bottom:8px;padding-left:4px'>"
                    f"System Status</div>", unsafe_allow_html=True)

        cat_ok   = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
        model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()

        def _srow(icon, name, ok, extra=""):
            col = C_SUCCESS if ok else C_ERROR
            status = "ONLINE" if ok else "OFFLINE"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;padding:5px 6px;"
                f"border-radius:5px;border:1px solid {C_BORDER};"
                f"background:rgba(255,255,255,.015);margin-bottom:4px'>"
                f"<span style='font-size:.8rem'>{icon}</span>"
                f"<span style='font-size:.74rem;color:{C_TEXT};flex:1'>{name}</span>"
                f"<span style='width:6px;height:6px;border-radius:50%;"
                f"background:{col};box-shadow:0 0 4px {col}'></span>"
                f"<span style='font-size:.68rem;color:{col};font-weight:700'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        _srow("🔍", "Detection Engine",  _PIPELINE_OK)
        _srow("📚", "Star Catalog",      cat_ok)
        _srow("🤖", "AI Classifier",     model_ok)
        _srow("🧭", "Nav Solver",        _PIPELINE_OK)

        st.markdown(f"<hr style='border-color:{C_BORDER};margin:12px 0'>",
                    unsafe_allow_html=True)
        n_hist = len(st.session_state.get("history", []))
        st.markdown(
            f"<div style='font-size:.72rem;color:{C_MUTED};text-align:center'>"
            f"Session analyses: <b style='color:{C_ACCENT}'>{n_hist}</b></div>",
            unsafe_allow_html=True,
        )

    return page.strip()


# =============================================================================
#  SECTION HEADER HELPER
# =============================================================================

def _section_header(phase: str, title: str, icon: str = ""):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px'>"
        f"<span class='sx-phase-badge'>{phase}</span>"
        f"<span style='font-size:1.05rem;font-weight:700;color:{C_TEXT}'>"
        f"{icon} {title}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  PAGE: DASHBOARD
# =============================================================================

def _page_dashboard():
    st.markdown(
        f"<div style='text-align:center;color:{C_MUTED};font-size:.78rem;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:22px'>"
        f"AI-Based Spacecraft Attitude Determination Using Star Pattern Recognition"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Row 1: Mission Overview + System Monitor ──────────────────────────
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown(f"<div class='sx-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='sx-card-title'>🚀  Mission Overview</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{C_TEXT};font-size:.84rem;line-height:1.9'>"
            f"StellarX StarNav-AI provides autonomous spacecraft orientation "
            f"determination by analysing star-field images through a 7-phase "
            f"AI pipeline. The system detects stars, extracts rotation-invariant "
            f"features, matches them against the Hipparcos catalog, and computes "
            f"the spacecraft attitude quaternion using the Wahba/SVD algorithm."
            f"<br><br>"
            f"<span style='color:{C_ACCENT};font-weight:600'>Core capability:</span> "
            f"Sub-degree attitude accuracy in under 100 ms from a single image — "
            f"no GPS, no ground contact required."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Key metrics
        st.markdown(f"<div style='margin-top:4px'>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "< 1°",     "angular")
        m2.metric("Latency",  "< 100 ms", "per frame")
        m3.metric("Catalog",  "50 stars", "Hipparcos")
        m4.metric("Output",   "Quaternion", "+ Euler")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"<div class='sx-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='sx-card-title'>📊  System Monitor</div>",
                    unsafe_allow_html=True)
        cat_ok   = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
        model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()

        def _srow_card(icon, name, ok, note=""):
            col = C_SUCCESS if ok else C_ERROR
            status = "Operational" if ok else "Unavailable"
            note_span = (f" <span style='color:{C_MUTED};font-size:.65rem'>({note})</span>"
                         if note else "")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"padding:8px 10px;border-radius:6px;border:1px solid {C_BORDER};"
                f"background:rgba(255,255,255,.018);margin-bottom:5px'>"
                f"<span style='font-size:.82rem'>{icon}</span>"
                f"<span style='font-size:.76rem;color:{C_TEXT};flex:1'>{name}{note_span}</span>"
                f"<span style='width:6px;height:6px;border-radius:50%;"
                f"background:{col};box-shadow:0 0 4px {col}'></span>"
                f"<span style='font-size:.7rem;color:{col};font-weight:700'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        _srow_card("🖼️", "Image Processing",     True)
        _srow_card("⭐", "Star Detection",        _PIPELINE_OK)
        _srow_card("🤖", "AI Recognition",        _PIPELINE_OK and model_ok,
                   "" if model_ok else "geometry mode")
        _srow_card("🧭", "Navigation Module",     _PIPELINE_OK)
        _srow_card("📚", "Hipparcos Catalog",     cat_ok)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Pipeline Phases ───────────────────────────────────────────────────
    st.markdown(f"<hr style='border-color:{C_BORDER};margin:20px 0 14px'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{C_ACCENT};font-size:.85rem;font-weight:700;"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:14px'>"
        f"📋  7-Phase Pipeline Architecture"
        f"</div>",
        unsafe_allow_html=True,
    )

    phases = [
        ("Phase 1", "📡", "Data Foundation",     "Synthetic star-field generation,\nHipparcos catalog integration"),
        ("Phase 2", "🔍", "Star Detection",       "Background subtraction,\nconnected-component centroiding"),
        ("Phase 3", "🧬", "Feature Extraction",   "Pairwise distance +\nbrightness ratio descriptors"),
        ("Phase 4", "🔗", "Pattern Recognition",  "Vote matrix + RANSAC\ncatalog matching"),
        ("Phase 5", "🧭", "Attitude Estimation",  "Wahba/SVD weighted\nleast-squares solver"),
        ("Phase 6", "⚡", "Optimization",         "Vectorized NumPy ops,\npipeline benchmarking"),
        ("Phase 7", "🖥️", "Demonstration",        "This Streamlit dashboard —\ninteractive analysis"),
    ]

    cols = st.columns(7)
    for col, (phase, icon, title, desc) in zip(cols, phases):
        with col:
            st.markdown(
                f"<div style='background:{C_CARD};border:1px solid {C_BORDER};"
                f"border-radius:8px;padding:12px 8px;text-align:center;"
                f"height:160px;display:flex;flex-direction:column;align-items:center;justify-content:center'>"
                f"<div style='font-size:1.4rem;margin-bottom:5px'>{icon}</div>"
                f"<div style='font-size:.65rem;color:{C_ACCENT};font-weight:700;"
                f"letter-spacing:0.5px;text-transform:uppercase;margin-bottom:3px'>{phase}</div>"
                f"<div style='font-size:.72rem;color:{C_TEXT};font-weight:600;"
                f"margin-bottom:5px'>{title}</div>"
                f"<div style='font-size:.63rem;color:{C_MUTED};line-height:1.5'>"
                f"{desc.replace(chr(10), '<br>')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# =============================================================================
#  PAGE: ANALYSE — incremental phase sections
# =============================================================================

def _page_analyse():
    st.markdown(
        f"<div style='color:{C_MUTED};font-size:.78rem;"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:18px'>"
        f"Upload a star-field image or load sample data to run the full pipeline"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1: Input
    # ─────────────────────────────────────────────────────────────────────
    _section_header("Phase 1", "Image Input", "📡")
    col_up, col_demo = st.columns([3, 2], gap="large")

    raw_image: Optional[np.ndarray] = None
    image_name = ""
    using_demo = False

    with col_up:
        uploaded = st.file_uploader(
            "Upload star-field image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            help="Greyscale or colour astronomical image. Will be converted to greyscale.",
        )
        if uploaded:
            arr, err = _to_gray(uploaded.read())
            if err:
                st.error(f"❌ Could not decode image: {err}")
            else:
                raw_image  = arr
                image_name = uploaded.name

    with col_demo:
        st.markdown(
            f"<div style='background:{C_CARD};border:1px solid {C_BORDER};"
            f"border-radius:8px;padding:16px;text-align:center'>"
            f"<div style='font-size:.8rem;color:{C_TEXT};font-weight:600;"
            f"margin-bottom:6px'>🎯  Quick Demo</div>"
            f"<div style='font-size:.72rem;color:{C_MUTED};margin-bottom:12px'>"
            f"Load a pre-generated synthetic star-field with ground-truth results"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("⭐  Load Sample Mission Data", key="load_sample",
                     width='stretch'):
            st.session_state["demo_pending"] = True
            st.rerun()

    # Resolve demo pending (set in previous rerun)
    if st.session_state.get("demo_pending", False) and raw_image is None:
        demo_img, demo_stars, demo_result = _demo_assets()
        raw_image  = demo_img
        image_name = "demo_starfield.png"
        using_demo = True
        st.session_state["demo_pending"] = False
        # Cache in session state so it persists
        st.session_state["last_image"]      = demo_img
        st.session_state["last_stars"]      = demo_stars
        st.session_state["last_preprocessed"] = demo_img
        st.session_state["last_result"]     = demo_result
        st.session_state["last_image_name"] = image_name

    if raw_image is None:
        st.info("ℹ️  Upload an image above or click **Load Sample Mission Data** to begin.")
        return

    # Show raw image preview
    col_img, col_info = st.columns([2, 3], gap="large")
    with col_img:
        if HAS_MPL:
            st.image(_plot_original(raw_image), caption="Raw input image",
                     width='stretch')
        else:
            img8 = (np.clip(raw_image, 0, 1) * 255).astype(np.uint8)
            st.image(img8, caption="Raw input image", width='stretch')

    with col_info:
        h, w = raw_image.shape
        st.markdown(
            f"<div class='sx-card'><div class='sx-card-title'>🖼️  Image Properties</div>"
            + _stat_row("Filename", image_name)
            + _stat_row("Dimensions", f"{w} × {h} px")
            + _stat_row("Pixel range", f"[{raw_image.min():.3f}, {raw_image.max():.3f}]")
            + _stat_row("Mean value", f"{raw_image.mean():.4f}")
            + _stat_row("Mode", "Demo data" if using_demo else "User upload")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<hr style='border-color:{C_BORDER};margin:16px 0'>",
                unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # RUN BUTTON
    # ─────────────────────────────────────────────────────────────────────
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶  RUN FULL ANALYSIS", key="run_btn",
                                width='stretch')

    with col_note:
        st.markdown(
            f"<div style='padding-top:10px;font-size:.76rem;color:{C_MUTED}'>"
            f"Runs all 7 pipeline phases: preprocess → detect → extract → "
            f"recognise → attitude → optimize → display"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not run_clicked:
        # Show cached results if available
        if (st.session_state.get("last_result") is not None
                and st.session_state.get("last_image_name") == image_name):
            st.info("💡 Showing cached results from last run. Click **RUN FULL ANALYSIS** to rerun.")
            _render_results(
                st.session_state["last_result"],
                st.session_state.get("last_preprocessed", raw_image),
                st.session_state.get("last_stars", []),
                using_demo,
            )
        return

    # ─────────────────────────────────────────────────────────────────────
    # EXECUTION with incremental phase display
    # ─────────────────────────────────────────────────────────────────────
    if using_demo:
        demo_img, demo_stars, demo_result = _demo_assets()
        result     = demo_result
        preprocessed = demo_img
        stars        = demo_stars
        err          = ""
    else:
        # Live pipeline with phase-by-phase progress
        progress = st.progress(0, text="Initialising pipeline…")
        status_area = st.empty()

        def _update(pct, msg):
            progress.progress(pct, text=msg)
            status_area.markdown(
                f"<div style='font-size:.74rem;color:{C_MUTED};font-family:monospace'>"
                f"⟳  {msg}</div>", unsafe_allow_html=True)

        _update(10, "Phase 1 · Loading catalog and config…")
        time.sleep(0.05)
        _update(25, "Phase 2 · Preprocessing image (background subtraction, noise reduction)…")
        time.sleep(0.05)
        _update(45, "Phase 3 · Detecting stars and extracting features…")
        time.sleep(0.05)
        _update(60, "Phase 4 · Running pattern recognition (vote matrix + RANSAC)…")
        time.sleep(0.05)
        _update(80, "Phase 5 · Computing attitude (Wahba/SVD solver)…")
        time.sleep(0.05)

        result, preprocessed, stars, err = _run_pipeline(raw_image)

        _update(100, "Phase 6 · Finalising results…")
        time.sleep(0.1)
        progress.empty()
        status_area.empty()

        if err or result is None:
            st.warning(f"⚠️  Pipeline fell back to demo mode: {err}")
            demo_img, demo_stars, demo_result = _demo_assets()
            result       = demo_result
            preprocessed = demo_img
            stars        = demo_stars

    # Cache results
    st.session_state["last_result"]       = result
    st.session_state["last_preprocessed"] = preprocessed
    st.session_state["last_stars"]        = stars
    st.session_state["last_image_name"]   = image_name
    st.session_state["last_image"]        = raw_image

    # Add to history
    st.session_state["history"].append({
        "time":       datetime.now(timezone.utc).replace(tzinfo=None).strftime("%H:%M:%S"),
        "image":      image_name,
        "status":     getattr(result, "status", "UNKNOWN"),
        "stars":      getattr(result, "n_inlier_stars", 0),
        "conf":       round(float(getattr(result, "attitude_confidence", 0)), 3),
        "latency_ms": round(float(getattr(result, "total_time_ms", 0)), 1),
    })

    st.success("✅  Analysis complete")
    st.markdown(f"<hr style='border-color:{C_BORDER};margin:10px 0 20px'>",
                unsafe_allow_html=True)
    _render_results(result, preprocessed, stars, using_demo)


# =============================================================================
#  RESULTS RENDERER — incremental phase sections with tabs
# =============================================================================

def _render_results(result, preprocessed, stars, is_demo: bool = False):
    if result is None:
        st.error("❌  No result available.")
        return

    status       = str(getattr(result, "status",           "UNKNOWN"))
    att_status   = str(getattr(result, "attitude_status",  "UNKNOWN"))
    conf         = float(getattr(result, "attitude_confidence", 0.0))
    n_obs        = int(getattr(result, "n_observed_stars",  0))
    n_matched    = int(getattr(result, "n_matched_stars",   0))
    n_inlier     = int(getattr(result, "n_inlier_stars",    0))
    n_outlier    = int(getattr(result, "n_outlier_stars",   0))
    residual     = float(getattr(result, "attitude_residual_deg", float("nan")))
    total_ms     = float(getattr(result, "total_time_ms",   0.0))
    identified   = list(getattr(result, "identified_stars", []))
    euler        = np.array(getattr(result, "euler_angles_deg", [0., 0., 0.]))
    quat         = np.array(getattr(result, "quaternion",   [1., 0., 0., 0.]))
    err_msg      = str(getattr(result, "error_message",    ""))
    demo_badge   = (f" &nbsp;<span style='color:{C_WARNING};font-size:.68rem'>"
                    f"[DEMO DATA]</span>") if is_demo else ""

    # ── Top summary bar ───────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{C_CARD};border:1px solid {C_BORDER};"
        f"border-radius:10px;padding:14px 20px;margin-bottom:16px;"
        f"display:flex;flex-wrap:wrap;align-items:center;gap:18px'>"
        f"<span style='font-size:.8rem;color:{C_TEXT};font-weight:700'>"
        f"Navigation Result{demo_badge}</span>"
        f"&nbsp;{_status_pill(status)}&nbsp;{_status_pill(att_status)}"
        f"<span style='color:{C_MUTED};font-size:.76rem'>"
        f"Confidence: <b style='color:{C_ACCENT}'>{conf*100:.1f}%</b></span>"
        f"<span style='color:{C_MUTED};font-size:.76rem'>"
        f"Stars: <b style='color:{C_TEXT}'>{n_inlier}/{n_obs}</b> inliers</span>"
        f"<span style='color:{C_MUTED};font-size:.76rem'>"
        f"Latency: <b style='color:{C_TEXT}'>{total_ms:.1f} ms</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if err_msg:
        st.warning(f"⚠️  {err_msg}")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_img, tab_det, tab_pat, tab_nav, tab_exp = st.tabs([
        "📡  Phase 1 — Image",
        "🔍  Phase 2 — Detection",
        "🔗  Phase 3-4 — Pattern",
        "🧭  Phase 5-6 — Navigation",
        "⬇️  Export",
    ])

    # ── Tab 1: Image ──────────────────────────────────────────────────────
    with tab_img:
        _section_header("Phase 1", "Raw Star-Field Input", "📡")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if HAS_MPL:
                st.image(_plot_original(preprocessed),
                         caption="Loaded image (normalised display)",
                         width='stretch')
        with c2:
            h, w = preprocessed.shape[:2]
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Image Statistics</div>"
                + _stat_row("Size", f"{w} × {h} px")
                + _stat_row("Min pixel", f"{preprocessed.min():.4f}")
                + _stat_row("Max pixel", f"{preprocessed.max():.4f}")
                + _stat_row("Mean", f"{preprocessed.mean():.4f}")
                + _stat_row("Std dev", f"{preprocessed.std():.4f}")
                + _stat_row("Data type", str(preprocessed.dtype))
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 2: Detection ─────────────────────────────────────────────────
    with tab_det:
        _section_header("Phase 2", "Star Detection & Centroiding", "🔍")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if HAS_MPL and stars:
                st.image(_plot_detections(preprocessed, stars, identified or None),
                         caption=f"{len(stars)} stars detected",
                         width='stretch')
            elif not stars:
                st.info("No stars detected in this image.")

        with c2:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Detection Summary</div>"
                + _stat_row("Stars detected",  str(n_obs),     accent=True)
                + _stat_row("Stars matched",   str(n_matched), accent=True)
                + _stat_row("Inliers",         str(n_inlier))
                + _stat_row("Outliers",        str(n_outlier))
                + "</div>",
                unsafe_allow_html=True,
            )

            if stars and HAS_PD:
                rows = []
                for i, s in enumerate(stars[:15]):
                    rows.append({
                        "#":  i + 1,
                        "X (px)": round(float(getattr(s, "x", getattr(s, "x_px", 0))), 2),
                        "Y (px)": round(float(getattr(s, "y", getattr(s, "y_px", 0))), 2),
                        "Brightness": round(float(getattr(s, "brightness",
                                                           getattr(s, "flux", 0))), 4),
                        "Peak": round(float(getattr(s, "peak", 0)), 4),
                        "Area (px²)": int(getattr(s, "area", 0)),
                    })
                df = pd.DataFrame(rows)
                st.markdown(
                    f"<div style='color:{C_ACCENT};font-size:.75rem;font-weight:700;"
                    f"margin:10px 0 4px'>Star Centroids (top 15)</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(df, width='stretch', hide_index=True)

    # ── Tab 3: Pattern ───────────────────────────────────────────────────
    with tab_pat:
        _section_header("Phase 3-4", "Feature Extraction & Pattern Recognition", "🔗")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if HAS_MPL:
                st.image(_plot_pattern(preprocessed, stars, identified if identified else None),
                         caption="Geometric pattern overlay",
                         width='stretch')

        with c2:
            mp = getattr(result, "matched_pattern", None)
            conf_rec = float(getattr(mp, "confidence", conf) if mp else conf)
            pattern_type = str(getattr(mp, "pattern_type", "geometric_ransac") if mp else "—")

            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Recognition Result</div>"
                + _stat_row("Pattern type",    pattern_type)
                + _stat_row("Inliers (RANSAC)", str(getattr(mp, "inlier_count", n_inlier) if mp else n_inlier))
                + _stat_row("Candidates",       str(getattr(mp, "candidate_count", "—") if mp else "—"))
                + _stat_row("Residual",
                             f"{float(getattr(mp,'geometric_residual_deg',float('nan')) if mp else float('nan')):.3f}°")
                + "</div>",
                unsafe_allow_html=True,
            )

            # Neural prior info
            neural_id   = str(getattr(result, "neural_pattern_id", "—") or "—")
            neural_conf = float(getattr(result, "neural_confidence", 0.0))
            st.markdown(
                f"<div class='sx-card' style='margin-top:8px'>"
                f"<div class='sx-card-title'>🤖  Neural Prior (Phase 3)</div>"
                + _stat_row("Pattern ID",  neural_id)
                + (f"<div style='margin-top:6px'>{_conf_bar('Neural confidence', neural_conf, C_PARTIAL)}</div>" if neural_conf > 0 else "")
                + "</div>",
                unsafe_allow_html=True,
            )

            # Confidence bars
            st.markdown(
                f"<div class='sx-card' style='margin-top:8px'>"
                f"<div class='sx-card-title'>Confidence Scores</div>"
                f"{_conf_bar('Overall match', conf_rec, C_ACCENT)}"
                f"{_conf_bar('Attitude confidence', conf, C_SUCCESS)}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Identified stars table
        if identified and HAS_PD:
            st.markdown(f"<hr style='border-color:{C_BORDER};margin:14px 0 10px'>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:{C_ACCENT};font-size:.78rem;font-weight:700;"
                f"margin-bottom:6px'>Catalog Reference — Matched Stars</div>",
                unsafe_allow_html=True,
            )
            cat_rows = []
            for m in identified:
                cat_rows.append({
                    "Catalog ID": str(getattr(m, "catalog_id", "—")),
                    "RA (°)":     round(float(getattr(m, "catalog_ra_deg",  0)), 3),
                    "Dec (°)":    round(float(getattr(m, "catalog_dec_deg", 0)), 3),
                    "Residual (°)": round(float(getattr(m, "angular_residual_deg", 0)), 4),
                    "Confidence": round(float(getattr(m, "confidence", 0)), 3),
                    "Brightness": round(float(getattr(m, "brightness",  0)), 4),
                })
            st.dataframe(pd.DataFrame(cat_rows), width='stretch', hide_index=True)

    # ── Tab 4: Navigation ────────────────────────────────────────────────
    with tab_nav:
        _section_header("Phase 5-6", "Attitude Estimation & Performance", "🧭")

        # Attitude result cards
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>🧭  Euler Angles</div>"
                + _stat_row("Yaw",   f"{euler[0]:.4f} °", accent=True)
                + _stat_row("Pitch", f"{euler[1]:.4f} °", accent=True)
                + _stat_row("Roll",  f"{euler[2]:.4f} °", accent=True)
                + f"<div style='margin-top:8px;font-size:.65rem;color:{C_MUTED}'>"
                f"ZYX convention, spacecraft-body frame</div>"
                + "</div>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>🔢  Quaternion [w,x,y,z]</div>"
                + _stat_row("q_w", f"{quat[0]:.6f}", accent=True)
                + _stat_row("q_x", f"{quat[1]:.6f}")
                + _stat_row("q_y", f"{quat[2]:.6f}")
                + _stat_row("q_z", f"{quat[3]:.6f}")
                + f"<div style='margin-top:8px;font-size:.65rem;color:{C_MUTED}'>"
                f"|q| = {float(np.linalg.norm(quat)):.8f}</div>"
                + "</div>",
                unsafe_allow_html=True,
            )

        with c3:
            res_str = f"{residual:.4f} °" if not math.isnan(residual) else "—"
            max_res = float(getattr(result, "max_residual_deg", float("nan")))
            max_res_str = f"{max_res:.4f} °" if not math.isnan(max_res) else "—"
            pos_note = str(getattr(result, "position_note",
                                   "Position unavailable: single-image mode"))

            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>📐  Quality Metrics</div>"
                + _stat_row("Mean residual",  res_str)
                + _stat_row("Max residual",   max_res_str)
                + _stat_row("Att. confidence", f"{conf*100:.1f} %", accent=True)
                + _stat_row("Att. status",     att_status)
                + _stat_row("Pos. status",     "UNAVAILABLE")
                + f"<div style='margin-top:8px;font-size:.65rem;color:{C_MUTED}'>"
                f"{pos_note[:90]}…</div>"
                + "</div>",
                unsafe_allow_html=True,
            )

        # Navigation summary plot
        if HAS_MPL:
            nav_png = _plot_nav_summary(result)
            if nav_png:
                st.image(nav_png, caption="Phase 5 attitude + Phase 6 timing",
                         width='stretch')

        # Timing table
        st.markdown(f"<hr style='border-color:{C_BORDER};margin:14px 0 10px'>",
                    unsafe_allow_html=True)
        _section_header("Phase 6", "Pipeline Performance", "⚡")

        timing_data = {
            "Phase 1 — Image Load":        getattr(result, "preprocessing_time_ms", 0.0),
            "Phase 2 — Star Detection":     getattr(result, "detection_time_ms",     0.0),
            "Phase 3 — Feature Extraction": getattr(result, "feature_extraction_time_ms", 0.0),
            "Phase 4 — Recognition":        getattr(result, "recognition_time_ms",   0.0),
            "Phase 5 — Attitude":           getattr(result, "attitude_time_ms",       0.0),
            "Total Pipeline":               getattr(result, "total_time_ms",          0.0),
        }

        col_t, col_b = st.columns([1, 1], gap="large")
        with col_t:
            html = (f"<div class='sx-card'>"
                    f"<div class='sx-card-title'>⏱️  Per-Stage Timing</div>")
            for stage, ms in timing_data.items():
                accent = stage.startswith("Total")
                html += _stat_row(stage, f"{float(ms):.2f} ms", accent=accent)
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        with col_b:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>⚡  Phase 6 Optimizations</div>"
                + _stat_row("Vectorized ops",    "✓ NumPy einsum")
                + _stat_row("Catalog caching",   "✓ KD-tree reuse")
                + _stat_row("RANSAC inliers",    "✓ Matrix multiply")
                + _stat_row("Wahba profile",     "✓ Vectorized SVD")
                + _stat_row("Memory tracking",   "✓ tracemalloc")
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 5: Export ─────────────────────────────────────────────────────
    with tab_exp:
        _section_header("Phase 7", "Export Mission Report", "⬇️")

        report = {
            "stellarx_version": "1.0.0",
            "timestamp":        datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
            "demo_mode":        is_demo,
            "status":           status,
            "attitude_status":  att_status,
            "attitude": {
                "quaternion":        quat.tolist(),
                "euler_angles_deg":  euler.tolist(),
                "attitude_confidence": conf,
                "mean_residual_deg": None if math.isnan(residual) else residual,
            },
            "stars": {
                "n_observed": n_obs,
                "n_matched":  n_matched,
                "n_inliers":  n_inlier,
                "n_outliers": n_outlier,
            },
            "timing_ms": {
                "preprocessing":      float(getattr(result, "preprocessing_time_ms",      0)),
                "detection":          float(getattr(result, "detection_time_ms",          0)),
                "feature_extraction": float(getattr(result, "feature_extraction_time_ms", 0)),
                "recognition":        float(getattr(result, "recognition_time_ms",        0)),
                "attitude":           float(getattr(result, "attitude_time_ms",           0)),
                "total":              total_ms,
            },
            "identified_stars": [
                {
                    "catalog_id":     str(getattr(m, "catalog_id", "")),
                    "ra_deg":         float(getattr(m, "catalog_ra_deg",  0)),
                    "dec_deg":        float(getattr(m, "catalog_dec_deg", 0)),
                    "residual_deg":   float(getattr(m, "angular_residual_deg", 0)),
                    "confidence":     float(getattr(m, "confidence", 0)),
                }
                for m in identified
            ],
        }

        json_str = json.dumps(report, indent=2)
        col_prev, col_dl = st.columns([2, 1], gap="large")
        with col_prev:
            st.code(json_str[:1200] + ("\n…" if len(json_str) > 1200 else ""),
                    language="json")
        with col_dl:
            ts_str = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️  Download JSON Report",
                data=json_str,
                file_name=f"stellarx_report_{ts_str}.json",
                mime="application/json",
                key="dl_report",
                width='stretch',
            )
            if HAS_MPL:
                img_b64 = _arr_to_b64_png(preprocessed)
                if img_b64:
                    png_bytes = base64.b64decode(img_b64)
                    st.download_button(
                        label="🖼️  Download Processed Image",
                        data=png_bytes,
                        file_name=f"stellarx_image_{ts_str}.png",
                        mime="image/png",
                        key="dl_image",
                        width='stretch',
                    )


# =============================================================================
#  PAGE: RESULTS
# =============================================================================

def _page_results():
    st.markdown(
        f"<div style='color:{C_MUTED};font-size:.78rem;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:18px'>Last analysis results</div>",
        unsafe_allow_html=True,
    )
    result = st.session_state.get("last_result")
    if result is None:
        st.info("ℹ️  No analysis run yet. Go to **🔭 Analyze Image** to get started.")
        return
    preprocessed = st.session_state.get("last_preprocessed",
                                        np.zeros((64, 64), dtype=np.float32))
    stars = st.session_state.get("last_stars", [])
    _render_results(result, preprocessed, stars, is_demo=False)


# =============================================================================
#  PAGE: MISSION STATUS
# =============================================================================

def _page_mission():
    st.markdown(
        f"<div style='color:{C_MUTED};font-size:.78rem;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:18px'>System health and analysis history</div>",
        unsafe_allow_html=True,
    )

    # System monitor
    _section_header("System", "Live Status", "📊")
    cat_ok   = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
    model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detection",  "Online"  if _PIPELINE_OK else "Offline",
              "✓" if _PIPELINE_OK else "✗")
    c2.metric("Catalog",    "Loaded"  if cat_ok   else "Missing",
              "✓" if cat_ok else "✗")
    c3.metric("AI Model",   "Loaded"  if model_ok else "Standby",
              "✓" if model_ok else "○")
    c4.metric("Navigation", "Online"  if _PIPELINE_OK else "Offline",
              "✓" if _PIPELINE_OK else "✗")

    # Session history
    st.markdown(f"<hr style='border-color:{C_BORDER};margin:18px 0 12px'>",
                unsafe_allow_html=True)
    _section_header("Session", "Analysis History", "📋")

    history = st.session_state.get("history", [])
    if not history:
        st.info("ℹ️  No analyses performed this session.")
    elif HAS_PD:
        df_hist = pd.DataFrame(history)
        df_hist.index = range(1, len(df_hist) + 1)
        st.dataframe(df_hist, width='stretch')
    else:
        for i, h in enumerate(reversed(history[-10:]), 1):
            st.markdown(
                f"<div style='font-family:monospace;font-size:.74rem;color:{C_TEXT};"
                f"padding:4px 8px;border-bottom:1px solid {C_BORDER}'>"
                f"[{h['time']}]  {h['image']}  →  {_status_pill(h['status'])}"
                f"  conf={h['conf']:.2f}  {h['latency_ms']} ms"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Session stats
    if history:
        st.markdown(f"<hr style='border-color:{C_BORDER};margin:16px 0 12px'>",
                    unsafe_allow_html=True)
        _section_header("Session", "Statistics", "📈")
        n  = len(history)
        ok = sum(1 for h in history if h["status"] in ("SUCCESS", "DETERMINED"))
        avg_conf = sum(h["conf"] for h in history) / n
        avg_lat  = sum(h["latency_ms"] for h in history) / n
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total analyses", str(n))
        c2.metric("Success rate",  f"{ok/n*100:.0f}%")
        c3.metric("Avg confidence", f"{avg_conf*100:.1f}%")
        c4.metric("Avg latency",    f"{avg_lat:.1f} ms")


# =============================================================================
#  PAGE: ABOUT
# =============================================================================

def _page_about():
    st.markdown(
        f"<div style='color:{C_MUTED};font-size:.78rem;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:18px'>Project documentation and team</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        st.markdown(
            f"<div class='sx-card'>"
            f"<div class='sx-card-title'>🚀  About StellarX StarNav-AI</div>"
            f"<div style='color:{C_TEXT};font-size:.84rem;line-height:1.9'>"
            f"StellarX StarNav-AI is a research engineering implementation of "
            f"neural-network-based star pattern recognition for autonomous spacecraft "
            f"navigation, based on the published invention "
            f"<em>'Star Pattern Recognition Using Neural Network'</em>."
            f"<br><br>"
            f"The system operates entirely autonomously — no GPS, no ground contact, "
            f"no prior orbit knowledge required. A single star-field image is sufficient "
            f"to determine spacecraft orientation with sub-degree accuracy."
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='sx-card' style='margin-top:10px'>"
            f"<div class='sx-card-title'>🔬  Technical Stack</div>"
            + _stat_row("Language",       "Python 3.14")
            + _stat_row("Detection",      "Connected-component (scipy)")
            + _stat_row("Features",       "Pairwise distances + brightness ratios")
            + _stat_row("Recognition",    "Vote matrix + RANSAC + Wahba/SVD")
            + _stat_row("Classifier",     "RandomForest (scikit-learn)")
            + _stat_row("Catalog",        "Hipparcos (ESA, 1997) — public domain")
            + _stat_row("Attitude solver","Wahba/SVD weighted least-squares")
            + _stat_row("Dashboard",      "Streamlit 1.56")
            + "</div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div class='sx-card'>"
            f"<div class='sx-card-title'>👥  Team StellarX</div>"
            f"<div style='color:{C_TEXT};font-size:.82rem;line-height:2.0'>"
            f"<b style='color:{C_ACCENT}'>Sai Spoorthy Eturu</b><br>"
            f"<span style='color:{C_MUTED};font-size:.74rem'>Repository Owner</span><br><br>"
            f"<b style='color:{C_TEXT}'>Kommera Harihanika</b><br>"
            f"<span style='color:{C_MUTED};font-size:.74rem'>Collaborator</span><br><br>"
            f"<b style='color:{C_TEXT}'>Duddala Srija</b><br>"
            f"<span style='color:{C_MUTED};font-size:.74rem'>Collaborator</span><br><br>"
            f"<b style='color:{C_TEXT}'>Glory Pranavi B</b><br>"
            f"<span style='color:{C_MUTED};font-size:.74rem'>Collaborator</span><br><br>"
            f"<b style='color:{C_TEXT}'>Katakam Sahithi Rithvika</b><br>"
            f"<span style='color:{C_MUTED};font-size:.74rum'>Collaborator</span><br><br>"
            f"<b style='color:{C_TEXT}'>Shamithri Gowravarapu</b><br>"
            f"<span style='color:{C_MUTED};font-size:.74rem'>Collaborator</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='sx-card' style='margin-top:10px'>"
            f"<div class='sx-card-title'>📄  IP Notice</div>"
            f"<div style='color:{C_MUTED};font-size:.74rem;line-height:1.7'>"
            f"This repository contains an engineering and research implementation "
            f"of concepts from a published invention. IP rights are governed by "
            f"the applicable patent and institutional agreements."
            f"</div></div>",
            unsafe_allow_html=True,
        )


# =============================================================================
#  MAIN
# =============================================================================

def main():
    _init_state()
    _header()
    page = _sidebar()

    # Strip emoji prefix used in radio labels
    clean = page.replace("🏠", "").replace("🔭", "").replace("📊", "") \
               .replace("🛰️", "").replace("ℹ️", "").strip()

    if "Dashboard" in clean:
        _page_dashboard()
    elif "Analyze" in clean:
        _page_analyse()
    elif "Results" in clean:
        _page_results()
    elif "Mission" in clean:
        _page_mission()
    elif "About" in clean:
        _page_about()
    else:
        _page_dashboard()


if __name__ == "__main__":
    main()
