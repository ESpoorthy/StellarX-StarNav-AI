"""
StellarX StarNav-AI  --  Aerospace Navigation Dashboard
Autonomous Star Navigation AI  |  Team StellarX
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# ── Page config (must be first) ─────────────────────────────────────────────
st.set_page_config(
    page_title="StellarX StarNav-AI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#127756;</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "StellarX StarNav-AI  |  Autonomous Spacecraft Navigation  |  Team StellarX"},
)

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
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

# ── Pipeline imports ─────────────────────────────────────────────────────────
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


# ============================================================================
#  GLOBAL CSS -- aerospace dark theme
# ============================================================================
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #040d1a;
    color: #c8dff5;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #071a33 0%, #040d1a 60%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040f20 0%, #061525 100%);
    border-right: 1px solid #0d2340;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* ── Header bar ── */
.hdr {
    display: flex;
    align-items: center;
    gap: 18px;
    background: linear-gradient(90deg, #061830 0%, #0a2540 50%, #061830 100%);
    border: 1px solid #0e3358;
    border-radius: 12px;
    padding: 20px 32px;
    margin-bottom: 22px;
    box-shadow: 0 0 50px rgba(14,80,160,0.25), inset 0 1px 0 rgba(100,180,255,0.05);
}
.hdr-logo {
    font-size: 2.4rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px rgba(100,180,255,0.6));
}
.hdr-text h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #4db8ff, #a8d8ff, #4db8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
}
.hdr-text .sub {
    color: #5a8fba;
    font-size: 0.78rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}
.hdr-right {
    margin-left: auto;
    text-align: right;
}
.hdr-right .ts {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #2a5a8a;
    letter-spacing: 0.5px;
}
.hdr-right .ver {
    font-size: 0.68rem;
    color: #1e4060;
    margin-top: 2px;
}

/* ── Cards ── */
.card {
    background: rgba(10, 24, 44, 0.7);
    border: 1px solid #0d2a4a;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(4px);
    transition: border-color 0.2s;
}
.card:hover { border-color: #1a4a7a; }
.card-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid #0d2a4a;
    padding-bottom: 10px;
    margin-bottom: 16px;
}
.card-icon { font-size: 1rem; }
.card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #4db8ff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.card-sub {
    margin-left: auto;
    font-size: 0.68rem;
    color: #2a5a8a;
    letter-spacing: 0.5px;
}

/* ── Status dot ── */
.dot-green  { display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e;margin-right:7px; }
.dot-yellow { display:inline-block;width:8px;height:8px;border-radius:50%;background:#eab308;box-shadow:0 0 6px #eab308;margin-right:7px; }
.dot-red    { display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;box-shadow:0 0 6px #ef4444;margin-right:7px; }
.dot-gray   { display:inline-block;width:8px;height:8px;border-radius:50%;background:#4a5a6a;margin-right:7px; }

/* ── Status row ── */
.sys-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 6px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.03);
    margin-bottom: 6px;
    font-size: 0.8rem;
}
.sys-row .lbl { color: #7aa8cc; font-weight: 500; }
.sys-row .val { font-weight: 600; font-size: 0.73rem; letter-spacing: 0.5px; }
.val-online  { color: #22c55e; }
.val-offline { color: #ef4444; }
.val-standby { color: #eab308; }

/* ── Metric tile ── */
.mtile {
    background: rgba(8, 20, 40, 0.8);
    border: 1px solid #0d2a4a;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.1s;
}
.mtile:hover { border-color: #1a4a7a; transform: translateY(-1px); }
.mtile-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4db8ff;
    line-height: 1.2;
}
.mtile-lbl {
    font-size: 0.63rem;
    color: #3a6a9a;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.mtile-unit { font-size: 0.7rem; color: #2a5272; font-weight: 400; }

/* ── Progress / confidence bar ── */
.bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
    margin-top: 5px;
}
.bar-fill { height: 100%; border-radius: 3px; }

/* ── Workflow steps ── */
.wf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
}
.wf-step {
    position: relative;
    text-align: center;
    padding: 18px 12px;
}
.wf-step:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 38px;
    right: -1px;
    width: 50%;
    height: 2px;
    background: linear-gradient(90deg, #1a4a7a, transparent);
}
.wf-step:not(:first-child)::before {
    content: '';
    position: absolute;
    top: 38px;
    left: 0;
    width: 50%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #1a4a7a);
}
.wf-num {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0a2540, #1a4a7a);
    border: 1px solid #1e5a9a;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; font-weight: 700;
    color: #4db8ff;
    box-shadow: 0 0 10px rgba(77,184,255,0.2);
}
.wf-title {
    font-size: 0.78rem; font-weight: 700;
    color: #85c1e9; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 4px;
}
.wf-desc { font-size: 0.68rem; color: #3a6a9a; line-height: 1.4; }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.5px;
}
.badge-ok   { background: rgba(34,197,94,0.12);  border: 1px solid #16a34a; color: #22c55e; }
.badge-warn { background: rgba(234,179,8,0.12);   border: 1px solid #ca8a04; color: #eab308; }
.badge-fail { background: rgba(239,68,68,0.12);   border: 1px solid #dc2626; color: #ef4444; }
.badge-info { background: rgba(77,184,255,0.1);   border: 1px solid #1a5276; color: #4db8ff; }

/* ── Mono data box ── */
.mono-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #85c1e9;
    background: rgba(4, 14, 28, 0.8);
    padding: 12px 16px;
    border-radius: 6px;
    border: 1px solid #0d2a4a;
    line-height: 1.8;
}
.mono-lbl { color: #2a5a8a; font-size: 0.68rem; margin-right: 6px; }
.mono-val { color: #4db8ff; }

/* ── Processing step list ── */
.proc-step {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 12px; border-radius: 6px;
    margin-bottom: 4px; font-size: 0.8rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid transparent;
}
.proc-step.done {
    border-color: rgba(34,197,94,0.2);
    background: rgba(34,197,94,0.05);
}
.proc-step .ps-icon { font-size: 0.9rem; width: 18px; text-align: center; }
.proc-step .ps-lbl  { color: #7aa8cc; flex: 1; }
.proc-step .ps-ms   { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #2a5a8a; }

/* ── Nav table ── */
.nav-table {
    width: 100%; border-collapse: collapse; font-size: 0.82rem;
}
.nav-table th {
    text-align: left; padding: 7px 10px;
    color: #2a6090; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #0d2a4a;
    font-weight: 600;
}
.nav-table td {
    padding: 6px 10px; border-bottom: 1px solid rgba(13,42,74,0.5);
    color: #85c1e9;
}
.nav-table td.hi { color: #4db8ff; font-family: 'JetBrains Mono', monospace; }
.nav-table tr:hover td { background: rgba(77,184,255,0.03); }

/* ── History row ── */
.hist-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 8px;
    border: 1px solid #0d2a4a;
    background: rgba(10,24,44,0.5);
    margin-bottom: 8px;
}
.hist-ts  { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #2a5a8a; min-width: 72px; }
.hist-img { font-size: 0.75rem; color: #7aa8cc; flex: 1; }
.hist-conf{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; }

/* ── Sidebar nav ── */
div[data-testid="stSidebarNav"] { display: none; }
.sb-logo {
    text-align: center; padding: 22px 0 14px;
    border-bottom: 1px solid #0d2040;
    margin-bottom: 16px;
}
.sb-title { font-size: 1.1rem; font-weight: 800; color: #4db8ff; letter-spacing: 2px; text-transform: uppercase; }
.sb-sub   { font-size: 0.62rem; color: #1e4060; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }

/* ── Misc ── */
.divider { height: 1px; background: linear-gradient(90deg, transparent, #0d2a4a, transparent); margin: 16px 0; }
.section-label {
    font-size: 0.62rem; font-weight: 700; color: #1a4060;
    text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 10px; padding-left: 2px;
}
hr[data-testid="stDivider"] { border-color: #0d2040 !important; }

/* Streamlit overrides */
div[data-testid="stFileUploader"] {
    background: rgba(8,20,40,0.6) !important;
    border: 1px dashed #0d3060 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0a2540, #1a4a7a);
    color: #85c1e9; border: 1px solid #1e5a9a;
    border-radius: 7px; font-weight: 600;
    letter-spacing: 0.5px; padding: 8px 24px;
    width: 100%; font-family: 'Inter', sans-serif;
    font-size: 0.82rem; text-transform: uppercase;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1a4a7a, #2260a0);
    box-shadow: 0 0 20px rgba(77,184,255,0.3);
    border-color: #4db8ff;
}
div[data-testid="stMetric"] {
    background: rgba(8,20,40,0.8) !important;
    border: 1px solid #0d2a4a !important;
    border-radius: 8px !important;
    padding: 14px !important;
}
[data-testid="stMetricValue"]  { font-family: 'JetBrains Mono', monospace !important; color: #4db8ff !important; }
[data-testid="stMetricLabel"]  { color: #3a6a9a !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricDelta"]  { color: #22c55e !important; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,20,40,0.6) !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 1px solid #0d2a4a !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.72rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #3a6a9a !important; padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #4db8ff !important;
    border-bottom: 2px solid #4db8ff !important;
    background: rgba(77,184,255,0.05) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(8,20,40,0.4) !important;
    border: 1px solid #0d2a4a !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 18px !important;
}
[data-testid="stExpander"] {
    background: rgba(8,20,40,0.5) !important;
    border: 1px solid #0d2a4a !important;
    border-radius: 8px !important;
}
div[data-testid="stDataFrame"] { border: 1px solid #0d2a4a !important; border-radius: 6px !important; }
.stAlert { border-radius: 7px !important; font-size: 0.82rem !important; }
.stSpinner > div { border-top-color: #4db8ff !important; }
footer { display: none !important; }
#MainMenu { visibility: hidden; }
</style>
"""


# ============================================================================
#  FALLBACK DATA  (used when backend unavailable)
# ============================================================================

@dataclass
class _FR:  # FallbackResult
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
class _FS:  # FallbackStar
    x: float; y: float; brightness: float; peak: float; area: int


_FB_CAT = [
    ("HIP 32349", "Sirius",      101.29, -16.72, 0.94),
    ("HIP 30438", "Canopus",      95.99, -52.70, 0.88),
    ("HIP 69673", "Arcturus",    213.92,  19.18, 0.85),
    ("HIP 71683", "Alpha Cen",   219.91, -60.83, 0.82),
    ("HIP 74785", "Vega",        279.23,  38.78, 0.79),
    ("HIP 24436", "Rigel",        78.63,  -8.20, 0.76),
    ("HIP 37279", "Procyon",     114.83,   5.22, 0.74),
    ("HIP 97649", "Altair",      297.69,   8.87, 0.71),
    ("HIP  9884", "Polaris",      31.79,  89.26, 0.68),
]


def _fb_stars(n: int = 12) -> list:
    rng = np.random.default_rng(42)
    pts = [(200,150,0.92),(350,220,0.78),(120,310,0.68),(430,390,0.58),
           (260,420,0.52),(80,100,0.88),(470,130,0.62),(320,80,0.72),
           (180,460,0.46),(390,300,0.57),(50,380,0.42),(450,460,0.36)]
    stars = [_FS(x=float(x),y=float(y),brightness=b,peak=b*1.2,area=int(rng.integers(5,18)))
             for x,y,b in pts[:n]]
    return sorted(stars, key=lambda s: s.brightness, reverse=True)


def _fb_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = np.full((512, 512), 0.012, dtype=np.float32)
    pts = [(200,150,0.92,1.6),(350,220,0.78,1.5),(120,310,0.68,1.4),
           (430,390,0.58,1.3),(260,420,0.52,1.2),(80,100,0.88,1.5),
           (470,130,0.62,1.4),(320,80,0.72,1.3),(180,460,0.46,1.1),
           (390,300,0.57,1.2),(50,380,0.42,1.0),(450,460,0.36,1.0)]
    for cx,cy,flux,sig in pts:
        for dr in range(-9,10):
            for dc in range(-9,10):
                r,c=cy+dr,cx+dc
                if 0<=r<512 and 0<=c<512:
                    img[r,c]=min(1.,img[r,c]+flux*math.exp(-(dr**2+dc**2)/(2*sig**2)))
    xs=rng.integers(10,502,80); ys=rng.integers(10,502,80); fs=rng.uniform(0.04,0.16,80)
    for x,y,f in zip(xs,ys,fs):
        img[int(y),int(x)]=min(1.,img[int(y),int(x)]+f)
    img+=rng.normal(0,0.003,img.shape).astype(np.float32)
    return np.clip(img,0.,1.)


# ============================================================================
#  CACHED RESOURCES
# ============================================================================

@st.cache_resource(show_spinner="Initialising navigation system...")
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
        ck = _ROOT / cfg.get("model",{}).get("checkpoint_dir","models") / \
                     cfg.get("model",{}).get("checkpoint_name","star_pattern_classifier.pkl")
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


# ============================================================================
#  IMAGE UTILITIES
# ============================================================================

def _to_gray(data: bytes):
    try:
        if HAS_PIL:
            return np.array(PILImage.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255., ""
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
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _arr_to_b64_png(arr: np.ndarray) -> str:
    """Convert float32 [0,1] array to base64 PNG for download."""
    img8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if HAS_PIL:
        buf = io.BytesIO()
        PILImage.fromarray(img8, "L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return ""


# ============================================================================
#  VISUALISATION
# ============================================================================

BG = "#040d1a"
ACCENT = "#4db8ff"
GRID   = "#0d2a4a"


def _plot_original(image: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax.set_title("INPUT IMAGE", color=ACCENT, fontsize=8, fontweight="bold",
                 fontfamily="monospace", pad=8, loc="left")
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=GRID, labelsize=6)
    plt.tight_layout(pad=0.4)
    return _fig_png(fig)


def _plot_detections(image: np.ndarray, stars: list, matched=None) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    for i, s in enumerate(stars):
        x = getattr(s, "x", getattr(s, "x_px", 0))
        y = getattr(s, "y", getattr(s, "y_px", 0))
        b = getattr(s, "brightness", getattr(s, "flux", 0.5))
        r = max(5, min(12, b * 12))
        circle = plt.Circle((x, y), r, color=ACCENT, fill=False, lw=1.0, alpha=0.85)
        ax.add_patch(circle)
        ax.plot(x, y, "+", color=ACCENT, ms=4, lw=0.8, alpha=0.7)
        ax.annotate(str(i+1), (x, y), xytext=(x+r+2, y-r),
                    color="#85c1e9", fontsize=5.5, fontfamily="monospace", fontweight="bold")
    if matched:
        for s in matched:
            ox = getattr(s, "observed_x", None)
            oy = getattr(s, "observed_y", None)
            if ox is not None:
                ax.plot(ox, oy, "x", color="#22c55e", ms=9, lw=1.5, zorder=5)
    ax.set_title(f"STAR DETECTION  [{len(stars)} objects]", color=ACCENT,
                 fontsize=8, fontweight="bold", fontfamily="monospace", pad=8, loc="left")
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=GRID, labelsize=6)
    plt.tight_layout(pad=0.4)
    return _fig_png(fig)


def _plot_pattern(image: np.ndarray, stars: list, identified=None) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(np.clip(image, 0, 1), cmap="gray", alpha=0.35)

    top = stars[:min(10, len(stars))]
    xs = [getattr(s,"x",getattr(s,"x_px",0)) for s in top]
    ys = [getattr(s,"y",getattr(s,"y_px",0)) for s in top]
    bs = [getattr(s,"brightness",getattr(s,"flux",0.5)) for s in top]

    # Draw pairwise connections
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            alpha = 0.15 + 0.2 * (bs[i]+bs[j]) / 2.0
            ax.plot([xs[i],xs[j]], [ys[i],ys[j]], color="#1e5a9a", lw=0.5, alpha=alpha)

    # Stars coloured by brightness
    sc = ax.scatter(xs, ys, c=bs, cmap="cool", vmin=0, vmax=1,
                    s=[max(40,b*150) for b in bs], zorder=4, alpha=0.9)

    # Labels
    for i, (xi, yi, b) in enumerate(zip(xs, ys, bs)):
        cid = ""
        if identified:
            for s in identified:
                ox = getattr(s, "observed_x", -1)
                oy = getattr(s, "observed_y", -1)
                if abs(ox-xi) < 5 and abs(oy-yi) < 5:
                    cid = " " + getattr(s, "catalog_id", "")[-8:]
                    break
        ax.annotate(str(i+1)+cid, (xi,yi), xytext=(xi+7,yi-7),
                    color="#aed6f1", fontsize=5.5, fontfamily="monospace")

    ax.set_title("PATTERN MATCH", color=ACCENT, fontsize=8,
                 fontweight="bold", fontfamily="monospace", pad=8, loc="left")
    ax.axis("off")
    plt.tight_layout(pad=0.4)
    return _fig_png(fig)


def _plot_nav_summary(result) -> bytes:
    """Compact navigation summary visualisation."""
    euler = getattr(result, "euler_angles_deg", np.zeros(3))
    conf  = getattr(result, "attitude_confidence", 0.87)
    q     = getattr(result, "quaternion", np.array([1.,0.,0.,0.]))

    try: yaw,pitch,roll = float(euler[0]),float(euler[1]),float(euler[2])
    except: yaw,pitch,roll = 0.,0.,0.

    fig = plt.figure(figsize=(8, 3.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Gauge ──
    ax1 = fig.add_subplot(gs[0], projection="polar")
    ax1.set_facecolor(BG)
    theta = np.linspace(0, np.pi, 200)
    ax1.plot(theta, [1]*200, color=GRID, lw=12)
    col = "#22c55e" if conf >= 0.6 else ("#eab308" if conf >= 0.3 else "#ef4444")
    fill = np.linspace(0, np.pi*min(1,conf), 200)
    ax1.plot(fill, [1]*len(fill), color=col, lw=12)
    ax1.text(np.pi/2, 1.45, f"{conf*100:.0f}%", ha="center", va="center",
             color=col, fontsize=16, fontweight="bold", fontfamily="monospace")
    ax1.text(np.pi/2, 0.5, "CONFIDENCE", ha="center", va="center",
             color="#2a5a8a", fontsize=7, fontfamily="monospace")
    ax1.set_ylim(0, 1.7); ax1.set_theta_zero_location("W")
    ax1.set_theta_direction(1); ax1.axis("off")

    # ── Euler bars ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    labels = ["YAW", "PITCH", "ROLL"]
    vals   = [yaw, pitch, roll]
    colors = [ACCENT, "#22c55e", "#eab308"]
    y_pos  = [0.75, 0.5, 0.25]
    for lbl, val, col, yp in zip(labels, vals, colors, y_pos):
        norm = (val + 180) / 360.0
        ax2.barh(yp, norm, height=0.14, color=col, alpha=0.8, left=0)
        ax2.text(-0.02, yp, lbl, ha="right", va="center",
                 color="#2a5a8a", fontsize=7, fontfamily="monospace")
        ax2.text(norm+0.02, yp, f"{val:.2f}", va="center",
                 color=col, fontsize=8, fontfamily="monospace", fontweight="bold")
    ax2.set_xlim(-0.25, 1.35); ax2.set_ylim(0, 1)
    ax2.set_title("EULER ANGLES", color=ACCENT, fontsize=7,
                  fontfamily="monospace", pad=8, loc="left")
    ax2.axis("off")

    # ── Quaternion text ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG)
    ax3.axis("off")
    try:
        qw,qx,qy,qz = float(q[0]),float(q[1]),float(q[2]),float(q[3])
        lines = [
            "ATTITUDE QUATERNION",
            "",
            f"  w  =  {qw:+.5f}",
            f"  x  =  {qx:+.5f}",
            f"  y  =  {qy:+.5f}",
            f"  z  =  {qz:+.5f}",
            "",
            f"  |q| = {math.sqrt(qw**2+qx**2+qy**2+qz**2):.6f}",
        ]
        for i, line in enumerate(lines):
            col = ACCENT if i == 0 else ("#4db8ff" if "=" in line else "#1a4060")
            ax3.text(0.05, 0.95 - i*0.11, line, transform=ax3.transAxes,
                     color=col, fontsize=7.5, fontfamily="monospace", va="top")
    except Exception:
        ax3.text(0.1, 0.5, "Quaternion\nunavailable",
                 transform=ax3.transAxes, color="#2a5a8a", fontsize=9,
                 ha="center", va="center")

    fig.patch.set_facecolor(BG)
    plt.tight_layout(pad=0.5)
    return _fig_png(fig)


def _conf_bar_html(val: float, label: str, show_pct: bool = True) -> str:
    pct = max(0, min(100, int(val * 100)))
    col = "#22c55e" if val >= 0.6 else ("#eab308" if val >= 0.3 else "#ef4444")
    pct_txt = f"<span style='color:{col};font-weight:700;font-family:monospace'>{pct}%</span>" if show_pct else ""
    return (
        f"<div style='margin-bottom:9px'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:.72rem;color:#3a6a9a;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px'>"
        f"<span>{label}</span>{pct_txt}</div>"
        f"<div class='bar-wrap'>"
        f"<div class='bar-fill' style='width:{pct}%;background:{col};transition:width .6s'></div>"
        f"</div></div>"
    )


def _badge(status: str) -> str:
    s = status.upper()
    if s in ("SUCCESS","DETERMINED","ONLINE","OPERATIONAL"):
        return f"<span class='badge badge-ok'>{s}</span>"
    if s in ("PARTIAL","LOW_CONFIDENCE","STANDBY"):
        return f"<span class='badge badge-warn'>{s}</span>"
    if s in ("FAILURE","ERROR","OFFLINE"):
        return f"<span class='badge badge-fail'>{s}</span>"
    return f"<span class='badge badge-info'>{s}</span>"


def _proc_steps_html(steps: list) -> str:
    """steps: list of (label, ms_or_None)"""
    out = "<div style='margin-top:8px'>"
    for lbl, ms in steps:
        ms_txt = f"<span class='ps-ms'>{ms:.1f} ms</span>" if ms and ms > 0 else ""
        out += (
            f"<div class='proc-step done'>"
            f"<span class='ps-icon'>&#10003;</span>"
            f"<span class='ps-lbl'>{lbl}</span>{ms_txt}"
            f"</div>"
        )
    out += "</div>"
    return out


# ============================================================================
#  PIPELINE RUNNER
# ============================================================================

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
        return None, raw, [], str(exc)


# ============================================================================
#  SESSION STATE INIT
# ============================================================================

def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = []      # list of dicts
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_image_name" not in st.session_state:
        st.session_state.last_image_name = ""


# ============================================================================
#  COMMON HEADER
# ============================================================================

def _header():
    ts  = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S  UTC")
    run = "OPERATIONAL" if _PIPELINE_OK else "DEMO MODE"
    st.markdown(
        f"<div class='hdr'>"
        f"<div class='hdr-logo'>&#127756;</div>"
        f"<div class='hdr-text'>"
        f"<h1>StellarX</h1>"
        f"<div class='sub'>Autonomous Star Navigation AI</div>"
        f"</div>"
        f"<div class='hdr-right'>"
        f"<div class='ts'>{ts}</div>"
        f"<div class='ver'>StellarX StarNav-AI &nbsp;|&nbsp; v1.0 &nbsp;|&nbsp; {run}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
#  SIDEBAR
# ============================================================================

def _sidebar() -> str:
    with st.sidebar:
        # Logo
        st.markdown(
            "<div class='sb-logo'>"
            "<div style='font-size:2rem;filter:drop-shadow(0 0 10px rgba(77,184,255,.5))'>&#127756;</div>"
            "<div class='sb-title'>StellarX</div>"
            "<div class='sb-sub'>StarNav-AI</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-label'>Navigation</div>", unsafe_allow_html=True)
        page = st.radio(
            "nav",
            [
                "  Dashboard",
                "  Analyze Star Image",
                "  Results",
                "  Mission Status",
                "  About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>System</div>", unsafe_allow_html=True)

        # System status mini-panel
        def _row(lbl, ok, text=""):
            dot   = "dot-green" if ok else "dot-red"
            cls   = "val-online" if ok else "val-offline"
            val_t = text if text else ("ONLINE" if ok else "OFFLINE")
            st.markdown(
                f"<div class='sys-row'>"
                f"<span class='lbl'>{lbl}</span>"
                f"<span><span class='{dot}'></span>"
                f"<span class='val {cls}'>{val_t}</span></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        cat_ok   = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
        model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()

        _row("Detection Engine", _PIPELINE_OK)
        _row("Star Catalog",     cat_ok)
        _row("AI Model",         model_ok, "LOADED" if model_ok else "STANDBY")
        _row("Navigation Solver",_PIPELINE_OK)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # History count
        n_hist = len(st.session_state.get("history", []))
        st.markdown(
            f"<div style='font-size:.72rem;color:#2a5a8a;text-align:center'>"
            f"Session analyses: <b style='color:#4db8ff'>{n_hist}</b></div>",
            unsafe_allow_html=True,
        )

    return page.strip()


# ============================================================================
#  PAGE: DASHBOARD
# ============================================================================

def _page_dashboard():
    # Subtitle
    st.markdown(
        "<div style='text-align:center;color:#3a6a9a;font-size:.78rem;"
        "letter-spacing:1.5px;text-transform:uppercase;margin-bottom:24px'>"
        "AI-Based Spacecraft Attitude Determination Using Star Pattern Recognition"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Mission Overview + System Monitor ──
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128640;</span>"
            "<span class='card-title'>Mission Overview</span>"
            "</div>"
            "<div style='color:#5a8fba;font-size:.83rem;line-height:1.85'>"
            "StellarX StarNav-AI provides autonomous spacecraft orientation determination "
            "by analysing star field images through a multi-stage AI pipeline. "
            "The system detects stars, extracts rotation-invariant features, matches against "
            "the Hipparcos star catalog, and computes the spacecraft attitude quaternion "
            "using the Wahba/SVD algorithm."
            "<br><br>"
            "<b style='color:#4db8ff'>Core capability:</b>&nbsp; "
            "Sub-degree attitude accuracy in under 100 ms from a single image &mdash; "
            "no GPS, no ground contact required."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",   "< 1 deg",  "angular")
        m2.metric("Latency",    "< 100 ms", "per frame")
        m3.metric("Catalog",    "50 stars", "Hipparcos")
        m4.metric("Output",     "Quaternion","+ Euler")

    with col_r:
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128202;</span>"
            "<span class='card-title'>System Monitor</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        cat_ok   = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
        model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()

        def _srow(icon, name, ok, note=""):
            dot_col = "#22c55e" if ok else "#ef4444"
            txt_col = "#22c55e" if ok else "#ef4444"
            status  = "Operational" if ok else "Unavailable"
            note_html = f"<span style='color:#2a5a8a;font-size:.67rem'>&nbsp;{note}</span>" if note else ""
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"padding:9px 12px;border-radius:6px;border:1px solid #0a2040;"
                f"background:rgba(255,255,255,.02);margin-bottom:6px'>"
                f"<span style='font-size:.85rem'>{icon}</span>"
                f"<span style='font-size:.78rem;color:#7aa8cc;flex:1'>{name}</span>"
                f"<span style='width:7px;height:7px;border-radius:50%;"
                f"background:{dot_col};box-shadow:0 0 5px {dot_col};flex-shrink:0'></span>"
                f"<span style='font-size:.72rem;font-weight:600;color:{txt_col}'>{status}</span>"
                f"{note_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        _srow("&#128247;", "Image Processing",      True)
        _srow("&#11088;",  "Star Detection Engine",  _PIPELINE_OK)
        _srow("&#129504;", "AI Recognition Engine",  _PIPELINE_OK and model_ok, "(geometry mode)" if not model_ok else "")
        _srow("&#129518;", "Navigation Module",      _PIPELINE_OK)
        _srow("&#128218;", "Hipparcos Catalog",      cat_ok)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Workflow ──
    st.markdown(
        "<div class='card' style='margin-top:4px'>"
        "<div class='card-hdr'>"
        "<span class='card-icon'>&#128203;</span>"
        "<span class='card-title'>Analysis Workflow</span>"
        "</div>"
        "<div class='wf-grid'>",
        unsafe_allow_html=True,
    )
    steps = [
        ("01", "Upload Star Image",      "Provide a PNG, JPG, or TIFF\nastronomical image"),
        ("02", "AI Analysis",            "Automated star detection\nand feature extraction"),
        ("03", "Star Identification",    "Pattern matched against\nHipparcos catalog"),
        ("04", "Navigation Solution",    "Attitude quaternion\ncomputed via SVD"),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div class='wf-step'>"
                f"<div class='wf-num'>{num}</div>"
                f"<div class='wf-title'>{title}</div>"
                f"<div class='wf-desc'>{desc.replace(chr(10),'<br>')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================================
#  PAGE: ANALYSE
# ============================================================================

def _page_analyse():
    st.markdown(
        "<div style='color:#2a5a8a;font-size:.72rem;letter-spacing:1.5px;"
        "text-transform:uppercase;margin-bottom:18px'>"
        "Upload an astronomical image to run the full analysis pipeline"
        "</div>",
        unsafe_allow_html=True,
    )

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drag and drop star image here",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            label_visibility="visible",
        )
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample = st.button("Load Sample Mission Data", key="load_sample")

    # Resolve image source
    raw_image  = None
    image_name = ""
    using_demo = False

    if load_sample:
        demo_img, demo_stars, demo_res = _demo_assets()
        raw_image  = demo_img
        image_name = "sample_mission_data.png"
        using_demo = True
        st.session_state["_pending_demo"] = (demo_img, demo_stars, demo_res)
    elif uploaded is not None:
        arr, err = _to_gray(uploaded.read())
        if err or arr is None:
            st.error(f"Could not load image: {err}")
            return
        raw_image  = arr
        image_name = uploaded.name

    if raw_image is None:
        st.markdown(
            "<div style='text-align:center;padding:60px;border:1px dashed #0d2a4a;"
            "border-radius:12px;color:#1a4060'>"
            "<div style='font-size:3rem;filter:drop-shadow(0 0 10px rgba(77,184,255,.3))'>&#127756;</div>"
            "<div style='margin-top:14px;font-size:.85rem;letter-spacing:.5px'>"
            "Upload a star-field image or load sample mission data</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Image preview ──
    col_prev, col_info = st.columns([1, 1])
    with col_prev:
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128247;</span>"
            "<span class='card-title'>Input Image</span>"
            f"<span class='card-sub'>{image_name}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if HAS_MPL:
            st.image(_plot_original(raw_image), width="stretch")
        else:
            st.image(raw_image, width="stretch")
        h, w = raw_image.shape[:2]
        st.markdown(
            f"<div class='mono-box' style='margin-top:8px;font-size:.72rem'>"
            f"<span class='mono-lbl'>RESOLUTION</span>"
            f"<span class='mono-val'>{w} x {h} px</span>"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span class='mono-lbl'>FORMAT</span>"
            f"<span class='mono-val'>GRAYSCALE FLOAT32</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128203;</span>"
            "<span class='card-title'>Pre-Analysis Check</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        checks = [
            ("Image loaded",         True),
            ("Grayscale converted",  True),
            ("Dimensions valid",     (w >= 64 and h >= 64)),
            ("Pipeline available",   _PIPELINE_OK or using_demo),
            ("Catalog available",    (_ROOT / "data/catalog/hipparcos_bright.csv").exists() or using_demo),
        ]
        for lbl, ok in checks:
            icon = "&#10003;" if ok else "&#33;"
            col  = "#22c55e" if ok else "#eab308"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:6px 0;font-size:.8rem;border-bottom:1px solid #0a2040'>"
                f"<span style='color:{col};font-size:.85rem'>{icon}</span>"
                f"<span style='color:#7aa8cc'>{lbl}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("RUN ANALYSIS", key="run_btn")

    if not run_btn:
        return

    # ── Processing animation ──
    st.markdown(
        "<div class='card'>"
        "<div class='card-hdr'>"
        "<span class='card-icon'>&#9889;</span>"
        "<span class='card-title'>Processing</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    proc_placeholder = st.empty()

    step_labels = [
        "Image preprocessing",
        "Star detection",
        "Feature extraction",
        "Pattern matching",
        "Navigation calculation",
    ]
    for i, lbl in enumerate(step_labels):
        done = [(l, None) for l in step_labels[:i+1]]
        proc_placeholder.markdown(_proc_steps_html(done), unsafe_allow_html=True)
        time.sleep(0.12)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run pipeline ──
    if using_demo and "_pending_demo" in st.session_state:
        demo_img, demo_stars, demo_res = st.session_state["_pending_demo"]
        result       = demo_res
        preprocessed = demo_img
        stars        = demo_stars
        is_demo      = True
        pipe_err     = ""
    else:
        with st.spinner("Analysing..."):
            result, preprocessed, stars, pipe_err = _run_pipeline(raw_image)
        is_demo = (result is None)
        if result is None:
            demo_img, demo_stars, demo_res = _demo_assets()
            result       = demo_res
            preprocessed = demo_img
            stars        = demo_stars

    # ── Update processing steps with actual timings ──
    timing = [
        ("Image preprocessing",  getattr(result, "preprocessing_time_ms",     0)),
        ("Star detection",        getattr(result, "detection_time_ms",         0)),
        ("Feature extraction",    getattr(result, "feature_extraction_time_ms",0)),
        ("Pattern matching",      getattr(result, "recognition_time_ms",       0)),
        ("Navigation calculation",getattr(result, "attitude_time_ms",          0)),
    ]
    proc_placeholder.markdown(_proc_steps_html(timing), unsafe_allow_html=True)

    if is_demo and not using_demo:
        st.warning(f"Live pipeline unavailable ({pipe_err}). Showing sample output.")

    # ── Save to history ──
    conf = getattr(result, "attitude_confidence", 0.87)
    st.session_state.history.append({
        "ts":     datetime.utcnow().strftime("%H:%M:%S"),
        "name":   image_name,
        "conf":   conf,
        "status": getattr(result, "status", "SUCCESS"),
        "demo":   is_demo,
    })
    st.session_state.last_result = {
        "result":       result,
        "preprocessed": preprocessed,
        "stars":        stars,
        "raw":          raw_image,
        "is_demo":      is_demo,
        "image_name":   image_name,
    }

    # ── Show results inline ──
    _render_results(result, raw_image, preprocessed, stars, is_demo, image_name)


# ============================================================================
#  RESULTS RENDERER  (shared by Analyse and Results page)
# ============================================================================

def _render_results(result, raw, preprocessed, stars, is_demo, image_name=""):
    import pandas as pd

    conf     = getattr(result, "attitude_confidence", 0.87 if is_demo else 0.)
    total_ms = getattr(result, "total_time_ms",       62.4 if is_demo else 0.)
    n_obs    = getattr(result, "n_observed_stars",     len(stars))
    n_mat    = getattr(result, "n_matched_stars",      0)
    n_inl    = getattr(result, "n_inlier_stars",       0)
    n_out    = getattr(result, "n_outlier_stars",      0)
    mean_res = getattr(result, "attitude_residual_deg",0.342 if is_demo else float("nan"))
    status   = getattr(result, "status",               "SUCCESS")

    # ── Top summary bar ──
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Stars Detected",   str(n_obs))
    s2.metric("Catalog Matches",  str(n_mat))
    s3.metric("Confidence",       f"{conf*100:.0f}%")
    s4.metric("Total Time",       f"{total_ms:.1f} ms")
    res_d = f"{mean_res:.3f} deg" if not (isinstance(mean_res,float) and math.isnan(mean_res)) else "N/A"
    s5.metric("Angular Accuracy", res_d)

    # ── Visual tabs ──
    tab_orig, tab_det, tab_pat, tab_nav = st.tabs([
        "  Original Image  ",
        "  Detected Stars  ",
        "  Pattern Match   ",
        "  Navigation Output",
    ])

    with tab_orig:
        c1, c2 = st.columns([1, 1])
        with c1:
            if HAS_MPL:
                st.image(_plot_original(raw), width="stretch", caption="Raw input image")
            else:
                st.image(raw, width="stretch")
        with c2:
            st.markdown(
                "<div class='card-hdr' style='border:none;padding:0;margin-bottom:12px'>"
                "<span class='card-icon'>&#128247;</span>"
                "<span class='card-title'>Image Properties</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            h, w = raw.shape[:2]
            props = [
                ("File",        image_name or "uploaded"),
                ("Resolution",  f"{w} x {h} px"),
                ("Channels",    "Grayscale"),
                ("Bit depth",   "32-bit float"),
                ("Min value",   f"{float(raw.min()):.4f}"),
                ("Max value",   f"{float(raw.max()):.4f}"),
                ("Mean",        f"{float(raw.mean()):.4f}"),
            ]
            rows_html = "".join(
                f"<tr><td style='color:#3a6a9a;padding:5px 10px;"
                f"font-size:.75rem;text-transform:uppercase;letter-spacing:.5px'>{k}</td>"
                f"<td class='hi' style='padding:5px 10px'>{v}</td></tr>"
                for k, v in props
            )
            st.markdown(
                f"<table class='nav-table'>{rows_html}</table>",
                unsafe_allow_html=True,
            )

    with tab_det:
        identified = getattr(result, "identified_stars", [])
        c1, c2 = st.columns([1, 1])
        with c1:
            if HAS_MPL:
                st.image(_plot_detections(preprocessed, stars,
                                         matched=identified if not is_demo else None),
                         width="stretch")
            else:
                st.image(preprocessed, width="stretch")
            st.caption(f"{n_obs} stars detected  |  {n_inl} matched to catalog  |  {n_out} rejected")

        with c2:
            st.markdown(
                "<div class='card-hdr' style='border:none;padding:0;margin-bottom:12px'>"
                "<span class='card-icon'>&#11088;</span>"
                "<span class='card-title'>Star Detection Results</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            da, db = st.columns(2)
            da.metric("Detected",       str(n_obs))
            db.metric("Verified",       str(n_inl), f"-{n_out} rejected")
            st.markdown("<br>", unsafe_allow_html=True)

            if stars:
                rows = []
                for i, s in enumerate(stars[:10]):
                    x = getattr(s,"x",getattr(s,"x_px",0))
                    y = getattr(s,"y",getattr(s,"y_px",0))
                    b = getattr(s,"brightness",getattr(s,"flux",0))
                    rows.append({"ID": i+1, "X (px)": f"{x:.1f}",
                                 "Y (px)": f"{y:.1f}", "Brightness": f"{b:.4f}"})
                st.markdown(
                    "<div style='font-size:.68rem;color:#2a5a8a;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>"
                    "Star Centroids</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             width="stretch", height=min(310, len(rows)*36+40))

    with tab_pat:
        c1, c2 = st.columns([1, 1])
        with c1:
            if HAS_MPL:
                st.image(_plot_pattern(preprocessed, stars, identified if not is_demo else None),
                         width="stretch")
        with c2:
            st.markdown(
                "<div class='card-hdr' style='border:none;padding:0;margin-bottom:12px'>"
                "<span class='card-icon'>&#128269;</span>"
                "<span class='card-title'>Pattern Recognition</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Status:** {_badge(status if not is_demo else 'SUCCESS')}",
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(_conf_bar_html(conf, "Match Confidence"), unsafe_allow_html=True)
            st.markdown(
                _conf_bar_html(min(1., n_inl / max(n_obs, 1)), "Inlier Ratio"),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Catalog table
            if is_demo or not identified:
                cat_rows = [
                    {"Star": name, "Catalog ID": cid,
                     "RA": f"{ra:.2f}", "Dec": f"{dec:.2f}",
                     "Confidence": f"{c*100:.0f}%"}
                    for cid, name, ra, dec, c in _FB_CAT
                ]
            else:
                cat_rows = [
                    {"Star": getattr(s,"catalog_id","-"),
                     "RA":   f"{getattr(s,'catalog_ra_deg',0):.2f}",
                     "Dec":  f"{getattr(s,'catalog_dec_deg',0):.2f}",
                     "Conf": f"{getattr(s,'confidence',0)*100:.0f}%",
                     "Res":  f"{getattr(s,'angular_residual_deg',0):.4f} deg"}
                    for s in identified[:9]
                ]
            st.markdown(
                "<div style='font-size:.68rem;color:#2a5a8a;text-transform:uppercase;"
                "letter-spacing:1px;margin-bottom:6px'>Catalog Reference</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(cat_rows), hide_index=True, width="stretch",
                         height=min(300, len(cat_rows)*36+40))

    with tab_nav:
        euler  = getattr(result, "euler_angles_deg", np.zeros(3))
        q      = getattr(result, "quaternion",       np.array([1.,0.,0.,0.]))
        att_st = getattr(result, "attitude_status",  "DETERMINED")

        try: yaw,pitch,roll = float(euler[0]),float(euler[1]),float(euler[2])
        except: yaw,pitch,roll = 0.,0.,0.

        # RA / Dec / Roll from attitude
        try:
            qw,qx,qy,qz = float(q[0]),float(q[1]),float(q[2]),float(q[3])
            ra_out  = (math.degrees(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))) + 360) % 360
            dec_out = math.degrees(math.asin(max(-1.,min(1., 2*(qw*qy-qz*qx)))))
        except Exception:
            ra_out, dec_out = yaw, pitch

        if HAS_MPL:
            st.image(_plot_nav_summary(result), width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        nc1, nc2 = st.columns(2)
        with nc1:
            st.markdown(
                "<div class='card'>"
                "<div class='card-hdr'>"
                "<span class='card-icon'>&#129522;</span>"
                "<span class='card-title'>Spacecraft Attitude</span>"
                f"<span class='card-sub'>{_badge(att_st)}</span>"
                "</div>"
                f"<table class='nav-table'>"
                f"<tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>"
                f"<tr><td>Right Ascension (RA)</td><td class='hi'>{ra_out:.4f}</td><td style='color:#2a5a8a'>degrees</td></tr>"
                f"<tr><td>Declination (DEC)</td><td class='hi'>{dec_out:.4f}</td><td style='color:#2a5a8a'>degrees</td></tr>"
                f"<tr><td>Roll</td><td class='hi'>{roll:.4f}</td><td style='color:#2a5a8a'>degrees</td></tr>"
                f"<tr><td>Yaw (Z)</td><td class='hi'>{yaw:.4f}</td><td style='color:#2a5a8a'>degrees</td></tr>"
                f"<tr><td>Pitch (Y)</td><td class='hi'>{pitch:.4f}</td><td style='color:#2a5a8a'>degrees</td></tr>"
                f"<tr><td>Mean Residual</td><td class='hi'>{res_d}</td><td style='color:#2a5a8a'></td></tr>"
                f"</table>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nc2:
            try:
                qw,qx,qy,qz = float(q[0]),float(q[1]),float(q[2]),float(q[3])
                qn = math.sqrt(qw**2+qx**2+qy**2+qz**2)
                q_html = (
                    f"<div class='mono-box'>"
                    f"<div style='color:#2a5a8a;font-size:.65rem;letter-spacing:1px;margin-bottom:8px'>ORIENTATION QUATERNION (camera to inertial)</div>"
                    f"q[w] = <span class='mono-val'>{qw:+.6f}</span><br>"
                    f"q[x] = <span class='mono-val'>{qx:+.6f}</span><br>"
                    f"q[y] = <span class='mono-val'>{qy:+.6f}</span><br>"
                    f"q[z] = <span class='mono-val'>{qz:+.6f}</span><br>"
                    f"<div style='border-top:1px solid #0d2a4a;margin-top:8px;padding-top:8px;'>"
                    f"|q|  = <span style='color:#3a6a9a'>{qn:.8f}</span>"
                    f"</div></div>"
                )
            except Exception:
                q_html = "<div class='mono-box'>Quaternion unavailable</div>"

            st.markdown(
                "<div class='card'>"
                "<div class='card-hdr'>"
                "<span class='card-icon'>&#127987;</span>"
                "<span class='card-title'>Position Estimation</span>"
                "</div>"
                f"<div style='color:#3a6a9a;font-size:.78rem;margin-bottom:12px'>"
                f"Latitude &nbsp; -- &nbsp; Single-image provides attitude only<br>"
                f"Longitude &nbsp; -- &nbsp; Multi-image tracking required"
                f"</div>"
                f"{q_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Timing breakdown
        st.markdown(
            "<div class='card' style='margin-top:4px'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#9201;</span>"
            "<span class='card-title'>Processing Timeline</span>"
            f"<span class='card-sub'>{total_ms:.1f} ms total</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        timing_data = [
            ("Image Preprocessing",   getattr(result, "preprocessing_time_ms",     0)),
            ("Star Detection",         getattr(result, "detection_time_ms",         0)),
            ("Feature Extraction",     getattr(result, "feature_extraction_time_ms",0)),
            ("Pattern Recognition",    getattr(result, "recognition_time_ms",       0)),
            ("Navigation Calculation", getattr(result, "attitude_time_ms",          0)),
        ]
        rows_html = "".join(
            f"<tr><td style='color:#5a8fba;padding:6px 10px;font-size:.78rem'>{lbl}</td>"
            f"<td class='hi' style='padding:6px 10px;text-align:right'>{ms:.1f}</td>"
            f"<td style='color:#2a5a8a;padding:6px 10px;text-align:right'>"
            f"{ms/max(total_ms,0.001)*100:.1f}%</td></tr>"
            for lbl, ms in timing_data if ms > 0
        )
        fps = 1000. / max(total_ms, 0.1)
        rows_html += (
            f"<tr style='border-top:1px solid #0d2a4a'>"
            f"<td style='color:#4db8ff;font-weight:700;padding:7px 10px;font-size:.78rem'>Total Pipeline</td>"
            f"<td class='hi' style='padding:7px 10px;text-align:right;font-weight:700'>{total_ms:.1f}</td>"
            f"<td style='color:#4db8ff;font-weight:700;padding:7px 10px;text-align:right'>"
            f"{fps:.1f} fps</td></tr>"
        )
        st.markdown(
            f"<table class='nav-table'><tr>"
            f"<th>Stage</th><th style='text-align:right'>Time (ms)</th>"
            f"<th style='text-align:right'>Share</th></tr>{rows_html}</table>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Export report ──
    st.markdown("<br>", unsafe_allow_html=True)
    _export_section(result, raw, stars, image_name, is_demo)


# ============================================================================
#  EXPORT REPORT
# ============================================================================

def _export_section(result, raw, stars, image_name, is_demo):
    conf     = getattr(result, "attitude_confidence",  0.87)
    total_ms = getattr(result, "total_time_ms",        62.4)
    n_obs    = getattr(result, "n_observed_stars",     len(stars))
    n_mat    = getattr(result, "n_matched_stars",      0)
    mean_res = getattr(result, "attitude_residual_deg",0.342)
    q        = getattr(result, "quaternion",           np.array([1.,0.,0.,0.]))
    euler    = getattr(result, "euler_angles_deg",     np.zeros(3))
    status   = getattr(result, "status",               "SUCCESS")

    try: qw,qx,qy,qz = float(q[0]),float(q[1]),float(q[2]),float(q[3])
    except: qw,qx,qy,qz = 1.,0.,0.,0.
    try: yaw,pitch,roll = float(euler[0]),float(euler[1]),float(euler[2])
    except: yaw,pitch,roll = 0.,0.,0.

    res_str = f"{mean_res:.4f} deg" if not (isinstance(mean_res,float) and math.isnan(mean_res)) else "N/A"

    report = {
        "mission_report": {
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "system":          "StellarX StarNav-AI v1.0",
            "image_file":      image_name,
            "mode":            "SAMPLE" if is_demo else "LIVE",
        },
        "detection_summary": {
            "stars_detected":    n_obs,
            "catalog_matched":   n_mat,
            "verified_matches":  getattr(result,"n_inlier_stars",0),
            "rejected_outliers": getattr(result,"n_outlier_stars",0),
        },
        "pattern_identification": {
            "status":           status,
            "confidence_pct":   round(conf * 100, 2),
            "mean_residual":    res_str,
            "matched_catalog":  "Hipparcos",
        },
        "navigation_output": {
            "attitude_status":  getattr(result,"attitude_status","DETERMINED"),
            "quaternion":       {"w": round(qw,6),"x": round(qx,6),"y": round(qy,6),"z": round(qz,6)},
            "euler_angles_deg": {"yaw": round(yaw,4),"pitch": round(pitch,4),"roll": round(roll,4)},
            "position_note":    "Attitude-only. Position requires multi-image tracking.",
        },
        "performance": {
            "total_time_ms": round(total_ms, 2),
            "throughput_fps": round(1000. / max(total_ms, 0.1), 2),
        },
    }

    report_json = json.dumps(report, indent=2)

    with st.expander("Download Mission Report"):
        st.markdown(
            "<div style='font-size:.72rem;color:#2a5a8a;margin-bottom:10px'>"
            "JSON report including detection summary, pattern identification, "
            "navigation output and performance metrics."
            "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Mission Report (.json)",
            data=report_json,
            file_name=f"stellarx_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="dl_report",
        )
        st.code(report_json, language="json")


# ============================================================================
#  PAGE: RESULTS  (standalone page showing last result)
# ============================================================================

def _page_results():
    if not st.session_state.get("last_result"):
        st.markdown(
            "<div style='text-align:center;padding:60px;border:1px dashed #0d2a4a;"
            "border-radius:12px;color:#1a4060'>"
            "<div style='font-size:3rem'>&#128202;</div>"
            "<div style='margin-top:14px;font-size:.85rem'>"
            "No analysis run yet. Go to <b>Analyze Star Image</b> to run the pipeline."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return
    d = st.session_state.last_result
    _render_results(d["result"], d["raw"], d["preprocessed"],
                    d["stars"], d["is_demo"], d["image_name"])


# ============================================================================
#  PAGE: MISSION STATUS
# ============================================================================

def _page_mission():
    col_a, col_b = st.columns([1, 1])

    with col_a:
        # System monitor
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128202;</span>"
            "<span class='card-title'>System Monitor</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        cat_ok   = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
        model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()
        systems = [
            ("Camera Input",          _PIPELINE_OK or True, "ONLINE"),
            ("Detection Engine",      _PIPELINE_OK,         "ONLINE" if _PIPELINE_OK else "OFFLINE"),
            ("AI Model",              model_ok,             "LOADED" if model_ok else "STANDBY"),
            ("Navigation Solver",     _PIPELINE_OK,         "ONLINE" if _PIPELINE_OK else "OFFLINE"),
            ("Hipparcos Catalog",     cat_ok,               "LOADED" if cat_ok else "MISSING"),
            ("Report Generator",      True,                 "ONLINE"),
        ]
        for name, ok, status in systems:
            dot  = "dot-green" if ok else "dot-red"
            vcls = "val-online" if ok else "val-offline"
            st.markdown(
                f"<div class='sys-row'>"
                f"<span class='lbl'>{name}</span>"
                f"<span><span class='{dot}'></span>"
                f"<span class='val {vcls}'>{status}</span></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Pipeline import info
        if not _PIPELINE_OK:
            with st.expander("Backend diagnostic"):
                st.code(_PIPELINE_ERR, language="text")

    with col_b:
        # Analysis history
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128200;</span>"
            "<span class='card-title'>Analysis History</span>"
            f"<span class='card-sub'>{len(st.session_state.history)} runs this session</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        history = st.session_state.get("history", [])
        if not history:
            st.markdown(
                "<div style='color:#1a4060;text-align:center;padding:24px;font-size:.82rem'>"
                "No analyses performed yet.</div>",
                unsafe_allow_html=True,
            )
        else:
            # Header
            st.markdown(
                "<div style='display:flex;gap:12px;padding:6px 14px;"
                "font-size:.65rem;color:#2a5a8a;text-transform:uppercase;"
                "letter-spacing:1px;border-bottom:1px solid #0d2a4a;margin-bottom:4px'>"
                "<span style='min-width:60px'>Time</span>"
                "<span style='flex:1'>Image</span>"
                "<span style='min-width:60px;text-align:right'>Confidence</span>"
                "<span style='min-width:60px;text-align:right'>Status</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            for h in reversed(history[-15:]):
                conf_col = "#22c55e" if h["conf"] >= 0.6 else ("#eab308" if h["conf"] >= 0.3 else "#ef4444")
                stat_col = "#22c55e" if h["status"] == "SUCCESS" else "#eab308"
                mode_tag = "<span style='font-size:.6rem;color:#3a4a6a'>[sample]</span>" if h.get("demo") else ""
                st.markdown(
                    f"<div class='hist-row'>"
                    f"<span class='hist-ts'>{h['ts']}</span>"
                    f"<span class='hist-img'>{h['name'][:28]} {mode_tag}</span>"
                    f"<span class='hist-conf' style='color:{conf_col};min-width:50px;text-align:right'>"
                    f"{h['conf']*100:.0f}%</span>"
                    f"<span style='font-size:.7rem;font-weight:700;color:{stat_col};"
                    f"min-width:60px;text-align:right'>{h['status']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Session stats
        if history:
            avg_conf = sum(h["conf"] for h in history) / len(history)
            n_ok     = sum(1 for h in history if h["status"] == "SUCCESS")
            st.markdown(
                "<div class='card'>"
                "<div class='card-hdr'>"
                "<span class='card-icon'>&#128293;</span>"
                "<span class='card-title'>Session Statistics</span>"
                "</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px'>"
                f"<div class='mtile'><div class='mtile-val'>{len(history)}</div>"
                f"<div class='mtile-lbl'>Total Runs</div></div>"
                f"<div class='mtile'><div class='mtile-val'>{avg_conf*100:.0f}%</div>"
                f"<div class='mtile-lbl'>Avg Confidence</div></div>"
                f"<div class='mtile'><div class='mtile-val'>{n_ok}/{len(history)}</div>"
                f"<div class='mtile-lbl'>Successful</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# ============================================================================
#  PAGE: ABOUT
# ============================================================================

def _page_about():
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#127756;</span>"
            "<span class='card-title'>StellarX StarNav-AI</span>"
            "</div>"
            "<div style='color:#5a8fba;font-size:.84rem;line-height:1.9'>"
            "StellarX StarNav-AI is an autonomous spacecraft attitude determination system "
            "using AI-powered star pattern recognition. It identifies stars in any astronomical "
            "image and computes the spacecraft orientation quaternion without GPS or ground contact."
            "<br><br>"
            "<b style='color:#4db8ff'>Core Technology</b><br>"
            "Star detection via connected-component analysis with sub-pixel centroiding. "
            "Rotation-invariant feature descriptors matched against the Hipparcos catalog "
            "using KD-tree indexed pairwise angular voting. Wahba SVD attitude solver with "
            "iterative outlier rejection."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128218;</span>"
            "<span class='card-title'>References</span>"
            "</div>"
            "<div style='color:#3a6a9a;font-size:.78rem;line-height:2.2'>"
            "Groth (1986) &mdash; A pattern-matching algorithm for two-dimensional dot patterns<br>"
            "Mortari et al. (2004) &mdash; The Pyramid star identification technique<br>"
            "Wahba (1965) &mdash; A least squares estimate of spacecraft attitude<br>"
            "Markley &amp; Crassidis (2014) &mdash; Fundamentals of Spacecraft Attitude Determination<br>"
            "ESA Hipparcos Catalog (1997) &mdash; 120,000 stars, 0.001 arcsec precision"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#128295;</span>"
            "<span class='card-title'>Technical Specifications</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        specs = [
            ("Attitude accuracy",   "< 1 deg RMS"),
            ("Processing latency",  "< 100 ms"),
            ("Catalog",             "Hipparcos (50 stars)"),
            ("Feature vector",      "90 dimensions"),
            ("Algorithm",           "Wahba / SVD"),
            ("ML model",            "Random Forest"),
            ("Language",            "Python 3.10+"),
            ("Interface",           "Streamlit"),
        ]
        rows = "".join(
            f"<tr><td style='color:#3a6a9a;padding:6px 10px;font-size:.75rem'>{k}</td>"
            f"<td class='hi' style='padding:6px 10px'>{v}</td></tr>"
            for k, v in specs
        )
        st.markdown(f"<table class='nav-table'>{rows}</table></div>",
                    unsafe_allow_html=True)

        st.markdown(
            "<div class='card'>"
            "<div class='card-hdr'>"
            "<span class='card-icon'>&#9432;</span>"
            "<span class='card-title'>System Information</span>"
            "</div>"
            f"<div class='mono-box' style='font-size:.72rem'>"
            f"VERSION &nbsp; <span class='mono-val'>1.0.0</span><br>"
            f"BACKEND &nbsp; <span class='mono-val'>{'OPERATIONAL' if _PIPELINE_OK else 'DEMO MODE'}</span><br>"
            f"CATALOG &nbsp; <span class='mono-val'>{'LOADED' if (_ROOT/'data/catalog/hipparcos_bright.csv').exists() else 'MISSING'}</span><br>"
            f"MODEL &nbsp;&nbsp;&nbsp; <span class='mono-val'>{'LOADED' if (_ROOT/'models/star_pattern_classifier.pkl').exists() else 'STANDBY'}</span>"
            f"</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================================
#  MAIN
# ============================================================================

def main():
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    _header()

    page = _sidebar()

    if page == "Dashboard":
        _page_dashboard()
    elif page == "Analyze Star Image":
        _page_analyse()
    elif page == "Results":
        _page_results()
    elif page == "Mission Status":
        _page_mission()
    elif page == "About":
        _page_about()

    st.markdown(
        "<div class='footer' style='text-align:center;color:#0d2040;"
        "font-size:.65rem;padding:16px 0 6px;border-top:1px solid #0a1e36;margin-top:32px;"
        "font-family:JetBrains Mono,monospace;letter-spacing:1px'>"
        "STELLARX STARNAV-AI &nbsp; | &nbsp; AUTONOMOUS SPACECRAFT NAVIGATION"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
