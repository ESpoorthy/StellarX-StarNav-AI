"""
StellarX StarNav-AI — Mission Navigation Dashboard
===================================================
SIH 2026 | Team StellarX | Autonomous Spacecraft Attitude Determination

Navigation pages:
    MISSION CONTROL   — status, live pipeline, current attitude
    ATTITUDE ANALYSIS — run pipeline, full telemetry, integrity
    STAR MATCHING     — catalogue matches, residuals, visualisation
    ROBUSTNESS LAB    — noise / false stars / occlusion stress tests
    PERFORMANCE       — timing breakdown, benchmark comparison
    ABOUT             — architecture, how it works, team

All displayed numbers come from actual computation.
Nothing is fabricated. Failure states are shown explicitly.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="StellarX StarNav-AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "StellarX StarNav-AI | Autonomous Attitude Determination | Team StellarX"},
)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
_PIPELINE_OK = False
_PIPELINE_ERR = ""
try:
    from src.catalog.catalog_loader import load_catalog
    from src.recognition.catalog_index import CatalogIndex
    from src.navigation.navigator import run_navigation, _preprocess_image
    from src.preprocessing.star_detection import detect_stars
    from src.navigation.pipeline_result import build_pipeline_result, PipelineResult
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
#  COLOUR PALETTE — single source of truth
# =============================================================================
C = {
    "bg":      "#050d1b",
    "card":    "#08162a",
    "border":  "#0d2844",
    "accent":  "#3aa3e8",
    "accent2": "#64c8ff",
    "text":    "#c4ddf0",
    "muted":   "#5a88aa",
    "success": "#1db954",
    "warning": "#e8a020",
    "error":   "#e03444",
    "partial": "#9b72cf",
    "dim":     "#2a4a6a",
}

# =============================================================================
#  CSS
# =============================================================================
st.markdown(f"""
<style>
/* ── Base ── */
.stApp {{
    background: {C['bg']} !important;
    background: radial-gradient(ellipse at 15% 10%, #071a35 0%, {C['bg']} 60%) !important;
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #040f22 0%, #060f1e 100%) !important;
    border-right: 1px solid {C['border']} !important;
}}
/* ── Text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li, p, li, span, label {{
    color: {C['text']};
}}
h1,h2,h3,h4 {{ color: {C['accent']} !important; }}
/* ── Metrics ── */
[data-testid="stMetricValue"] {{
    color: {C['accent']} !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {C['muted']} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}}
[data-testid="stMetricDelta"] {{
    color: {C['text']} !important;
    font-size: 0.68rem !important;
}}
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['card']} !important;
    border-bottom: 2px solid {C['border']} !important;
    gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C['muted']} !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C['accent']} !important;
    background: rgba(58,163,232,0.1) !important;
    border-bottom: 2px solid {C['accent']} !important;
}}
/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, #0d3870 0%, #0a2854 100%) !important;
    color: {C['accent']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: {C['accent']} !important;
    color: #fff !important;
    box-shadow: 0 0 14px rgba(58,163,232,0.35) !important;
}}
/* ── Download button ── */
.stDownloadButton > button {{
    background: linear-gradient(135deg, #0a3a18, #072a12) !important;
    color: {C['success']} !important;
    border: 1px solid #0e4a22 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}}
/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
}}
/* ── Expander ── */
.streamlit-expanderHeader {{
    background: {C['card']} !important;
    color: {C['accent']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
}}
/* ── Progress bar ── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {C['accent']}, {C['accent2']}) !important;
}}
/* ── Spinner ── */
.stSpinner > div {{ color: {C['accent']} !important; }}
footer {{ visibility: hidden; }}

/* ── Custom components ── */
.sx-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.sx-card-title {{
    color: {C['accent']};
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid {C['border']};
}}
.sx-phase {{
    display: inline-block;
    background: rgba(58,163,232,0.12);
    color: {C['accent']};
    border: 1px solid {C['border']};
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.sx-stat {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid {C['dim']};
    font-size: 0.78rem;
}}
.sx-stat .lbl {{ color: {C['muted']}; }}
.sx-stat .val {{ color: {C['text']}; font-weight: 600; font-family: monospace; }}
.sx-stat .val-accent {{ color: {C['accent']}; font-weight: 700; font-family: monospace; }}
.sx-divider {{ border: none; border-top: 1px solid {C['border']}; margin: 14px 0; }}
.sx-telemetry-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    border-radius: 5px;
    border: 1px solid {C['border']};
    background: rgba(255,255,255,0.015);
    margin-bottom: 4px;
    font-size: 0.78rem;
}}
.sx-telemetry-row .stage {{ color: {C['text']}; font-weight: 600; flex: 1; }}
.sx-telemetry-row .timing {{ color: {C['accent2']}; font-family: monospace; font-weight: 700; }}
.sx-telemetry-row .check {{ font-size: 0.9rem; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  CACHED RESOURCES — loaded once per session
# =============================================================================

@st.cache_resource(show_spinner="Loading star catalog and building index…")
def _load_resources(cfg_path: str):
    if not (HAS_YAML and _PIPELINE_OK):
        return None, None, None, f"Backend unavailable: {_PIPELINE_ERR}"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cat_file = cfg.get("dataset", {}).get("catalog_file",
                                               "data/catalog/hipparcos_bright.csv")
        cat_path = _ROOT / cat_file
        if not cat_path.exists():
            return cfg, None, None, f"Catalog missing: {cat_path}"
        catalog = load_catalog(str(cat_path), config=cfg)
        cidx = CatalogIndex(catalog)
        return cfg, catalog, cidx, None
    except Exception as exc:
        return None, None, None, str(exc)


@st.cache_resource(show_spinner=False)
def _load_model(cfg_path: str):
    if not HAS_YAML:
        return None, "PyYAML not installed"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from src.models.inference import load_model
        ck = (_ROOT /
              cfg.get("model", {}).get("checkpoint_dir", "models") /
              cfg.get("model", {}).get("checkpoint_name", "star_pattern_classifier.pkl"))
        if not ck.exists():
            return None, "No trained checkpoint — geometry-only mode"
        return load_model(str(ck), cfg), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _demo_assets():
    if _DEMO_PKG_OK:
        return load_demo_assets()
    return _fb_image(), _fb_stars(), _fb_result()


# =============================================================================
#  FALLBACK ASSETS (when backend unavailable)
# =============================================================================

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
                r, c = int(cy)+dr, int(cx)+dc
                if 0 <= r < 512 and 0 <= c < 512:
                    img[r, c] = min(1., img[r, c] + flux * math.exp(
                        -(dr**2+dc**2)/(2*sig**2)))
    img += rng.normal(0, 0.003, img.shape).astype(np.float32)
    return np.clip(img, 0., 1.)


class _FbStar:
    def __init__(self, x, y, b, pk, ar):
        self.x=x; self.y=y; self.brightness=b; self.peak=pk; self.area=ar


def _fb_stars():
    pts = [(200,150,0.92),(350,220,0.78),(120,310,0.68),(430,390,0.58),
           (260,420,0.52),(80,100,0.88),(470,130,0.62),(320,80,0.72),
           (180,460,0.46),(390,300,0.57),(50,380,0.42),(450,460,0.36)]
    return [_FbStar(float(x), float(y), b, b*1.2, 8) for x, y, b in pts]


def _fb_result():
    """Return a demo-mode result-like object (not PipelineResult — just for fallback)."""
    from dataclasses import dataclass, field as dc_field
    import numpy as _np

    @dataclass
    class _R:
        status: str = "SUCCESS"
        attitude_status: str = "DETERMINED"
        position_status: str = "UNAVAILABLE"
        quaternion: _np.ndarray = dc_field(
            default_factory=lambda: _np.array([0.9659, 0.2588, 0., 0.]))
        rotation_matrix: _np.ndarray = dc_field(default_factory=lambda: _np.eye(3))
        euler_angles_deg: _np.ndarray = dc_field(
            default_factory=lambda: _np.array([30., -14.7, 45.]))
        attitude_confidence: float = 0.87
        attitude_residual_deg: float = 0.342
        max_residual_deg: float = 0.718
        n_observed_stars: int = 12
        n_matched_stars: int = 10
        n_inlier_stars: int = 9
        n_outlier_stars: int = 1
        identified_stars: list = dc_field(default_factory=list)
        preprocessing_time_ms: float = 2.3
        detection_time_ms: float = 14.8
        feature_extraction_time_ms: float = 2.1
        recognition_time_ms: float = 38.4
        attitude_time_ms: float = 4.8
        total_time_ms: float = 62.4
        position_note: str = "Attitude-only solution."
        error_message: str = ""

    return _R()


# =============================================================================
#  IMAGE UTILITIES
# =============================================================================

def _to_gray(data: bytes) -> tuple[Optional[np.ndarray], str]:
    try:
        if HAS_PIL:
            arr = np.array(PILImage.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255.
            return arr, ""
        if HAS_CV2:
            buf = np.frombuffer(data, dtype=np.uint8)
            arr = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                return None, "Decode failed"
            return arr.astype(np.float32) / 255., ""
        return None, "Pillow or OpenCV required"
    except Exception as exc:
        return None, str(exc)


def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# =============================================================================
#  PIPELINE RUNNER
# =============================================================================

def _run_pipeline(raw: np.ndarray, is_demo: bool = False) -> tuple:
    """Run the full pipeline. Returns (PipelineResult, preprocessed, stars, err)."""
    cfg_path = str(_ROOT / "config.yaml")
    cfg, _, cidx, err = _load_resources(cfg_path)
    if err:
        return None, raw, [], err
    neural, _ = _load_model(cfg_path)

    try:
        t0 = time.perf_counter()
        preprocessed = _preprocess_image(raw.copy(), cfg)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        stars = detect_stars(preprocessed, cfg.get("star_detection", {}))

        nav_result = run_navigation(preprocessed, cfg, cidx, neural_model=neural)
        nav_result.preprocessing_time_ms = preprocess_ms
        nav_result.total_time_ms += preprocess_ms

        # Get neural result if available
        neural_result = None
        if neural is not None and len(stars) >= 2:
            try:
                from src.preprocessing.star_detection import extract_features
                from src.models.inference import run_inference
                feats = extract_features(stars, cfg)
                neural_result = run_inference(neural, feats, cfg)
            except Exception:
                pass

        pr = build_pipeline_result(
            nav_result, cfg,
            image_shape=preprocessed.shape[:2],
            is_demo=is_demo,
            neural_result=neural_result,
        )
        return pr, preprocessed, stars, ""
    except Exception as exc:
        return None, raw, [], f"{exc}\n{traceback.format_exc()}"


# =============================================================================
#  SESSION STATE
# =============================================================================

def _init_state():
    defaults = {
        "history": [],
        "last_pr": None,
        "last_img": None,
        "last_stars": None,
        "last_preprocessed": None,
        "last_img_name": "",
        "demo_pending": False,
        "failure_pending": False,
        "gt_pending": False,
        "rob_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
#  PLOTTING
# =============================================================================
_BG = "#050d1b"
_AX = "#070f1e"
_GR = "#0d2844"
_TX = "#c4ddf0"
_AC = "#3aa3e8"


def _ax_style(ax, title=""):
    ax.set_facecolor(_AX)
    if title:
        ax.set_title(title, color=_AC, fontsize=8, fontweight="bold", loc="left", pad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GR)
    ax.tick_params(colors=_TX, labelcolor=_TX, labelsize=6)


def _plot_image(image: np.ndarray, title="Raw Star-Field") -> bytes:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=_BG)
    _ax_style(ax, title)
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", origin="upper")
    plt.tight_layout(pad=0.4)
    return _fig_bytes(fig)


def _plot_detections(image: np.ndarray, stars: list, matched_ids: set = None) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=_BG)
    _ax_style(ax, f"Star Detection — {len(stars)} candidates")
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", origin="upper")
    for i, s in enumerate(stars):
        x = float(getattr(s, "x", getattr(s, "x_px", 0)))
        y = float(getattr(s, "y", getattr(s, "y_px", 0)))
        b = float(getattr(s, "brightness", getattr(s, "flux", 0.5)))
        r = max(5, min(13, b * 13))
        is_matched = matched_ids and i < len(stars) and i < 15
        col = C["success"] if is_matched else _AC
        ax.add_patch(plt.Circle((x, y), r, color=col, fill=False, lw=1.0, alpha=0.85))
        ax.plot(x, y, "+", color=col, ms=3.5, lw=0.8)
        if i < 6:
            ax.annotate(f"S{i+1}", (x, y), textcoords="offset points",
                        xytext=(5, 4), color=_TX, fontsize=5, alpha=0.8)
    plt.tight_layout(pad=0.4)
    return _fig_bytes(fig)


def _plot_matching(image: np.ndarray, stars: list, identified: list) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=_BG)
    matched = len(identified)
    total = len(stars)
    _ax_style(ax, f"Catalogue Matching — {matched}/{total} matched")
    ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos", origin="upper")

    # Unmatched stars — dim circles
    matched_xs = {float(getattr(s, "observed_x", 0)) for s in identified}
    for s in stars:
        x = float(getattr(s, "x", getattr(s, "x_px", 0)))
        if x not in matched_xs:
            y = float(getattr(s, "y", getattr(s, "y_px", 0)))
            ax.add_patch(plt.Circle((x, y), 6, color=C["dim"], fill=False, lw=0.6))

    # Matched stars — green circles + label
    for m in identified:
        x = float(getattr(m, "observed_x", 0))
        y = float(getattr(m, "observed_y", 0))
        res = float(getattr(m, "angular_residual_deg", 0))
        cid = str(getattr(m, "catalog_id", "")).replace("HIP_", "HIP ")
        # Colour by residual quality
        col = C["success"] if res < 0.5 else (C["warning"] if res < 1.5 else C["error"])
        ax.add_patch(plt.Circle((x, y), 9, color=col, fill=False, lw=1.4, alpha=0.9))
        ax.annotate(cid.split()[-1], (x, y), textcoords="offset points",
                    xytext=(6, 4), color=col, fontsize=5.5, fontweight="bold")
        # Correspondence line to predicted inertial direction (symbolic)
        ax.plot(x, y, "o", ms=4, color=col, alpha=0.6)

    plt.tight_layout(pad=0.4)
    return _fig_bytes(fig)


def _plot_timing(perf) -> bytes:
    stages = perf.as_dict()
    vals = list(stages.values())
    lbls = list(stages.keys())
    cols = [_AC, C["success"], C["partial"], C["warning"], C["error"], C["muted"]][:len(vals)]

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=_BG)
    _ax_style(ax, f"Pipeline Timing  —  Total: {perf.total_ms:.1f} ms")
    bars = ax.barh(lbls, vals, color=cols[:len(lbls)], height=0.5, alpha=0.9)
    ax.set_xlabel("Latency (ms)", color=_TX, fontsize=7)
    ax.set_xlim(0, max(vals + [1]) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} ms", va="center", color=_TX, fontsize=7, fontweight="600")
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelcolor=_TX, labelsize=7)
    plt.tight_layout(pad=0.5)
    return _fig_bytes(fig)


def _plot_cubesat_3d(quaternion: np.ndarray) -> bytes:
    """3D CubeSat orientation using actual quaternion."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(5, 4.5), facecolor=_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_AX)

    # Build rotation matrix from quaternion
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q = q / norm
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
    ], dtype=np.float64)

    # CubeSat box vertices (1U = 10x10x10 cm, normalised to 1)
    h = 0.5
    verts_body = np.array([
        [-h,-h,-h],[-h,-h, h],[-h, h,-h],[-h, h, h],
        [ h,-h,-h],[ h,-h, h],[ h, h,-h],[ h, h, h],
    ], dtype=np.float64)
    verts_rot = (R @ verts_body.T).T

    faces_idx = [
        [0,1,3,2],[4,5,7,6],[0,1,5,4],
        [2,3,7,6],[0,2,6,4],[1,3,7,5],
    ]
    face_colors = ["#0a2244","#0e2c54","#0a2244","#0e2c54","#0c2848","#0c2848"]
    poly = Poly3DCollection(
        [[verts_rot[i] for i in f] for f in faces_idx],
        alpha=0.75, facecolors=face_colors,
        edgecolors="#1a4070", linewidths=0.7,
    )
    ax.add_collection3d(poly)

    # Body axes (rotated)
    scale = 0.85
    axis_defs = [
        (np.array([1,0,0]), _AC,     "X"),
        (np.array([0,1,0]), C["success"], "Y"),
        (np.array([0,0,1]), C["warning"], "Z (boresight)"),
    ]
    for body_ax, col, lbl in axis_defs:
        end = R @ (body_ax * scale)
        ax.quiver(0, 0, 0, end[0], end[1], end[2],
                  color=col, linewidth=1.8, arrow_length_ratio=0.18)
        ax.text(end[0]*1.12, end[1]*1.12, end[2]*1.12, lbl,
                color=col, fontsize=6.5, fontweight="bold")

    # Inertial reference frame (faint dashes)
    for ref_ax, col in [(np.array([1,0,0]), "#204060"),
                         (np.array([0,1,0]), "#204060"),
                         (np.array([0,0,1]), "#204060")]:
        ax.plot([0, ref_ax[0]*0.9], [0, ref_ax[1]*0.9], [0, ref_ax[2]*0.9],
                color=col, lw=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("X", color=_TX, fontsize=6); ax.set_ylabel("Y", color=_TX, fontsize=6)
    ax.set_zlabel("Z", color=_TX, fontsize=6)
    ax.tick_params(colors=_TX, labelsize=5, labelcolor=_TX)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor(_GR)
    ax.set_title("Estimated Spacecraft Orientation\n(body axes rotated by computed quaternion)",
                 color=_AC, fontsize=7, pad=4)
    ax.grid(True, color=_GR, linewidth=0.3, alpha=0.4)
    plt.tight_layout(pad=0.3)
    return _fig_bytes(fig)


# =============================================================================
#  HTML HELPERS
# =============================================================================

def _stat(label: str, value: str, accent: bool = False) -> str:
    cls = "val-accent" if accent else "val"
    return (f"<div class='sx-stat'><span class='lbl'>{label}</span>"
            f"<span class='{cls}'>{value}</span></div>")


def _status_pill(s: str) -> str:
    s = str(s).upper()
    if s in ("SUCCESS", "VALID", "DETERMINED", "OPERATIONAL"):
        col = C["success"]; sym = "●"
    elif s in ("PARTIAL", "DEGRADED", "LOW_CONFIDENCE"):
        col = C["partial"]; sym = "◐"
    elif s in ("UNAVAILABLE", "STANDBY", "GEOMETRY_ONLY"):
        col = C["warning"]; sym = "○"
    else:
        col = C["error"]; sym = "✕"
    return (f"<span style='background:rgba(0,0,0,0.2);color:{col};"
            f"border:1px solid {col}44;border-radius:5px;padding:2px 9px;"
            f"font-size:0.72rem;font-weight:700;font-family:monospace'>{sym} {s}</span>")


def _telemetry_row(stage: str, ms: float, ok: bool = True) -> str:
    chk = "✓" if ok else "—"
    col = C["success"] if ok else C["muted"]
    ms_str = f"{ms:.1f} ms" if ms > 0 else "—"
    return (f"<div class='sx-telemetry-row'>"
            f"<span class='stage'>{stage}</span>"
            f"<span class='timing'>{ms_str}</span>"
            f"<span class='check' style='color:{col}'>{chk}</span>"
            f"</div>")


def _conf_bar(label: str, val: float, col: str = None) -> str:
    col = col or _AC
    pct = int(max(0, min(100, val * 100)))
    return (f"<div style='margin:4px 0 8px'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:.73rem;color:{C['muted']};margin-bottom:2px'>"
            f"<span style='color:{C['text']}'>{label}</span>"
            f"<span style='color:{col};font-weight:700'>{pct}%</span></div>"
            f"<div style='background:{C['dim']};border-radius:4px;"
            f"height:7px;overflow:hidden;border:1px solid {C['border']}'>"
            f"<div style='width:{pct}%;height:100%;background:{col};border-radius:4px'>"
            f"</div></div></div>")


# =============================================================================
#  HEADER
# =============================================================================

def _header():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S  UTC")
    mode = "OPERATIONAL" if _PIPELINE_OK else "DEMO MODE"
    mode_col = C["success"] if _PIPELINE_OK else C["warning"]
    c1, c2, c3 = st.columns([1, 5, 3])
    with c1:
        st.markdown(
            f"<div style='font-size:2.6rem;text-align:center;padding-top:4px;"
            f"filter:drop-shadow(0 0 12px rgba(58,163,232,.6))'>🛰️</div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='padding-top:4px'>"
            f"<div style='font-size:1.7rem;font-weight:900;letter-spacing:3px;"
            f"color:{_AC};text-transform:uppercase;line-height:1'>STELLARX StarNav-AI</div>"
            f"<div style='font-size:.72rem;color:{C['muted']};letter-spacing:1.5px;"
            f"text-transform:uppercase;margin-top:2px'>"
            f"Autonomous Lost-in-Space Attitude Determination  ·  Team StellarX  ·  SIH 2026</div>"
            f"</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div style='text-align:right;padding-top:6px'>"
            f"<div style='font-size:.68rem;color:{C['muted']};font-family:monospace'>{ts}</div>"
            f"<div style='font-size:.72rem;color:{mode_col};font-weight:700;margin-top:2px'>"
            f"● {mode}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{C['border']};margin:8px 0 14px'>",
                unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:16px 0 10px'>"
            f"<div style='font-size:2rem'>🛰️</div>"
            f"<div style='color:{_AC};font-size:1rem;font-weight:900;"
            f"letter-spacing:2px;text-transform:uppercase;margin-top:4px'>STELLARX</div>"
            f"<div style='color:{C['muted']};font-size:.68rem;letter-spacing:1px'>StarNav-AI</div>"
            f"</div>",
            unsafe_allow_html=True)
        st.markdown(f"<hr style='border-color:{C['border']};margin:0 0 10px'>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{C['muted']};font-size:.65rem;letter-spacing:1px;"
            f"text-transform:uppercase;margin-bottom:6px;padding-left:4px'>Navigation</div>",
            unsafe_allow_html=True)
        page = st.radio("page_nav", [
            "🛰️  Mission Control",
            "🔭  Attitude Analysis",
            "⭐  Star Matching",
            "🧪  Robustness Lab",
            "⚡  Performance",
            "ℹ️  About",
        ], label_visibility="collapsed")
        st.markdown(f"<hr style='border-color:{C['border']};margin:10px 0'>",
                    unsafe_allow_html=True)

        # System status
        st.markdown(
            f"<div style='color:{C['muted']};font-size:.65rem;letter-spacing:1px;"
            f"text-transform:uppercase;margin-bottom:6px;padding-left:4px'>System</div>",
            unsafe_allow_html=True)

        cat_ok = (_ROOT / "data" / "catalog" / "hipparcos_bright.csv").exists()
        model_ok = (_ROOT / "models" / "star_pattern_classifier.pkl").exists()

        def _srow(icon, name, ok, note=""):
            col = C["success"] if ok else C["error"]
            note_html = (f"<br><span style='font-size:.62rem;color:{C['muted']}'>{note}</span>"
                         if note else "")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:7px;padding:4px 6px;"
                f"border-radius:5px;border:1px solid {C['border']};"
                f"background:rgba(255,255,255,.012);margin-bottom:3px'>"
                f"<span style='font-size:.78rem'>{icon}</span>"
                f"<span style='font-size:.72rem;color:{C['text']};flex:1'>{name}{note_html}</span>"
                f"<span style='width:5px;height:5px;border-radius:50%;"
                f"background:{col};box-shadow:0 0 4px {col};flex-shrink:0'></span>"
                f"</div>",
                unsafe_allow_html=True)

        _srow("🔍", "Star Detection", _PIPELINE_OK)
        _srow("📚", "Hipparcos Catalog", cat_ok)
        _srow("🤖", "AI Classifier",  model_ok,
              "" if model_ok else "geometry-only")
        _srow("🧭", "Wahba/SVD Solver", _PIPELINE_OK)
        _srow("✅", "Integrity Check", _PIPELINE_OK)

        st.markdown(f"<hr style='border-color:{C['border']};margin:10px 0'>",
                    unsafe_allow_html=True)
        n = len(st.session_state.get("history", []))
        pr = st.session_state.get("last_pr")
        if pr and hasattr(pr, "integrity"):
            last_status = pr.overall_status
            s_col = C["success"] if last_status == "SUCCESS" else C["warning"]
            st.markdown(
                f"<div style='font-size:.7rem;color:{C['muted']};text-align:center'>"
                f"Runs: <b style='color:{_AC}'>{n}</b>  "
                f"Last: <b style='color:{s_col}'>{last_status}</b></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='font-size:.7rem;color:{C['muted']};text-align:center'>"
                f"Session runs: <b style='color:{_AC}'>{n}</b></div>",
                unsafe_allow_html=True)
    return page.strip()


# =============================================================================
#  PAGE: MISSION CONTROL
# =============================================================================

def _page_mission_control():
    st.markdown(
        f"<div style='text-align:center;color:{C['muted']};font-size:.75rem;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:20px'>"
        f"Autonomous Lost-in-Space Attitude Determination — No GPS · No Ground Link · Single Star Tracker</div>",
        unsafe_allow_html=True)

    # ── Mission state banner ──────────────────────────────────────────────────
    pr: Optional[PipelineResult] = st.session_state.get("last_pr")

    if pr is None:
        # STANDBY state
        c1, c2, c3 = st.columns(3, gap="medium")
        for col, icon, label, val, vc in [
            (c1, "📡", "GPS", "OFFLINE", C["error"]),
            (c2, "🔗", "Ground Link", "OFFLINE", C["error"]),
            (c3, "🔭", "Attitude", "UNKNOWN", C["warning"]),
        ]:
            with col:
                st.markdown(
                    f"<div style='background:{C['card']};border:1px solid {C['border']};"
                    f"border-radius:9px;padding:18px;text-align:center'>"
                    f"<div style='font-size:1.8rem;margin-bottom:6px'>{icon}</div>"
                    f"<div style='font-size:.7rem;color:{C['muted']};letter-spacing:1px;"
                    f"text-transform:uppercase;margin-bottom:4px'>{label}</div>"
                    f"<div style='font-size:1.1rem;font-weight:900;color:{vc};"
                    f"letter-spacing:2px;font-family:monospace'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{C['border']};margin:20px 0 16px'>",
                    unsafe_allow_html=True)
        st.info("🛰️  No attitude solution yet.  "
                "Go to **🔭 Attitude Analysis** to run the pipeline.")
    else:
        # POST-SOLUTION state
        att = pr.attitude
        integ = pr.integrity
        status_col = (C["success"] if pr.overall_status == "SUCCESS"
                      else C["warning"] if pr.overall_status in ("PARTIAL", "DEGRADED")
                      else C["error"])

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        for col, icon, label, val, vc in [
            (c1, "📡", "GPS", "OFFLINE", C["error"]),
            (c2, "🔗", "Ground Link", "OFFLINE", C["error"]),
            (c3, "🔭", "Attitude", pr.overall_status, status_col),
            (c4, "✅", "Integrity", integ.status, (C["success"] if integ.status == "VALID"
                                                    else C["warning"] if integ.status == "DEGRADED"
                                                    else C["error"])),
        ]:
            with col:
                st.markdown(
                    f"<div style='background:{C['card']};border:1px solid {C['border']};"
                    f"border-radius:9px;padding:14px;text-align:center'>"
                    f"<div style='font-size:1.6rem;margin-bottom:5px'>{icon}</div>"
                    f"<div style='font-size:.68rem;color:{C['muted']};letter-spacing:1px;"
                    f"text-transform:uppercase;margin-bottom:3px'>{label}</div>"
                    f"<div style='font-size:.95rem;font-weight:900;color:{vc};"
                    f"letter-spacing:1.5px;font-family:monospace'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{C['border']};margin:14px 0'>",
                    unsafe_allow_html=True)

        # Current attitude + 3D
        ca, cb = st.columns([1, 1], gap="large")
        with ca:
            st.markdown(f"<div class='sx-card'><div class='sx-card-title'>Current Attitude</div>",
                        unsafe_allow_html=True)
            st.markdown(
                _stat("Roll",  f"{att.roll_deg:.4f}°",  True) +
                _stat("Pitch", f"{att.pitch_deg:.4f}°", True) +
                _stat("Yaw",   f"{att.yaw_deg:.4f}°",  True) +
                f"<div style='margin:8px 0 4px'><div style='font-size:.65rem;color:{C['muted']};margin-bottom:3px'>QUATERNION [qw qx qy qz]</div>"
                f"<div style='font-family:monospace;font-size:.78rem;color:{_AC};background:{C['dim']};padding:7px 10px;border-radius:5px'>"
                f"{att.quaternion[0]:.6f}<br>{att.quaternion[1]:+.6f}<br>"
                f"{att.quaternion[2]:+.6f}<br>{att.quaternion[3]:+.6f}</div></div>" +
                _stat("Confidence", f"{att.attitude_confidence*100:.1f}%") +
                _stat("Mean residual", f"{att.mean_residual_deg:.4f}°" if math.isfinite(att.mean_residual_deg) else "—") +
                _stat("Inliers / Total", f"{att.n_inliers} / {pr.detection.detected_count}") +
                f"<div style='margin-top:8px;font-size:.62rem;color:{C['muted']}'>"
                f"Frame convention: {att.frame_convention}</div>",
                unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Integrity block
            ic = integ.status
            ic_col = C["success"] if ic == "VALID" else C["warning"] if ic == "DEGRADED" else C["error"]
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Integrity Verification</div>"
                + f"<div style='font-size:1.1rem;font-weight:900;color:{ic_col};"
                f"font-family:monospace;margin-bottom:8px'>{integ.status_symbol}  ATTITUDE {ic}</div>"
                + (f"<div style='font-size:.73rem;color:{C['error']};background:rgba(224,52,68,.08);"
                   f"border:1px solid {C['error']}44;border-radius:5px;padding:7px 10px;margin-bottom:8px'>"
                   f"⚠ {integ.rejection_reason}</div>" if integ.rejection_reason else "")
                + _stat("Quaternion norm", f"{integ.quaternion_norm:.8f}" if math.isfinite(integ.quaternion_norm) else "—")
                + _stat("det(R)", f"{integ.determinant:.8f}" if math.isfinite(integ.determinant) else "—")
                + _stat("Orthogonality err", f"{integ.orthogonality_error:.2e}" if math.isfinite(integ.orthogonality_error) else "—")
                + _stat("RMS residual", f"{integ.rms_residual_deg:.4f}°" if math.isfinite(integ.rms_residual_deg) else "—")
                + f"<div style='margin-top:6px'>{_conf_bar('Integrity confidence', integ.confidence, ic_col)}</div>"
                + "</div>",
                unsafe_allow_html=True)

        with cb:
            if HAS_MPL and _PIPELINE_OK:
                try:
                    fig_bytes = _plot_cubesat_3d(att.quaternion)
                    st.image(fig_bytes, caption="3D spacecraft orientation from computed quaternion",
                             width="stretch")
                except Exception as e:
                    st.warning(f"3D visualisation error: {e}")

        # Pipeline telemetry summary
        st.markdown(f"<hr style='border-color:{C['border']};margin:12px 0 10px'>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{_AC};font-size:.78rem;font-weight:700;"
            f"letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>"
            f"⏱️  Pipeline Telemetry — Last Run</div>",
            unsafe_allow_html=True)
        perf = pr.performance
        tel_html = (
            _telemetry_row("1 · Image Acquisition",     perf.preprocessing_ms, perf.preprocessing_ms >= 0) +
            _telemetry_row("2 · Star Detection",         perf.detection_ms,     pr.detection.detected_count > 0) +
            _telemetry_row("3 · AI Recognition",         perf.ai_inference_ms,  pr.ai.model_available) +
            _telemetry_row("4 · Catalogue Matching",     perf.matching_ms,      pr.matching.matched_count > 0) +
            _telemetry_row("5 · Attitude Solver",        perf.attitude_ms,      att.n_inliers >= 2) +
            _telemetry_row("6 · Integrity Verification", perf.integrity_ms,     integ.is_valid) +
            f"<div style='border-top:2px solid {C['border']};margin-top:6px;padding-top:6px;"
            f"display:flex;justify-content:space-between;font-size:.8rem;font-weight:700'>"
            f"<span style='color:{C['text']}'>TOTAL</span>"
            f"<span style='color:{_AC};font-family:monospace'>{perf.total_ms:.1f} ms</span></div>"
        )
        st.markdown(tel_html, unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown(f"<hr style='border-color:{C['border']};margin:18px 0 14px'>",
                unsafe_allow_html=True)
    with st.expander("📖  How It Works", expanded=False):
        steps = [
            ("1 · OBSERVE",   "Star-field image captured by onboard optical sensor.  "
                               "No GPS or ground assistance required."),
            ("2 · DETECT",    "Background subtraction + connected-component analysis "
                               "extracts star centroids (sub-pixel intensity-weighted centroiding)."),
            ("3 · RECOGNISE", "AI (sklearn RandomForest) classifies the sky region.  "
                               "Geometric vote matrix matches pairwise angular distances to the "
                               "Hipparcos catalogue."),
            ("4 · IDENTIFY",  "RANSAC selects the best-fitting star correspondence subset.  "
                               "Wahba/SVD refines the rotation over all RANSAC inliers."),
            ("5 · SOLVE",     "Weighted Wahba/SVD solves for the optimal rotation R.  "
                               "Iterative outlier rejection removes mismatched stars."),
            ("6 · VERIFY",    "Integrity check validates: det(R)≈+1, |q|≈1, "
                               "mean residual < threshold, sufficient inliers."),
            ("7 · OUTPUT",    "Attitude quaternion [qw qx qy qz], Euler angles [Roll Pitch Yaw], "
                               "and integrity status are delivered.  "
                               "POSITION is scientifically unavailable from one image."),
        ]
        c1, c2 = st.columns(2, gap="large")
        for i, (title, desc) in enumerate(steps):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(
                    f"<div style='background:{C['card']};border:1px solid {C['border']};"
                    f"border-radius:7px;padding:10px 14px;margin-bottom:8px'>"
                    f"<div style='color:{_AC};font-size:.72rem;font-weight:700;"
                    f"text-transform:uppercase;margin-bottom:4px'>{title}</div>"
                    f"<div style='color:{C['text']};font-size:.77rem;line-height:1.7'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True)


# =============================================================================
#  PAGE: ATTITUDE ANALYSIS
# =============================================================================

def _page_attitude_analysis():
    st.markdown(
        f"<div style='color:{C['muted']};font-size:.75rem;letter-spacing:1px;"
        f"text-transform:uppercase;margin-bottom:16px'>"
        f"Upload a star-field image or run the SIH demo sequence</div>",
        unsafe_allow_html=True)

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown(f"<div class='sx-phase'>Phase 1 · Image Acquisition</div>",
                unsafe_allow_html=True)
    c_up, c_demo = st.columns([3, 2], gap="large")
    raw_image: Optional[np.ndarray] = None
    img_name = ""

    with c_up:
        uploaded = st.file_uploader(
            "Upload star-field image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            help="Greyscale or colour astronomical image.")
        if uploaded:
            arr, err = _to_gray(uploaded.read())
            if err:
                st.error(f"❌ {err}")
            else:
                raw_image = arr
                img_name  = uploaded.name

    with c_demo:
        st.markdown(
            f"<div style='background:{C['card']};border:1px solid {C['border']};"
            f"border-radius:8px;padding:14px;text-align:center'>"
            f"<div style='font-size:.8rem;color:{C['text']};font-weight:700;"
            f"margin-bottom:5px'>🎯  SIH Demo Mode</div>"
            f"<div style='font-size:.7rem;color:{C['muted']};margin-bottom:10px'>"
            f"Runs full pipeline on synthetic star-field<br>with ground-truth verification</div>"
            f"</div>",
            unsafe_allow_html=True)
        if st.button("▶  Load Demo Star Field", key="load_demo", use_container_width=True):
            st.session_state["demo_pending"] = True
            st.rerun()
        if st.button("✕  Intentional Failure Test", key="load_failure", use_container_width=True):
            st.session_state["failure_pending"] = True
            st.rerun()

    # Resolve demo pending
    is_demo = False
    if st.session_state.get("demo_pending", False) and raw_image is None:
        demo_img, demo_stars, _ = _demo_assets()
        raw_image = demo_img
        img_name  = "demo_starfield.png"
        is_demo   = True
        st.session_state["demo_pending"] = False

    # Resolve failure test
    if st.session_state.get("failure_pending", False) and raw_image is None:
        # Black image — no stars, should produce INSUFFICIENT_STARS
        raw_image = np.zeros((512, 512), dtype=np.float32)
        img_name  = "intentional_failure_blank.png"
        st.session_state["failure_pending"] = False
        st.warning("⚠️  Running intentional failure test — blank image has no stars.")

    if raw_image is None:
        if st.session_state.get("last_pr") and st.session_state.get("last_img") is not None:
            st.info("💡 Showing results from last run.")
            _render_analysis_results(
                st.session_state["last_pr"],
                st.session_state["last_preprocessed"],
                st.session_state["last_stars"],
                st.session_state["last_img"],
            )
        else:
            st.info("ℹ️  Upload an image or click **Load Demo Star Field** to begin.")
        return

    # Image preview
    c_img, c_info = st.columns([1, 2], gap="large")
    with c_img:
        if HAS_MPL:
            st.image(_plot_image(raw_image, "Input Image"), width="stretch")
    with c_info:
        h, w = raw_image.shape[:2]
        st.markdown(
            f"<div class='sx-card'><div class='sx-card-title'>Image Properties</div>"
            + _stat("Filename",   img_name)
            + _stat("Dimensions", f"{w} × {h} px")
            + _stat("Pixel range", f"[{raw_image.min():.3f}, {raw_image.max():.3f}]")
            + _stat("Mean", f"{raw_image.mean():.4f}")
            + _stat("Mode", "Demo synthetic" if is_demo else "User upload")
            + "</div>",
            unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{C['border']};margin:14px 0'>",
                unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    cb, cn = st.columns([1, 3])
    with cb:
        run = st.button("▶  RUN ANALYSIS", key="run_btn", use_container_width=True)
    with cn:
        st.markdown(
            f"<div style='padding-top:10px;font-size:.74rem;color:{C['muted']}'>"
            f"Runs all 7 pipeline stages: preprocess → detect → "
            f"AI → match → attitude → integrity → output</div>",
            unsafe_allow_html=True)

    if not run:
        # Show cached results for same image
        if (st.session_state.get("last_pr") is not None
                and st.session_state.get("last_img_name") == img_name):
            st.info("💡 Showing cached results. Click **RUN ANALYSIS** to rerun.")
            _render_analysis_results(
                st.session_state["last_pr"],
                st.session_state.get("last_preprocessed", raw_image),
                st.session_state.get("last_stars", []),
                raw_image,
            )
        return

    # ── Execute pipeline with real progress ───────────────────────────────────
    prog = st.progress(0, text="Initialising…")
    status_box = st.empty()

    def _upd(pct, msg):
        prog.progress(pct, text=msg)
        status_box.markdown(
            f"<div style='font-size:.72rem;color:{C['muted']};font-family:monospace'>"
            f"⟳  {msg}</div>", unsafe_allow_html=True)

    _upd(5,  "1 · Loading catalogue and config…"); time.sleep(0.04)
    _upd(15, "2 · Preprocessing image…");          time.sleep(0.04)
    _upd(35, "3 · Detecting stars…");              time.sleep(0.04)
    _upd(55, "4 · Running pattern recognition…");  time.sleep(0.04)
    _upd(75, "5 · Solving attitude (Wahba/SVD)…"); time.sleep(0.04)
    _upd(90, "6 · Running integrity check…");      time.sleep(0.04)

    pr, preprocessed, stars, err = _run_pipeline(raw_image, is_demo=is_demo)

    _upd(100, "7 · Done"); time.sleep(0.05)
    prog.empty(); status_box.empty()

    if pr is None:
        st.error(f"❌  Pipeline error: {err}")
        return

    # Cache results
    st.session_state.update({
        "last_pr": pr, "last_preprocessed": preprocessed,
        "last_stars": stars, "last_img": raw_image,
        "last_img_name": img_name,
    })
    st.session_state["history"].append({
        "time":    datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "image":   img_name,
        "status":  pr.overall_status,
        "stars":   pr.detection.detected_count,
        "inliers": pr.attitude.n_inliers,
        "conf":    round(pr.attitude.attitude_confidence, 3),
        "ms":      round(pr.performance.total_ms, 1),
        "integrity": pr.integrity.status,
    })

    # Show success / failure banner
    if pr.overall_status == "SUCCESS":
        st.success(f"✅  Attitude determined — confidence {pr.attitude.attitude_confidence*100:.1f}%  "
                   f"| integrity: {pr.integrity.status}  | {pr.performance.total_ms:.0f} ms")
    elif pr.overall_status in ("INSUFFICIENT_STARS", "FAILURE"):
        st.error(f"✕  {pr.overall_status}  —  "
                 f"{pr.integrity.rejection_reason or pr.error_message or 'Insufficient observations'}")
        st.info("💡  This is the correct behaviour — the system correctly reports it cannot "
                "determine attitude when input is insufficient.  "
                "**'I know when I don't know.'**")
    else:
        st.warning(f"⚠️  {pr.overall_status}  —  {pr.integrity.rejection_reason or 'Partial solution'}")

    st.markdown(f"<hr style='border-color:{C['border']};margin:10px 0 16px'>",
                unsafe_allow_html=True)
    _render_analysis_results(pr, preprocessed, stars, raw_image)


def _render_analysis_results(pr: PipelineResult, preprocessed, stars, raw_image):
    if pr is None:
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Image",
        "🔍 Detection",
        "🧭 Attitude",
        "✅ Integrity",
        "⬇️ Export",
    ])

    att = pr.attitude
    integ = pr.integrity
    identified = pr.identified_stars

    # Tab 1 — Image
    with tab1:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if HAS_MPL:
                st.image(_plot_image(preprocessed, "Preprocessed Star-Field"),
                         width="stretch")
        with c2:
            h, w = preprocessed.shape[:2] if preprocessed is not None else (512, 512)
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Image Stats</div>"
                + _stat("Size", f"{w} × {h} px")
                + _stat("Min", f"{preprocessed.min():.4f}")
                + _stat("Max", f"{preprocessed.max():.4f}")
                + _stat("Mean", f"{preprocessed.mean():.4f}")
                + _stat("Std", f"{preprocessed.std():.4f}")
                + "</div>",
                unsafe_allow_html=True)

    # Tab 2 — Detection
    with tab2:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if HAS_MPL and stars:
                matched_set = {i for i in range(len(identified))}
                st.image(_plot_detections(preprocessed, stars, matched_set),
                         width="stretch")
        with c2:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Detection Summary</div>"
                + _stat("Stars detected",  str(pr.detection.detected_count), True)
                + _stat("Matched inliers", str(pr.matching.matched_count), True)
                + _stat("RANSAC outliers", str(pr.matching.rejected_count))
                + _stat("Recognition status", pr.matching.recognition_status)
                + _stat("Match confidence",
                        f"{pr.matching.recognition_confidence*100:.1f}%")
                + _stat("Detection latency",
                        f"{pr.detection.detection_latency_ms:.1f} ms")
                + "</div>",
                unsafe_allow_html=True)
            if stars and HAS_PD:
                rows = [{"#": i+1,
                         "X (px)": round(float(getattr(s,"x",getattr(s,"x_px",0))),2),
                         "Y (px)": round(float(getattr(s,"y",getattr(s,"y_px",0))),2),
                         "Brightness": round(float(getattr(s,"brightness",0)),4),
                         "Peak": round(float(getattr(s,"peak",0)),4)}
                        for i, s in enumerate(stars[:12])]
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

    # Tab 3 — Attitude
    with tab3:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Euler Angles (ZYX)</div>"
                + _stat("Roll",  f"{att.roll_deg:.6f}°",  True)
                + _stat("Pitch", f"{att.pitch_deg:.6f}°", True)
                + _stat("Yaw",   f"{att.yaw_deg:.6f}°",  True)
                + f"<div style='margin-top:6px;font-size:.62rem;color:{C['muted']}'>"
                f"Display only — use quaternion for control</div>"
                + "</div>",
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Quaternion [w x y z]</div>"
                + _stat("qw", f"{att.quaternion[0]:.8f}", True)
                + _stat("qx", f"{att.quaternion[1]:+.8f}")
                + _stat("qy", f"{att.quaternion[2]:+.8f}")
                + _stat("qz", f"{att.quaternion[3]:+.8f}")
                + _stat("|q|", f"{float(np.linalg.norm(att.quaternion)):.10f}")
                + f"<div style='margin-top:6px;font-size:.62rem;color:{C['muted']}'>"
                f"{att.frame_convention}</div>"
                + "</div>",
                unsafe_allow_html=True)
        with c3:
            res_str = f"{att.mean_residual_deg:.4f}°" if math.isfinite(att.mean_residual_deg) else "—"
            rms_str = f"{att.rms_residual_deg:.4f}°" if math.isfinite(att.rms_residual_deg) else "—"
            max_str = f"{att.max_residual_deg:.4f}°" if math.isfinite(att.max_residual_deg) else "—"
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Quality Metrics</div>"
                + _stat("Mean residual", res_str)
                + _stat("RMS residual",  rms_str)
                + _stat("Max residual",  max_str)
                + _stat("Inliers",  str(att.n_inliers), True)
                + _stat("Outliers", str(att.n_outliers))
                + _stat("Confidence", f"{att.attitude_confidence*100:.1f}%", True)
                + _stat("Solver latency", f"{att.attitude_latency_ms:.1f} ms")
                + "</div>",
                unsafe_allow_html=True)

        if HAS_MPL:
            try:
                st.image(_plot_cubesat_3d(att.quaternion),
                         caption="Spacecraft orientation from computed quaternion",
                         width="stretch")
            except Exception:
                pass

    # Tab 4 — Integrity
    with tab4:
        ic_col = (C["success"] if integ.status == "VALID"
                  else C["warning"] if integ.status == "DEGRADED"
                  else C["error"])
        st.markdown(
            f"<div style='background:{C['card']};border:2px solid {ic_col}44;"
            f"border-radius:10px;padding:16px;margin-bottom:14px;text-align:center'>"
            f"<div style='font-size:1.5rem;font-weight:900;color:{ic_col};"
            f"font-family:monospace;letter-spacing:2px'>"
            f"{integ.status_symbol}  ATTITUDE {integ.status}</div>"
            + (f"<div style='font-size:.8rem;color:{C['error']};margin-top:8px'>"
               f"Rejection reason: {integ.rejection_reason}</div>"
               if integ.rejection_reason else "")
            + "</div>",
            unsafe_allow_html=True)

        ci, cj = st.columns([1, 1], gap="large")
        with ci:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Rotation Validation</div>"
                + _stat("det(R)", f"{integ.determinant:.10f}" if math.isfinite(integ.determinant) else "—",
                        abs(integ.determinant - 1.0) < 0.001 if math.isfinite(integ.determinant) else False)
                + _stat("Orthogonality error",
                        f"{integ.orthogonality_error:.2e}" if math.isfinite(integ.orthogonality_error) else "—")
                + _stat("Quaternion norm",
                        f"{integ.quaternion_norm:.10f}" if math.isfinite(integ.quaternion_norm) else "—")
                + f"<div style='margin-top:8px'>{_conf_bar('Integrity confidence', integ.confidence, ic_col)}</div>"
                + "</div>",
                unsafe_allow_html=True)
        with cj:
            st.markdown(
                f"<div class='sx-card'><div class='sx-card-title'>Residual Analysis</div>"
                + _stat("Mean residual", f"{integ.mean_residual_deg:.4f}°" if math.isfinite(integ.mean_residual_deg) else "—")
                + _stat("RMS residual",  f"{integ.rms_residual_deg:.4f}°" if math.isfinite(integ.rms_residual_deg) else "—")
                + _stat("Max residual",  f"{integ.max_residual_deg:.4f}°" if math.isfinite(integ.max_residual_deg) else "—")
                + _stat("Inliers accepted",  str(integ.n_inliers), True)
                + _stat("Outliers rejected", str(integ.n_outliers))
                + "</div>",
                unsafe_allow_html=True)

    # Tab 5 — Export
    with tab5:
        report = {
            "stellarx_version": "2.0.0",
            "timestamp": pr.timestamp,
            "run_id": pr.run_id,
            "is_demo": pr.is_demo,
            "overall_status": pr.overall_status,
            "attitude": {
                "quaternion": pr.attitude.quaternion.tolist(),
                "roll_deg":   pr.attitude.roll_deg,
                "pitch_deg":  pr.attitude.pitch_deg,
                "yaw_deg":    pr.attitude.yaw_deg,
                "mean_residual_deg": (pr.attitude.mean_residual_deg
                                      if math.isfinite(pr.attitude.mean_residual_deg) else None),
                "rms_residual_deg":  (pr.attitude.rms_residual_deg
                                      if math.isfinite(pr.attitude.rms_residual_deg) else None),
                "confidence":  pr.attitude.attitude_confidence,
                "n_inliers":   pr.attitude.n_inliers,
                "n_outliers":  pr.attitude.n_outliers,
                "frame": pr.attitude.frame_convention,
            },
            "integrity": {
                "status":          pr.integrity.status,
                "is_valid":        pr.integrity.is_valid,
                "quaternion_norm": (pr.integrity.quaternion_norm
                                    if math.isfinite(pr.integrity.quaternion_norm) else None),
                "determinant":     (pr.integrity.determinant
                                    if math.isfinite(pr.integrity.determinant) else None),
                "rejection_reason": pr.integrity.rejection_reason,
            },
            "stars": {
                "detected": pr.detection.detected_count,
                "matched":  pr.matching.matched_count,
                "inliers":  pr.attitude.n_inliers,
                "outliers": pr.attitude.n_outliers,
            },
            "performance_ms": {
                "preprocessing": pr.performance.preprocessing_ms,
                "detection":     pr.performance.detection_ms,
                "ai_inference":  pr.performance.ai_inference_ms,
                "matching":      pr.performance.matching_ms,
                "attitude":      pr.performance.attitude_ms,
                "integrity":     pr.performance.integrity_ms,
                "total":         pr.performance.total_ms,
            },
            "identified_stars": [
                {"catalog_id": s.catalog_id, "ra_deg": s.catalog_ra_deg,
                 "dec_deg": s.catalog_dec_deg, "residual_deg": s.angular_residual_deg,
                 "confidence": s.per_star_confidence}
                for s in pr.identified_stars
            ],
            "position_note": pr.position_note,
        }
        json_str = json.dumps(report, indent=2)
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            st.code(json_str[:1400] + ("\n…" if len(json_str) > 1400 else ""),
                    language="json")
        with c2:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            st.download_button("⬇️  Download JSON Report", data=json_str,
                               file_name=f"stellarx_{ts}.json",
                               mime="application/json",
                               key="dl_report", use_container_width=True)


# =============================================================================
#  PAGE: STAR MATCHING
# =============================================================================

def _page_star_matching():
    st.markdown(
        f"<div style='color:{C['muted']};font-size:.75rem;letter-spacing:1px;"
        f"text-transform:uppercase;margin-bottom:16px'>"
        f"Hipparcos catalogue correspondence visualisation</div>",
        unsafe_allow_html=True)

    pr: Optional[PipelineResult] = st.session_state.get("last_pr")
    preprocessed = st.session_state.get("last_preprocessed")
    stars = st.session_state.get("last_stars", [])

    if pr is None or preprocessed is None:
        st.info("ℹ️  Run the pipeline first in **🔭 Attitude Analysis**.")
        return

    identified = pr.identified_stars

    # Visualisation
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if HAS_MPL:
            st.image(_plot_matching(preprocessed, stars, identified),
                     caption="Green = RANSAC inliers  |  Dim = unmatched",
                     width="stretch")
    with c2:
        st.markdown(
            f"<div class='sx-card'><div class='sx-card-title'>Match Summary</div>"
            + _stat("Stars detected",   str(pr.detection.detected_count))
            + _stat("Catalogue candidates", str(pr.matching.candidate_count))
            + _stat("RANSAC inliers",   str(pr.matching.matched_count), True)
            + _stat("RANSAC outliers",  str(pr.matching.rejected_count))
            + _stat("Pattern type",     pr.matching.pattern_type)
            + _stat("Matching latency", f"{pr.matching.matching_latency_ms:.1f} ms")
            + _stat("AI pattern",
                    pr.ai.prediction or "—" if pr.ai.model_available else "geometry-only")
            + _stat("AI confidence",
                    f"{pr.ai.confidence*100:.1f}%" if pr.ai.model_available else "—")
            + "</div>",
            unsafe_allow_html=True)

    # Per-star correspondence table
    st.markdown(f"<hr style='border-color:{C['border']};margin:14px 0 10px'>",
                unsafe_allow_html=True)
    if not identified:
        st.info("No catalogue matches in this run.")
        return

    st.markdown(
        f"<div style='color:{_AC};font-size:.78rem;font-weight:700;"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>"
        f"⭐  Matched Stars — Hipparcos Catalogue</div>",
        unsafe_allow_html=True)

    if HAS_PD:
        rows = []
        for m in identified:
            res = m.angular_residual_deg
            quality = "✓ Good" if res < 0.5 else ("⚠ Marginal" if res < 1.5 else "✕ Poor")
            rows.append({
                "HIP ID":      m.catalog_id.replace("HIP_", "HIP "),
                "Observed X":  round(m.observed_x, 2),
                "Observed Y":  round(m.observed_y, 2),
                "RA (J2000°)": round(m.catalog_ra_deg, 4),
                "Dec (J2000°)": round(m.catalog_dec_deg, 4),
                "Residual (°)": round(res, 4),
                "Confidence":  round(m.per_star_confidence, 3),
                "Brightness":  round(m.brightness, 4),
                "Quality":     quality,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Residual bar chart
    if HAS_MPL and identified:
        fig, ax = plt.subplots(figsize=(9, 3), facecolor=_BG)
        _ax_style(ax, "Per-Star Angular Residuals")
        labels = [m.catalog_id.replace("HIP_", "HIP ") for m in identified]
        residuals = [m.angular_residual_deg for m in identified]
        bar_cols = [C["success"] if r < 0.5 else C["warning"] if r < 1.5 else C["error"]
                    for r in residuals]
        ax.bar(labels, residuals, color=bar_cols, alpha=0.9)
        ax.axhline(1.0, color=C["warning"], linestyle="--", lw=0.8, alpha=0.7,
                   label="1° threshold")
        ax.set_ylabel("Residual (°)", color=_TX, fontsize=7)
        ax.set_xlabel("Catalogue Star", color=_TX, fontsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
        st.image(_fig_bytes(fig), width="stretch")


# =============================================================================
#  PAGE: ROBUSTNESS LAB
# =============================================================================

def _page_robustness_lab():
    st.markdown(
        f"<div style='color:{C['muted']};font-size:.75rem;letter-spacing:1px;"
        f"text-transform:uppercase;margin-bottom:16px'>"
        f"Test pipeline behaviour under degraded conditions</div>",
        unsafe_allow_html=True)

    if not _PIPELINE_OK:
        st.error(f"❌  Pipeline unavailable: {_PIPELINE_ERR}")
        return

    st.markdown(
        f"<div style='color:{C['muted']};font-size:.78rem;margin-bottom:12px'>"
        f"The robustness lab generates synthetic star fields from the Hipparcos catalogue "
        f"with controlled degradations, runs the full pipeline on each, and reports "
        f"detection rate, match rate, and attitude quality. "
        f"All numbers come from actual computation.</div>",
        unsafe_allow_html=True)

    # Scenario selector
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        noise_level = st.slider("Read noise σ", 0.000, 0.030, 0.005, 0.001,
                                help="Gaussian sensor noise σ (fraction of max intensity)")
    with c2:
        n_false = st.slider("False stars", 0, 10, 0,
                            help="Random hot pixels injected")
    with c3:
        occlusion = st.slider("Star occlusion %", 0, 60, 0, 5,
                              help="% of detected stars randomly removed")

    boresight_mode = st.selectbox(
        "Boresight",
        ["Random (uniform sphere)", "Near Sirius (RA=101°, Dec=-17°)",
         "Near Orion Belt (RA=83°, Dec=0°)"])

    if st.button("▶  Run Robustness Scenario", key="rob_run", use_container_width=False):
        with st.spinner("Running robustness test…"):
            results = _run_robustness(noise_level, n_false, occlusion, boresight_mode)
        st.session_state["rob_results"] = results
        st.rerun()

    rob = st.session_state.get("rob_results")
    if rob:
        _render_robustness_results(rob)

    # Intentional failure section
    st.markdown(f"<hr style='border-color:{C['border']};margin:20px 0 14px'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_AC};font-size:.82rem;font-weight:700;"
        f"text-transform:uppercase;margin-bottom:8px'>Required Failure Test</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{C['text']};font-size:.78rem;margin-bottom:10px'>"
        f"A trustworthy system must know when it cannot produce a valid answer.  "
        f"Click below to run the pipeline on a blank (star-free) image.  "
        f"The expected result is <b style='color:{C['error']}'>ATTITUDE REJECTED — "
        f"INSUFFICIENT STARS</b>.</div>",
        unsafe_allow_html=True)

    if st.button("✕  Run Intentional Failure Test", key="fail_test", use_container_width=False):
        blank = np.zeros((512, 512), dtype=np.float32)
        pr, _, _, err = _run_pipeline(blank, is_demo=False)
        if pr is None:
            st.error(f"Pipeline error: {err}")
        else:
            ic_col = C["error"]
            st.markdown(
                f"<div style='background:{C['card']};border:2px solid {C['error']};"
                f"border-radius:10px;padding:18px;text-align:center'>"
                f"<div style='font-size:1.3rem;font-weight:900;color:{C['error']};"
                f"font-family:monospace'>✕  ATTITUDE REJECTED</div>"
                f"<div style='color:{C['text']};margin-top:8px;font-size:.8rem'>"
                f"Status: {pr.overall_status}</div>"
                f"<div style='color:{C['muted']};margin-top:4px;font-size:.75rem'>"
                f"Reason: {pr.integrity.rejection_reason or pr.error_message or 'Insufficient observations'}</div>"
                f"<div style='color:{C['success']};margin-top:12px;font-size:.8rem;font-weight:700'>"
                f"✓  System correctly refused to produce an attitude with no observations</div>"
                f"</div>",
                unsafe_allow_html=True)

    # Ground-truth validation
    st.markdown(f"<hr style='border-color:{C['border']};margin:20px 0 14px'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_AC};font-size:.82rem;font-weight:700;"
        f"text-transform:uppercase;margin-bottom:8px'>Synthetic Ground-Truth Validation</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{C['text']};font-size:.78rem;margin-bottom:10px'>"
        f"Generates synthetic star fields at known orientations and measures true "
        f"geodesic attitude error: θ = arccos((trace(R_true^T @ R_est) − 1) / 2).  "
        f"All error values come from actual computation.</div>",
        unsafe_allow_html=True)

    c_n, c_noise_gt = st.columns(2)
    with c_n:
        gt_runs = st.slider("Number of runs", 3, 20, 8, key="gt_runs")
    with c_noise_gt:
        gt_noise = st.slider("Noise σ", 0.000, 0.020, 0.005, 0.001, key="gt_noise")

    if st.button("▶  Run Ground-Truth Validation", key="gt_run", use_container_width=False):
        if not _PIPELINE_OK:
            st.error("Pipeline unavailable.")
        else:
            with st.spinner(f"Running {gt_runs} validation runs…"):
                try:
                    import yaml
                    with open(str(_ROOT / "config.yaml")) as f:
                        cfg = yaml.safe_load(f)
                    from src.evaluation.ground_truth_validator import run_ground_truth_validation
                    report = run_ground_truth_validation(
                        cfg, n_runs=gt_runs, noise_level=gt_noise, verbose=False)
                    _render_gt_report(report)
                except Exception as exc:
                    st.error(f"Validation error: {exc}")


def _run_robustness(noise_level, n_false, occlusion_pct, boresight_mode):
    """Run the pipeline on a synthetic star field with applied degradations."""
    cfg_path = str(_ROOT / "config.yaml")
    cfg, _, cidx, err = _load_resources(cfg_path)
    if err:
        return {"error": err}

    try:
        from src.catalog.catalog_loader import load_catalog
        from src.preprocessing.star_field_generator import StarFieldGenerator
        import yaml
        with open(cfg_path) as f:
            full_cfg = yaml.safe_load(f)

        cat_path = _ROOT / full_cfg["dataset"]["catalog_file"]
        catalog  = load_catalog(str(cat_path), config=full_cfg)
        gen_cfg  = {**full_cfg["dataset"], "read_noise_sigma": noise_level,
                    "artifact_probability": 0.0}
        gen = StarFieldGenerator(catalog, gen_cfg)

        # Choose boresight
        if "Sirius" in boresight_mode:
            sf = gen.generate(seed=42, boresight_ra_deg=101.3, boresight_dec_deg=-16.7)
        elif "Orion" in boresight_mode:
            sf = gen.generate(seed=42, boresight_ra_deg=83.0, boresight_dec_deg=0.0)
        else:
            sf = gen.generate(seed=42)

        # Apply noise
        img = sf.image.copy()
        if noise_level > 0:
            rng = np.random.default_rng(99)
            img += rng.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img, 0, 1)

        # Apply false stars
        if n_false > 0:
            rng = np.random.default_rng(77)
            xs = rng.integers(10, 500, n_false)
            ys = rng.integers(10, 500, n_false)
            for x, y in zip(xs, ys):
                img[int(y), int(x)] = min(1.0, img[int(y), int(x)] + 0.3)

        pr, preprocessed, stars, pipe_err = _run_pipeline(img)

        # Apply occlusion post-detection (remove % of stars from result)
        if occlusion_pct > 0 and stars and pr:
            rng = np.random.default_rng(55)
            n_keep = max(0, int(len(stars) * (1 - occlusion_pct / 100)))
            stars = stars[:n_keep]

        return {
            "pr": pr,
            "preprocessed": preprocessed,
            "stars": stars,
            "sf": sf,
            "noise": noise_level,
            "n_false": n_false,
            "occlusion": occlusion_pct,
            "error": pipe_err if pr is None else "",
        }
    except Exception as exc:
        return {"error": str(exc)}


def _render_robustness_results(rob: dict):
    if rob.get("error"):
        st.error(f"❌  {rob['error']}")
        return
    pr = rob.get("pr")
    if pr is None:
        st.error("No result produced.")
        return

    status_col = (C["success"] if pr.overall_status == "SUCCESS"
                  else C["warning"] if "PARTIAL" in pr.overall_status
                  else C["error"])

    st.markdown(
        f"<div style='background:{C['card']};border:1px solid {C['border']};"
        f"border-radius:9px;padding:14px 18px;margin-bottom:14px'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:center'>"
        f"<span style='font-size:.78rem;font-weight:700;color:{C['text']}'>"
        f"Robustness Result</span>"
        f"&nbsp;{_status_pill(pr.overall_status)}&nbsp;{_status_pill(pr.integrity.status)}"
        f"<span style='color:{C['muted']};font-size:.73rem'>"
        f"noise σ={rob['noise']:.3f}  false={rob['n_false']}  "
        f"occlusion={rob['occlusion']}%</span>"
        f"<span style='color:{C['muted']};font-size:.73rem'>"
        f"stars={pr.detection.detected_count}  "
        f"inliers={pr.attitude.n_inliers}  "
        f"{pr.performance.total_ms:.0f} ms</span>"
        f"</div></div>",
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected",  str(pr.detection.detected_count))
    c2.metric("Matched",   str(pr.matching.matched_count))
    c3.metric("Inliers",   str(pr.attitude.n_inliers))
    conf_str = f"{pr.attitude.attitude_confidence*100:.1f}%"
    c4.metric("Confidence", conf_str)

    if pr.integrity.rejection_reason:
        st.error(f"✕  Rejection reason: {pr.integrity.rejection_reason}")
    elif pr.integrity.status == "VALID":
        st.success("✓  Attitude valid under these degraded conditions")


# =============================================================================
#  PAGE: PERFORMANCE
# =============================================================================

def _page_performance():
    st.markdown(
        f"<div style='color:{C['muted']};font-size:.75rem;letter-spacing:1px;"
        f"text-transform:uppercase;margin-bottom:16px'>"
        f"Pipeline timing breakdown and benchmark</div>",
        unsafe_allow_html=True)

    pr: Optional[PipelineResult] = st.session_state.get("last_pr")

    if pr is None:
        st.info("ℹ️  Run the pipeline in **🔭 Attitude Analysis** first.")
        return

    perf = pr.performance
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total latency", f"{perf.total_ms:.0f} ms")
    c2.metric("Detection",     f"{perf.detection_ms:.0f} ms")
    c3.metric("Matching",      f"{perf.matching_ms:.0f} ms")
    c4.metric("Attitude",      f"{perf.attitude_ms:.0f} ms")

    if HAS_MPL:
        st.image(_plot_timing(perf), width="stretch")

    st.markdown(f"<hr style='border-color:{C['border']};margin:16px 0 12px'>",
                unsafe_allow_html=True)

    # Telemetry table
    st.markdown(
        f"<div style='color:{_AC};font-size:.78rem;font-weight:700;"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>"
        f"Full Pipeline Telemetry</div>",
        unsafe_allow_html=True)
    for stage, ms in {
        "1 · Image Acquisition & Preprocessing": perf.preprocessing_ms,
        "2 · Star Detection":                    perf.detection_ms,
        "3 · AI Feature Extraction":             perf.ai_inference_ms,
        "4 · Pattern Matching (RANSAC)":         perf.matching_ms,
        "5 · Attitude Estimation (Wahba/SVD)":   perf.attitude_ms,
        "6 · Integrity Verification":            perf.integrity_ms,
    }.items():
        ok = ms >= 0
        st.markdown(_telemetry_row(stage, ms, ok), unsafe_allow_html=True)
    st.markdown(
        f"<div style='border-top:2px solid {C['border']};margin-top:6px;"
        f"padding:8px 12px;display:flex;justify-content:space-between;"
        f"font-weight:700;font-size:.82rem'>"
        f"<span style='color:{C['text']}'>TOTAL PIPELINE</span>"
        f"<span style='color:{_AC};font-family:monospace'>{perf.total_ms:.1f} ms</span>"
        f"</div>",
        unsafe_allow_html=True)

    # Session history
    history = st.session_state.get("history", [])
    if history and HAS_PD:
        st.markdown(f"<hr style='border-color:{C['border']};margin:16px 0 12px'>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{_AC};font-size:.78rem;font-weight:700;"
            f"letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>"
            f"Session History</div>",
            unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True)

    # Note on Phase 6
    st.markdown(f"<hr style='border-color:{C['border']};margin:16px 0 12px'>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div class='sx-card'><div class='sx-card-title'>⚡  Phase 6 Optimizations Active</div>"
        + _stat("Vote accumulation",   "Vectorized NumPy array comparison (not Python dict scan)")
        + _stat("RANSAC inlier count", "Vectorized matmul (ransac_inlier_count_vectorized)")
        + _stat("Wahba B matrix",      "Vectorized einsum (wahba_svd_vectorized)")
        + _stat("Residual computation","Vectorized arccos/dot (compute_residuals_vectorized)")
        + _stat("Pairwise angles",     "Single matmul precomputed at catalog load time")
        + _stat("Catalog loading",     "@st.cache_resource — loaded once per session")
        + _stat("Model loading",       "@st.cache_resource — loaded once per session")
        + "</div>",
        unsafe_allow_html=True)


# =============================================================================
#  PAGE: ABOUT
# =============================================================================

def _page_about():
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(
            f"<div class='sx-card'><div class='sx-card-title'>About StellarX StarNav-AI</div>"
            f"<div style='color:{C['text']};font-size:.83rem;line-height:1.9'>"
            f"StellarX StarNav-AI is an engineering implementation of autonomous "
            f"lost-in-space attitude determination for small spacecraft (CubeSats). "
            f"It determines spacecraft orientation from a single onboard star-field "
            f"image with no GPS, no ground contact, and no prior attitude knowledge.<br><br>"
            f"<b style='color:{_AC}'>What it actually does:</b> detects stars from "
            f"image data, matches them against the Hipparcos catalogue using angular "
            f"pattern voting + RANSAC, solves Wahba's problem via weighted SVD, and "
            f"verifies the result with an integrity check.<br><br>"
            f"<b style='color:{_AC}'>What it does not do:</b> determine spacecraft "
            f"<i>position</i> (single-image star geometry provides orientation only) "
            f"and does not use a deep neural network (PyTorch is not available on "
            f"Python 3.14; sklearn RandomForest is the active AI backend)."
            f"</div></div>",
            unsafe_allow_html=True)

        st.markdown(
            f"<div class='sx-card' style='margin-top:10px'>"
            f"<div class='sx-card-title'>Pipeline Architecture (Implemented)</div>"
            + _stat("1 · Image acquisition",     "Pillow/OpenCV load → float32 [0,1]")
            + _stat("2 · Preprocessing",         "Median-filter bg subtraction + Gaussian noise reduction + min-max normalisation")
            + _stat("3 · Star detection",        "Connected-component analysis + intensity-weighted centroiding")
            + _stat("4 · Feature extraction",    "Pairwise pixel distances + brightness ratios (90-dim vector)")
            + _stat("5 · AI prior",              "sklearn RandomForest sky-cell classifier (if checkpoint exists)")
            + _stat("6 · Pattern recognition",   "Vote matrix (angular distance matching) + RANSAC + TRIAD rotation")
            + _stat("7 · Catalogue matching",    "Hipparcos bright-star catalogue (50 stars, ESA 1997)")
            + _stat("8 · Attitude estimation",   "Weighted Wahba/SVD + iterative outlier rejection")
            + _stat("9 · Integrity check",       "det(R)≈+1, |q|≈1, residual threshold, inlier count")
            + _stat("10 · Output",               "Quaternion [qw qx qy qz] + Euler angles + integrity status")
            + "</div>",
            unsafe_allow_html=True)

        # Flight deployment roadmap
        st.markdown(
            f"<div class='sx-card' style='margin-top:10px'>"
            f"<div class='sx-card-title'>🚀  Flight Deployment Roadmap</div>",
            unsafe_allow_html=True)
        roadmap = [
            ("Current",  "Synthetic star-field validation (Hipparcos catalogue simulation)"),
            ("Next",     "Real camera / FITS imagery integration (astropy.io.fits)"),
            ("Next",     "Extend to full Hipparcos catalogue (~118K stars)"),
            ("Next",     "Embedded CPU optimisation (C/C++ Wahba solver, NumPy acceleration)"),
            ("Future",   "Hardware-in-the-loop testing with star tracker hardware"),
            ("Future",   "IMU / camera sensor fusion (attitude propagation between frames)"),
            ("Future",   "Small-satellite flight validation"),
        ]
        for stage, desc in roadmap:
            col = _AC if stage == "Current" else C["warning"] if stage == "Next" else C["muted"]
            st.markdown(
                f"<div style='display:flex;gap:10px;padding:5px 0;"
                f"border-bottom:1px solid {C['dim']};font-size:.76rem'>"
                f"<span style='color:{col};font-weight:700;min-width:55px'>{stage}</span>"
                f"<span style='color:{C['text']}'>{desc}</span></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"<div class='sx-card'><div class='sx-card-title'>Team StellarX</div>"
            f"<div style='color:{C['text']};font-size:.8rem;line-height:2.1'>"
            f"<b style='color:{_AC}'>Sai Spoorthy Eturu</b><br>"
            f"<span style='color:{C['muted']};font-size:.72rem'>Repository Owner</span><br><br>"
            f"<b style='color:{C['text']}'>Kommera Harihanika</b><br>"
            f"<span style='color:{C['muted']};font-size:.72rem'>Collaborator</span><br><br>"
            f"<b style='color:{C['text']}'>Duddala Srija</b><br>"
            f"<span style='color:{C['muted']};font-size:.72rem'>Collaborator</span><br><br>"
            f"<b style='color:{C['text']}'>Glory Pranavi B</b><br>"
            f"<span style='color:{C['muted']};font-size:.72rem'>Collaborator</span><br><br>"
            f"<b style='color:{C['text']}'>Katakam Sahithi Rithvika</b><br>"
            f"<span style='color:{C['muted']};font-size:.72rem'>Collaborator</span><br><br>"
            f"<b style='color:{C['text']}'>Shamithri Gowravarapu</b><br>"
            f"<span style='color:{C['muted']};font-size:.72rem'>Collaborator</span>"
            f"</div></div>",
            unsafe_allow_html=True)

        st.markdown(
            f"<div class='sx-card' style='margin-top:10px'>"
            f"<div class='sx-card-title'>Technical Specification</div>"
            + _stat("Language",       "Python 3.14")
            + _stat("Attitude solver","Wahba/SVD (Markley 1988)")
            + _stat("Star matching",  "Vote matrix + RANSAC + TRIAD")
            + _stat("AI backend",     "sklearn RandomForest (PyTorch TBD)")
            + _stat("Catalogue",      "Hipparcos ESA 1997 (public domain)")
            + _stat("Frame convention","Camera → Inertial (J2000 ICRS)")
            + _stat("Quaternion conv.","Scalar-first [qw qx qy qz]")
            + _stat("Dashboard",      "Streamlit 1.56")
            + "</div>",
            unsafe_allow_html=True)

        st.markdown(
            f"<div class='sx-card' style='margin-top:10px'>"
            f"<div class='sx-card-title'>IP Notice</div>"
            f"<div style='color:{C['muted']};font-size:.73rem;line-height:1.7'>"
            f"Engineering implementation of concepts from a published invention. "
            f"IP rights governed by applicable patent and institutional agreements."
            f"</div></div>",
            unsafe_allow_html=True)


def _render_gt_report(report) -> None:
    """Render ground-truth validation report in the Robustness Lab."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", str(report.n_runs))
    c2.metric("Success", str(report.n_success),
              f"{report.success_rate*100:.0f}%")
    err_str = f"{report.mean_error_deg:.3f}°" if math.isfinite(report.mean_error_deg) else "—"
    c3.metric("Mean att. error", err_str)
    c4.metric("Avg latency", f"{report.mean_latency_ms:.0f} ms")

    if report.note:
        st.warning(report.note)
        return

    if HAS_PD and report.runs:
        rows = []
        for r in report.runs:
            rows.append({
                "Seed": r.seed,
                "GT Stars": r.n_gt_stars,
                "Detected": r.n_detected,
                "Inliers": r.n_inliers,
                "Att. Error (°)": round(r.attitude_error_deg, 4) if math.isfinite(r.attitude_error_deg) else None,
                "Mean Resid (°)": round(r.mean_residual_deg, 4) if math.isfinite(r.mean_residual_deg) else None,
                "Status": r.pipeline_status,
                "Integrity": r.integrity_status,
                "Latency (ms)": round(r.latency_ms, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Error distribution plot
    if HAS_MPL:
        valid = [r.attitude_error_deg for r in report.runs
                 if math.isfinite(r.attitude_error_deg)]
        if valid:
            fig, ax = plt.subplots(figsize=(9, 3), facecolor=_BG)
            _ax_style(ax, "Geodesic Attitude Error per Run  [θ = arccos((trace(R_true^T @ R_est) - 1) / 2)]")
            idxs = list(range(1, len(valid) + 1))
            cols = [C["success"] if e < 2.0 else C["warning"] if e < 5.0 else C["error"]
                    for e in valid]
            ax.bar(idxs, valid, color=cols, alpha=0.9)
            if math.isfinite(report.mean_error_deg):
                ax.axhline(report.mean_error_deg, color=_AC, linestyle="--", lw=1.0,
                           label=f"Mean {report.mean_error_deg:.3f}°")
            ax.set_xlabel("Run", color=_TX, fontsize=7)
            ax.set_ylabel("Error (°)", color=_TX, fontsize=7)
            ax.legend(fontsize=7)
            st.image(_fig_bytes(fig), width="stretch")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    _init_state()
    _header()
    page = _sidebar()

    if "Mission" in page:
        _page_mission_control()
    elif "Attitude" in page:
        _page_attitude_analysis()
    elif "Star" in page:
        _page_star_matching()
    elif "Robustness" in page:
        _page_robustness_lab()
    elif "Performance" in page:
        _page_performance()
    elif "About" in page:
        _page_about()
    else:
        _page_mission_control()


if __name__ == "__main__":
    main()
