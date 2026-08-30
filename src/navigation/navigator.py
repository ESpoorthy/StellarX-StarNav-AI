"""
navigator.py — Phase 5
======================
End-to-end navigation pipeline: image → attitude determination.

Pipeline stages:
  1. Star detection          (Phase 2)
  2. Feature extraction      (Phase 3, optional neural prior)
  3. Pattern building        (Phase 4)
  4. Pattern recognition     (Phase 4: vote + RANSAC + Wahba)
  5. Attitude estimation     (Phase 5: weighted Wahba/SVD + outlier rejection)
  6. Position note           (Phase 5: scientifically honest unavailability)

Navigation status semantics:
  SUCCESS          — attitude determined, meets all quality thresholds
  PARTIAL          — attitude estimated but below SUCCESS threshold
  LOW_CONFIDENCE   — weak match, attitude unreliable
  INSUFFICIENT_STARS — too few stars detected or matched
  ATTITUDE_FAILURE — sufficient stars but attitude estimation failed
  FAILURE          — recognition failed
  ERROR            — unexpected exception

Position status:
  ATTITUDE_DETERMINED / POSITION_UNAVAILABLE
  (single-image star tracking cannot determine absolute position)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.navigation.attitude_estimator import (
    AttitudeEstimate,
    angular_error_deg,
    estimate_attitude,
)
from src.navigation.position_estimator import PositionEstimate, estimate_position
from src.preprocessing.star_detection import detect_stars, extract_features
from src.recognition.catalog_index import CatalogIndex
from src.recognition.pattern_builder import build_pattern
from src.recognition.pattern_matcher import RecognitionOutput, RecognitionStatus, run_recognition


# ---------------------------------------------------------------------------
# NavigationResult
# ---------------------------------------------------------------------------


@dataclass
class NavigationResult:
    """Complete Phase 5 navigation output for one image.

    Attributes
    ----------
    timestamp : float
        Unix timestamp (seconds).
    status : str
        Overall navigation status string.
    attitude_status : str
        "DETERMINED", "PARTIAL", "LOW_CONFIDENCE", "FAILURE".
    position_status : str
        Always "UNAVAILABLE" for single-image case.
    velocity_status : str
        Always "UNAVAILABLE" for single-image case.

    quaternion : np.ndarray  [qw, qx, qy, qz], camera→inertial.
    rotation_matrix : np.ndarray  3×3, camera→inertial.
    euler_angles_deg : np.ndarray  [yaw, pitch, roll] degrees, display only.
    attitude_confidence : float  In [0, 1].
    attitude_residual_deg : float  Mean angular residual, degrees.
    max_residual_deg : float  Max per-star residual, degrees.

    position_note : str  Explanation of why position is unavailable.

    n_observed_stars : int
    n_matched_stars : int
    n_inlier_stars : int
    n_outlier_stars : int
    identified_stars : list[IdentifiedStar]

    preprocessing_time_ms : float
    detection_time_ms : float
    feature_extraction_time_ms : float
    recognition_time_ms : float
    attitude_time_ms : float
    total_time_ms : float

    error_message : str
    """

    timestamp: float = 0.0
    status: str = "FAILURE"
    attitude_status: str = "FAILURE"
    position_status: str = "UNAVAILABLE"
    velocity_status: str = "UNAVAILABLE"

    # Attitude
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    euler_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude_confidence: float = 0.0
    attitude_residual_deg: float = float("nan")
    max_residual_deg: float = float("nan")

    # Position (always unavailable for single image)
    position_note: str = (
        "POSITION UNAVAILABLE: Single-image star tracking provides attitude only. "
        "Position requires multi-image data, orbital mechanics, or additional sensors."
    )

    # Stars
    n_observed_stars: int = 0
    n_matched_stars: int = 0
    n_inlier_stars: int = 0
    n_outlier_stars: int = 0
    identified_stars: list = field(default_factory=list)

    # Timing (ms)
    preprocessing_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    feature_extraction_time_ms: float = 0.0
    recognition_time_ms: float = 0.0
    attitude_time_ms: float = 0.0
    total_time_ms: float = 0.0

    error_message: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_navigation(
    image: np.ndarray,
    config: dict,
    catalog_index: CatalogIndex,
    neural_model=None,
) -> NavigationResult:
    """Run complete Phase 1-5 navigation on a preprocessed image.

    Parameters
    ----------
    image : np.ndarray
        2D float32, values in [0, 1]. Already preprocessed.
    config : dict
        Full project configuration dict.
    catalog_index : CatalogIndex
        Pre-built star catalog index.
    neural_model : optional
        Trained classifier for neural prior. None = geometric only.

    Returns
    -------
    NavigationResult
    """
    t_total = time.perf_counter()
    timestamp = time.time()

    try:
        nav_cfg = config.get("navigation", {})
        min_inliers_attitude = int(nav_cfg.get("min_correspondences", 2))

        # ── Step 1: Star detection ────────────────────────────────────────
        t0 = time.perf_counter()
        stars = detect_stars(image, config.get("star_detection", {}))
        detection_ms = (time.perf_counter() - t0) * 1000.0

        # ── Step 2: Neural prior (optional) ──────────────────────────────
        t0 = time.perf_counter()
        neural_result = None
        if neural_model is not None and len(stars) >= 2:
            try:
                features = extract_features(stars, config)
                from src.models.inference import run_inference
                neural_result = run_inference(neural_model, features, config)
            except Exception:
                neural_result = None
        feature_ms = (time.perf_counter() - t0) * 1000.0

        # ── Step 3: Pattern building ──────────────────────────────────────
        t0 = time.perf_counter()
        pattern = build_pattern(stars, config)
        pattern_ms = (time.perf_counter() - t0) * 1000.0

        # ── Step 4: Pattern recognition (Phase 4) ────────────────────────
        t0 = time.perf_counter()
        rec: RecognitionOutput = run_recognition(
            pattern, catalog_index, config, neural_result=neural_result
        )
        recognition_ms = (time.perf_counter() - t0) * 1000.0 + pattern_ms

        # ── Step 5: Attitude estimation (Phase 5) ────────────────────────
        t0 = time.perf_counter()
        att = _attitude_from_recognition(rec, config)
        attitude_ms = (time.perf_counter() - t0) * 1000.0

        # ── Determine navigation status ───────────────────────────────────
        nav_status, att_status = _determine_status(rec, att, config)

        total_ms = (time.perf_counter() - t_total) * 1000.0

        return NavigationResult(
            timestamp=timestamp,
            status=nav_status,
            attitude_status=att_status,
            position_status="UNAVAILABLE",
            velocity_status="UNAVAILABLE",
            quaternion=att.quaternion.copy(),
            rotation_matrix=att.rotation_matrix.copy(),
            euler_angles_deg=att.euler_angles_deg.copy(),
            attitude_confidence=att.attitude_confidence,
            attitude_residual_deg=att.residual_deg,
            max_residual_deg=att.max_residual_deg,
            n_observed_stars=len(stars),
            n_matched_stars=rec.n_matched,
            n_inlier_stars=att.n_inliers if att.is_valid else rec.n_inliers,
            n_outlier_stars=att.n_outliers,
            identified_stars=rec.identified_stars,
            detection_time_ms=detection_ms,
            feature_extraction_time_ms=feature_ms,
            recognition_time_ms=recognition_ms,
            attitude_time_ms=attitude_ms,
            total_time_ms=total_ms,
        )

    except Exception as exc:
        total_ms = (time.perf_counter() - t_total) * 1000.0
        return NavigationResult(
            timestamp=timestamp,
            status="ERROR",
            attitude_status="FAILURE",
            total_time_ms=total_ms,
            error_message=str(exc),
        )


def run_full_pipeline(
    image_path_or_array,
    config: dict,
    catalog_index: CatalogIndex,
    neural_model=None,
) -> NavigationResult:
    """Full pipeline including preprocessing from raw image or array."""
    t_total = time.perf_counter()
    timestamp = time.time()

    try:
        t0 = time.perf_counter()
        if isinstance(image_path_or_array, np.ndarray):
            raw = image_path_or_array.copy()
        else:
            from pathlib import Path
            import cv2
            path = Path(image_path_or_array)
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"Cannot load: {path}")
            raw = raw.astype(np.float32) / 255.0

        image = _preprocess_image(raw, config)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        result = run_navigation(image, config, catalog_index, neural_model)
        result.preprocessing_time_ms = preprocess_ms
        result.total_time_ms = (time.perf_counter() - t_total) * 1000.0
        result.timestamp = timestamp
        return result

    except Exception as exc:
        return NavigationResult(
            timestamp=timestamp,
            status="ERROR",
            attitude_status="FAILURE",
            total_time_ms=(time.perf_counter() - t_total) * 1000.0,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attitude_from_recognition(
    rec: RecognitionOutput,
    config: dict,
) -> AttitudeEstimate:
    """Extract Phase 4 inlier correspondences and run Phase 5 attitude estimation."""
    if not rec.identified_stars or len(rec.identified_stars) < 2:
        return AttitudeEstimate(
            num_correspondences=len(rec.identified_stars),
            is_valid=False,
            attitude_confidence=0.0,
            n_inliers=0,
            n_outliers=len(rec.identified_stars),
        )

    obs_vecs = np.array([s.observed_unit_vec for s in rec.identified_stars])
    cat_vecs = np.array([s.catalog_unit_vec for s in rec.identified_stars])
    # Use Phase 4 per-star confidence as weights
    weights = np.array([max(s.confidence, 1e-6) for s in rec.identified_stars])

    return estimate_attitude(obs_vecs, cat_vecs, config, weights=weights)


def _determine_status(
    rec: RecognitionOutput,
    att: AttitudeEstimate,
    config: dict,
) -> tuple:
    """Return (navigation_status, attitude_status) strings."""
    nav_cfg = config.get("navigation", {})
    min_stars = int(nav_cfg.get("min_correspondences", 2))

    if rec.n_inliers < min_stars:
        return "INSUFFICIENT_STARS", "FAILURE"

    if rec.status == RecognitionStatus.FAILURE:
        return "FAILURE", "FAILURE"

    if not att.is_valid:
        return "ATTITUDE_FAILURE", "FAILURE"

    if rec.status == RecognitionStatus.SUCCESS and att.attitude_confidence >= 0.6:
        return "SUCCESS", "DETERMINED"
    elif rec.status in (RecognitionStatus.PARTIAL, RecognitionStatus.SUCCESS):
        return "PARTIAL", "PARTIAL"
    else:
        return "LOW_CONFIDENCE", "LOW_CONFIDENCE"


def _preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Background subtraction, noise reduction, normalization."""
    from scipy.ndimage import median_filter, gaussian_filter

    pp = config.get("preprocessing", {})
    img = image.astype(np.float32)

    if pp.get("background_subtraction", True):
        method = pp.get("background_method", "median_filter")
        if method == "median_filter":
            bg = median_filter(img, size=int(pp.get("background_filter_size", 31)))
            img = np.clip(img - bg, 0.0, None)
        elif method == "constant":
            img = np.clip(img - float(pp.get("background_level", 0.02)), 0.0, None)

    if pp.get("noise_reduction", True):
        if pp.get("noise_method", "gaussian") == "gaussian":
            img = gaussian_filter(img, sigma=float(pp.get("noise_sigma", 0.8))).astype(np.float32)

    norm = pp.get("normalization", "min_max")
    if norm == "min_max":
        lo, hi = float(img.min()), float(img.max())
        if hi > lo:
            img = (img - lo) / (hi - lo)
    elif norm == "z_score":
        mean, std = float(img.mean()), float(img.std())
        if std > 1e-8:
            img = np.clip((img - mean) / std, 0.0, 1.0)

    return img.astype(np.float32)
