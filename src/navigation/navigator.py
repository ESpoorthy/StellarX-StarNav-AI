"""
navigator.py — Phase 5
======================
End-to-end navigation pipeline combining Phase 4 (recognition) and Phase 5 (attitude).

Orchestrates:
  1. Star detection
  2. Feature extraction for neural prior
  3. Neural inference (if model provided)
  4. Pattern building
  5. Pattern recognition / catalog matching
  6. Attitude estimation from inlier correspondences
  7. Position note (single-image position not available)

All failure cases are handled gracefully (zero stars, no match, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.navigation.attitude_estimator import AttitudeEstimate, estimate_attitude
from src.preprocessing.star_detection import detect_stars, extract_features
from src.recognition.catalog_index import CatalogIndex
from src.recognition.pattern_builder import build_pattern
from src.recognition.pattern_matcher import RecognitionOutput, RecognitionStatus, run_recognition


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NavigationResult:
    """Complete output from one navigation pipeline run.

    Attributes
    ----------
    timestamp : float
        Unix timestamp of when the result was produced, seconds.
    status : str
        "SUCCESS", "PARTIAL", "LOW_CONFIDENCE", "FAILURE", or "ERROR".
    quaternion : np.ndarray
        [qw, qx, qy, qz], rotation from camera frame to inertial frame.
    rotation_matrix : np.ndarray
        3x3 rotation matrix (camera to inertial).
    euler_angles_deg : np.ndarray
        [yaw, pitch, roll] in ZYX convention, degrees.
    attitude_confidence : float
        Confidence in attitude estimate, in [0, 1].
    attitude_residual_deg : float
        Mean angular residual of the attitude fit, degrees.
    position_note : str
        Explanation of why position is not estimated from a single image.
    n_observed_stars : int
        Number of stars detected in the image.
    n_matched_stars : int
        Number of stars with tentative catalog matches.
    n_inlier_stars : int
        Number of RANSAC inlier correspondences.
    identified_stars : list
        List of IdentifiedStar from recognition.
    preprocessing_time_ms : float
        Time for star detection in milliseconds.
    detection_time_ms : float
        Time for star detection step in milliseconds.
    feature_extraction_time_ms : float
        Time for feature extraction in milliseconds.
    recognition_time_ms : float
        Time for catalog matching in milliseconds.
    attitude_time_ms : float
        Time for attitude estimation in milliseconds.
    total_time_ms : float
        Total pipeline wall-clock time in milliseconds.
    error_message : str
        Error description if status is "ERROR".
    """

    timestamp: float = 0.0
    status: str = "FAILURE"

    # Attitude output
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    euler_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude_confidence: float = 0.0
    attitude_residual_deg: float = float("nan")

    # Position (not estimated from single image)
    position_note: str = (
        "Position estimation requires multi-image data, orbital mechanics, "
        "or additional sensors. Single-image star tracking provides attitude only."
    )

    # Matched stars
    n_observed_stars: int = 0
    n_matched_stars: int = 0
    n_inlier_stars: int = 0
    identified_stars: list = field(default_factory=list)

    # Performance timing (milliseconds)
    preprocessing_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    feature_extraction_time_ms: float = 0.0
    recognition_time_ms: float = 0.0
    attitude_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Error info
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
    """Run the complete navigation pipeline on a preprocessed image.

    Parameters
    ----------
    image : np.ndarray
        2D float32 image array, values in [0, 1]. Should already be
        preprocessed (background-subtracted, noise-reduced, normalized).
    config : dict
        Project configuration dict.
    catalog_index : CatalogIndex
        Pre-built indexed star catalog.
    neural_model : optional
        Trained StarPatternClassifier or similar, for neural prior.
        If None, only geometric matching is used.

    Returns
    -------
    NavigationResult
        Complete result including attitude and performance metrics.
    """
    t_total_start = time.perf_counter()
    timestamp = time.time()

    try:
        # Step 1: Star detection
        t0 = time.perf_counter()
        star_detection_cfg = config.get("star_detection", {})
        stars = detect_stars(image, star_detection_cfg)
        detection_time_ms = (time.perf_counter() - t0) * 1000.0

        n_observed = len(stars)

        # Step 2: Feature extraction for neural prior
        t0 = time.perf_counter()
        neural_result = None
        if neural_model is not None and len(stars) >= 2:
            try:
                features = extract_features(stars, config)
                from src.models.inference import run_inference
                neural_result = run_inference(neural_model, features, config)
            except Exception:
                neural_result = None
        feature_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: Pattern building
        t0 = time.perf_counter()
        pattern = build_pattern(stars, config)
        pattern_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 4: Pattern recognition / catalog matching
        t0 = time.perf_counter()
        rec_output: RecognitionOutput = run_recognition(
            pattern, catalog_index, config, neural_result=neural_result
        )
        recognition_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 5: Attitude estimation from inlier correspondences
        t0 = time.perf_counter()
        attitude_estimate = _estimate_attitude_from_recognition(rec_output, config)
        attitude_time_ms = (time.perf_counter() - t0) * 1000.0

        # Determine navigation status
        nav_status = rec_output.status.value  # "SUCCESS", "PARTIAL", etc.

        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0

        return NavigationResult(
            timestamp=timestamp,
            status=nav_status,
            quaternion=attitude_estimate.quaternion.copy(),
            rotation_matrix=attitude_estimate.rotation_matrix.copy(),
            euler_angles_deg=attitude_estimate.euler_angles_deg.copy(),
            attitude_confidence=attitude_estimate.attitude_confidence,
            attitude_residual_deg=attitude_estimate.residual_deg,
            n_observed_stars=n_observed,
            n_matched_stars=rec_output.n_matched,
            n_inlier_stars=rec_output.n_inliers,
            identified_stars=rec_output.identified_stars,
            preprocessing_time_ms=0.0,  # image already preprocessed
            detection_time_ms=detection_time_ms,
            feature_extraction_time_ms=feature_time_ms,
            recognition_time_ms=recognition_time_ms + pattern_time_ms,
            attitude_time_ms=attitude_time_ms,
            total_time_ms=total_time_ms,
        )

    except Exception as exc:
        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
        return NavigationResult(
            timestamp=timestamp,
            status="ERROR",
            total_time_ms=total_time_ms,
            error_message=str(exc),
        )


def run_full_pipeline(
    image_path_or_array,
    config: dict,
    catalog_index: CatalogIndex,
    neural_model=None,
) -> NavigationResult:
    """Full pipeline from raw image file or array to NavigationResult.

    Includes preprocessing (background subtraction, noise reduction,
    normalization) before running the navigation pipeline.

    Parameters
    ----------
    image_path_or_array : str, Path, or np.ndarray
        Path to an image file or a pre-loaded numpy array.
    config : dict
        Project configuration dict.
    catalog_index : CatalogIndex
        Pre-built indexed star catalog.
    neural_model : optional
        Trained neural model for prior.

    Returns
    -------
    NavigationResult
        Complete navigation result.
    """
    t_total_start = time.perf_counter()
    timestamp = time.time()

    try:
        # Load image if path provided
        t0 = time.perf_counter()
        if isinstance(image_path_or_array, np.ndarray):
            raw_image = image_path_or_array.copy()
        else:
            from pathlib import Path
            import cv2
            img_path = Path(image_path_or_array)
            raw_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if raw_image is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
            raw_image = raw_image.astype(np.float32) / 255.0

        # Preprocessing: background subtraction, noise reduction, normalization
        image = _preprocess_image(raw_image, config)
        preprocessing_time_ms = (time.perf_counter() - t0) * 1000.0

        # Run navigation on preprocessed image
        result = run_navigation(image, config, catalog_index, neural_model)

        # Update preprocessing time and total time
        result.preprocessing_time_ms = preprocessing_time_ms
        result.total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
        result.timestamp = timestamp

        return result

    except Exception as exc:
        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
        return NavigationResult(
            timestamp=timestamp,
            status="ERROR",
            total_time_ms=total_time_ms,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_attitude_from_recognition(
    rec_output: RecognitionOutput,
    config: dict,
) -> AttitudeEstimate:
    """Extract correspondences from RecognitionOutput and estimate attitude.

    Parameters
    ----------
    rec_output : RecognitionOutput
        Output from run_recognition().
    config : dict
        Project configuration dict.

    Returns
    -------
    AttitudeEstimate
    """
    from src.navigation.attitude_estimator import AttitudeEstimate

    if len(rec_output.identified_stars) < 2:
        return AttitudeEstimate(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3),
            euler_angles_deg=np.zeros(3),
            residual_deg=float("nan"),
            num_correspondences=len(rec_output.identified_stars),
            is_valid=False,
            attitude_confidence=0.0,
        )

    obs_vecs = np.array([s.observed_unit_vec for s in rec_output.identified_stars])
    cat_vecs = np.array([s.catalog_unit_vec for s in rec_output.identified_stars])
    weights = np.array([max(s.confidence, 1e-6) for s in rec_output.identified_stars])

    return estimate_attitude(obs_vecs, cat_vecs, config, weights=weights)


def _preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Apply background subtraction, noise reduction, and normalization.

    Parameters
    ----------
    image : np.ndarray
        Raw 2D float32 image, values in [0, 1].
    config : dict
        Project configuration dict. Reads from config['preprocessing'].

    Returns
    -------
    np.ndarray
        Preprocessed float32 image.
    """
    from scipy.ndimage import median_filter, gaussian_filter

    pp_cfg = config.get("preprocessing", {})
    result = image.astype(np.float32)

    # Background subtraction
    if pp_cfg.get("background_subtraction", True):
        method = pp_cfg.get("background_method", "median_filter")
        if method == "median_filter":
            filter_size = int(pp_cfg.get("background_filter_size", 31))
            bg = median_filter(result, size=filter_size)
            result = result - bg
            result = np.clip(result, 0.0, None)
        elif method == "constant":
            bg_level = float(pp_cfg.get("background_level", 0.02))
            result = np.clip(result - bg_level, 0.0, None)

    # Noise reduction
    if pp_cfg.get("noise_reduction", True):
        method = pp_cfg.get("noise_method", "gaussian")
        if method == "gaussian":
            sigma = float(pp_cfg.get("noise_sigma", 0.8))
            result = gaussian_filter(result, sigma=sigma).astype(np.float32)

    # Normalization
    normalization = pp_cfg.get("normalization", "min_max")
    if normalization == "min_max":
        lo = float(result.min())
        hi = float(result.max())
        if hi > lo:
            result = (result - lo) / (hi - lo)
    elif normalization == "z_score":
        mean = float(result.mean())
        std = float(result.std())
        if std > 1e-8:
            result = (result - mean) / std
            result = np.clip(result, 0.0, 1.0)

    return result.astype(np.float32)
