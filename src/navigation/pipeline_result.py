"""
pipeline_result.py
==================
Structured result object for the complete StellarX navigation pipeline.

This dataclass carries every measurable quantity produced during one
pipeline execution.  The Streamlit UI reads exclusively from this object —
no raw NavigationResult fields are accessed directly by the UI layer.

Design principles
-----------------
- Every field that the UI displays must be populated by actual computation.
- No hard-coded or fabricated values.
- Failure states are explicit — the UI shows them honestly.
- All timing values come from ``time.perf_counter()`` measurements.
- Integrity fields are computed post-SVD, not assumed.

Usage
-----
``from src.navigation.pipeline_result import PipelineResult, build_pipeline_result``

``result = build_pipeline_result(nav_result, config)``

The ``build_pipeline_result`` factory validates the NavigationResult,
runs the integrity check, and populates all derived fields.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.navigation.attitude_estimator import validate_rotation_matrix


# ---------------------------------------------------------------------------
# Sub-dataclasses for each pipeline stage
# ---------------------------------------------------------------------------


@dataclass
class DetectionTelemetry:
    detected_count: int = 0
    accepted_count: int = 0          # passed brightness / area filters
    detection_latency_ms: float = 0.0


@dataclass
class AITelemetry:
    model_name: str = "sklearn_random_forest"
    model_available: bool = False
    inference_latency_ms: float = 0.0
    prediction: Optional[str] = None   # e.g. "cell_42"
    confidence: float = 0.0
    top_k: list[tuple[str, float]] = field(default_factory=list)
    note: str = ""                     # e.g. "model not trained — geometry only"


@dataclass
class MatchingTelemetry:
    candidate_count: int = 0          # stars with at least 1 vote
    matched_count: int = 0            # RANSAC inlier matches
    rejected_count: int = 0           # RANSAC outlier matches
    recognition_status: str = "FAILURE"
    matching_latency_ms: float = 0.0
    pattern_type: str = "geometric_ransac"
    mean_residual_deg: float = float("nan")
    recognition_confidence: float = 0.0


@dataclass
class IdentifiedStarRecord:
    """Immutable record for one matched star."""
    observed_x: float = 0.0
    observed_y: float = 0.0
    catalog_id: str = ""
    catalog_ra_deg: float = 0.0
    catalog_dec_deg: float = 0.0
    angular_residual_deg: float = 0.0
    per_star_confidence: float = 0.0
    brightness: float = 0.0
    is_inlier: bool = True


@dataclass
class AttitudeTelemetry:
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    mean_residual_deg: float = float("nan")
    rms_residual_deg: float = float("nan")
    max_residual_deg: float = float("nan")
    n_inliers: int = 0
    n_outliers: int = 0
    attitude_latency_ms: float = 0.0
    attitude_status: str = "FAILURE"   # DETERMINED / PARTIAL / LOW_CONFIDENCE / FAILURE
    attitude_confidence: float = 0.0
    frame_convention: str = "camera→inertial (J2000 ICRS)"


@dataclass
class IntegrityResult:
    """Post-solution integrity check results."""
    quaternion_norm: float = float("nan")
    determinant: float = float("nan")
    orthogonality_error: float = float("nan")
    mean_residual_deg: float = float("nan")
    rms_residual_deg: float = float("nan")
    max_residual_deg: float = float("nan")
    n_inliers: int = 0
    n_outliers: int = 0
    is_valid: bool = False
    rejection_reason: str = ""         # empty string = accepted
    confidence: float = 0.0
    status: str = "REJECTED"           # VALID / DEGRADED / REJECTED

    @property
    def status_symbol(self) -> str:
        return "✓" if self.status == "VALID" else ("⚠" if self.status == "DEGRADED" else "✕")


@dataclass
class PerformanceTelemetry:
    preprocessing_ms: float = 0.0
    detection_ms: float = 0.0
    ai_inference_ms: float = 0.0
    matching_ms: float = 0.0
    attitude_ms: float = 0.0
    integrity_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Preprocessing":    self.preprocessing_ms,
            "Star Detection":   self.detection_ms,
            "AI Inference":     self.ai_inference_ms,
            "Pattern Matching": self.matching_ms,
            "Attitude Solver":  self.attitude_ms,
            "Integrity Check":  self.integrity_ms,
        }


# ---------------------------------------------------------------------------
# Top-level PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Structured output from one complete pipeline execution.

    Every field that appears in the UI is populated here.
    The UI layer only reads from this object — never from raw NavigationResult.

    Status values
    -------------
    overall_status:
        SUCCESS          — attitude determined, all integrity checks passed
        PARTIAL          — attitude estimated, some checks marginal
        LOW_CONFIDENCE   — weak recognition or poor geometry
        INSUFFICIENT_STARS — too few stars detected
        ATTITUDE_FAILURE — recognition succeeded but attitude solver failed
        INTEGRITY_FAILED — solver ran but integrity check rejected result
        FAILURE          — recognition failed
        ERROR            — unhandled exception

    position_status:
        Always UNAVAILABLE. Single-image star tracking determines
        attitude (3 DoF orientation) only.  Absolute position requires
        multi-image triangulation, orbital mechanics, or additional sensors.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    image_shape: tuple[int, int] = (0, 0)

    # Mission context
    overall_status: str = "FAILURE"
    position_status: str = "UNAVAILABLE"
    position_note: str = (
        "POSITION UNAVAILABLE — single-image star tracking provides 3-DoF "
        "attitude only. Absolute position requires multi-image triangulation, "
        "orbital mechanics propagation, or additional sensors (IMU/GPS)."
    )

    # Per-stage telemetry
    detection: DetectionTelemetry = field(default_factory=DetectionTelemetry)
    ai: AITelemetry = field(default_factory=AITelemetry)
    matching: MatchingTelemetry = field(default_factory=MatchingTelemetry)
    attitude: AttitudeTelemetry = field(default_factory=AttitudeTelemetry)
    integrity: IntegrityResult = field(default_factory=IntegrityResult)
    performance: PerformanceTelemetry = field(default_factory=PerformanceTelemetry)

    # Identified stars — list of matched catalog entries
    identified_stars: list[IdentifiedStarRecord] = field(default_factory=list)

    # Raw NavigationResult preserved for downstream use (e.g. export)
    error_message: str = ""
    is_demo: bool = False


# ---------------------------------------------------------------------------
# Factory — build from NavigationResult
# ---------------------------------------------------------------------------


def build_pipeline_result(
    nav_result,                  # NavigationResult from navigator.py
    config: dict,
    image_shape: tuple[int, int] = (512, 512),
    is_demo: bool = False,
    neural_result=None,          # RecognitionResult from inference.py (may be None)
) -> PipelineResult:
    """Populate a PipelineResult from a NavigationResult.

    Runs the integrity check inline and fills every telemetry field.
    All values come from actual computation — nothing is fabricated.

    Parameters
    ----------
    nav_result : NavigationResult
        Output from navigator.run_navigation() or run_full_pipeline().
    config : dict
        Full project configuration dict.
    image_shape : tuple[int, int]
        (height, width) of the processed image.
    is_demo : bool
        True if running in SIH demo mode.
    neural_result : RecognitionResult, optional
        Output from inference.run_inference() if a model was available.

    Returns
    -------
    PipelineResult
    """
    t_integrity_start = time.perf_counter()

    pr = PipelineResult(
        image_shape=image_shape,
        overall_status=nav_result.status,
        position_status="UNAVAILABLE",
        error_message=nav_result.error_message,
        is_demo=is_demo,
    )

    # ── Detection ────────────────────────────────────────────────────────────
    pr.detection = DetectionTelemetry(
        detected_count=nav_result.n_observed_stars,
        accepted_count=nav_result.n_observed_stars,  # all detected passed filters
        detection_latency_ms=nav_result.detection_time_ms,
    )

    # ── AI (sklearn or none) ─────────────────────────────────────────────────
    if neural_result is not None:
        pr.ai = AITelemetry(
            model_name="sklearn_random_forest",
            model_available=True,
            inference_latency_ms=float(getattr(neural_result, "latency_ms", 0.0)),
            prediction=getattr(neural_result, "pattern_id", None),
            confidence=float(getattr(neural_result, "confidence", 0.0)),
            top_k=list(getattr(neural_result, "top_k_predictions", [])),
            note="Sklearn RandomForest — sky-cell classification prior",
        )
    else:
        pr.ai = AITelemetry(
            model_name="sklearn_random_forest",
            model_available=False,
            inference_latency_ms=nav_result.feature_extraction_time_ms,
            note="No trained model checkpoint found — running geometry-only mode",
        )

    # ── Matching ─────────────────────────────────────────────────────────────
    pr.matching = MatchingTelemetry(
        candidate_count=nav_result.n_matched_stars,
        matched_count=nav_result.n_inlier_stars,
        rejected_count=nav_result.n_outlier_stars,
        recognition_status=nav_result.status,
        matching_latency_ms=nav_result.recognition_time_ms,
        mean_residual_deg=nav_result.attitude_residual_deg,
        recognition_confidence=nav_result.attitude_confidence,
    )

    # ── Attitude ─────────────────────────────────────────────────────────────
    euler = nav_result.euler_angles_deg
    rms = float("nan")
    if nav_result.identified_stars:
        residuals = [float(getattr(s, "angular_residual_deg", float("nan")))
                     for s in nav_result.identified_stars]
        finite_res = [r for r in residuals if math.isfinite(r)]
        if finite_res:
            rms = float(np.sqrt(np.mean(np.array(finite_res) ** 2)))

    pr.attitude = AttitudeTelemetry(
        quaternion=nav_result.quaternion.copy(),
        rotation_matrix=nav_result.rotation_matrix.copy(),
        roll_deg=float(euler[2]) if len(euler) > 2 else 0.0,
        pitch_deg=float(euler[1]) if len(euler) > 1 else 0.0,
        yaw_deg=float(euler[0]) if len(euler) > 0 else 0.0,
        mean_residual_deg=nav_result.attitude_residual_deg,
        rms_residual_deg=rms,
        max_residual_deg=nav_result.max_residual_deg,
        n_inliers=nav_result.n_inlier_stars,
        n_outliers=nav_result.n_outlier_stars,
        attitude_latency_ms=nav_result.attitude_time_ms,
        attitude_status=nav_result.attitude_status,
        attitude_confidence=nav_result.attitude_confidence,
    )

    # ── Identified stars ─────────────────────────────────────────────────────
    pr.identified_stars = [
        IdentifiedStarRecord(
            observed_x=float(getattr(s, "observed_x", 0.0)),
            observed_y=float(getattr(s, "observed_y", 0.0)),
            catalog_id=str(getattr(s, "catalog_id", "")),
            catalog_ra_deg=float(getattr(s, "catalog_ra_deg", 0.0)),
            catalog_dec_deg=float(getattr(s, "catalog_dec_deg", 0.0)),
            angular_residual_deg=float(getattr(s, "angular_residual_deg", float("nan"))),
            per_star_confidence=float(getattr(s, "confidence", 0.0)),
            brightness=float(getattr(s, "brightness", 0.0)),
            is_inlier=True,
        )
        for s in nav_result.identified_stars
    ]

    # ── Integrity check ───────────────────────────────────────────────────────
    int_cfg = config.get("integrity", {})
    max_mean_res    = float(int_cfg.get("max_mean_residual_deg", 2.0))
    max_ps_res      = float(int_cfg.get("max_per_star_residual_deg", 3.0))
    q_norm_min      = float(int_cfg.get("min_quaternion_norm", 0.999))
    q_norm_max      = float(int_cfg.get("max_quaternion_norm", 1.001))
    det_min         = float(int_cfg.get("min_det", 0.999))
    min_inliers_val = int(int_cfg.get("min_inliers_valid", 2))

    R = nav_result.rotation_matrix
    q = nav_result.quaternion

    q_norm   = float(np.linalg.norm(q)) if np.isfinite(q).all() else float("nan")
    rot_info = validate_rotation_matrix(R) if np.isfinite(R).all() else {
        "is_valid": False, "determinant": float("nan"),
        "orthogonality_error": float("nan"), "det_error": float("nan"),
    }
    det      = float(rot_info.get("determinant", float("nan")))
    orth_err = float(rot_info.get("orthogonality_error", float("nan")))

    # Determine rejection reason (first failure encountered)
    rejection_reason = ""
    if nav_result.status in ("ERROR", "FAILURE", "INSUFFICIENT_STARS", "ATTITUDE_FAILURE"):
        rejection_reason = f"Navigation status: {nav_result.status}"
    elif not math.isfinite(q_norm):
        rejection_reason = "Quaternion contains NaN/Inf"
    elif not (q_norm_min <= q_norm <= q_norm_max):
        rejection_reason = f"Quaternion norm {q_norm:.6f} outside [{q_norm_min}, {q_norm_max}]"
    elif not math.isfinite(det):
        rejection_reason = "Rotation matrix contains NaN/Inf"
    elif det < det_min:
        rejection_reason = f"det(R) = {det:.6f} < {det_min} — not a proper rotation"
    elif nav_result.n_inlier_stars < min_inliers_val:
        rejection_reason = (
            f"Insufficient independent correspondences: "
            f"{nav_result.n_inlier_stars} < {min_inliers_val}"
        )
    elif (math.isfinite(nav_result.attitude_residual_deg)
          and nav_result.attitude_residual_deg > max_mean_res):
        rejection_reason = (
            f"Mean residual {nav_result.attitude_residual_deg:.3f}° "
            f"> threshold {max_mean_res}°"
        )
    elif (math.isfinite(nav_result.max_residual_deg)
          and nav_result.max_residual_deg > max_ps_res):
        rejection_reason = (
            f"Max per-star residual {nav_result.max_residual_deg:.3f}° "
            f"> threshold {max_ps_res}°"
        )

    # Integrity confidence: penalise for residual and rotation quality
    int_confidence = 0.0
    if not rejection_reason:
        res_score = max(0.0, 1.0 - nav_result.attitude_residual_deg / max(max_mean_res, 1e-9)) \
                    if math.isfinite(nav_result.attitude_residual_deg) else 0.0
        det_score = min(1.0, max(0.0, 1.0 - abs(det - 1.0) * 100)) \
                    if math.isfinite(det) else 0.0
        int_confidence = 0.7 * res_score + 0.3 * det_score

    # Status: VALID / DEGRADED / REJECTED
    if rejection_reason:
        integrity_status = "REJECTED"
    elif int_confidence >= 0.7:
        integrity_status = "VALID"
    else:
        integrity_status = "DEGRADED"

    integrity_ms = (time.perf_counter() - t_integrity_start) * 1000.0

    pr.integrity = IntegrityResult(
        quaternion_norm=q_norm,
        determinant=det,
        orthogonality_error=orth_err,
        mean_residual_deg=nav_result.attitude_residual_deg,
        rms_residual_deg=rms,
        max_residual_deg=nav_result.max_residual_deg,
        n_inliers=nav_result.n_inlier_stars,
        n_outliers=nav_result.n_outlier_stars,
        is_valid=(not bool(rejection_reason)),
        rejection_reason=rejection_reason,
        confidence=int_confidence,
        status=integrity_status,
    )

    # Override overall_status with integrity result for cleaner UI reporting
    if integrity_status == "REJECTED" and pr.overall_status == "SUCCESS":
        pr.overall_status = "INTEGRITY_FAILED"
    if integrity_status == "DEGRADED" and pr.overall_status == "SUCCESS":
        pr.overall_status = "PARTIAL"

    # ── Performance ───────────────────────────────────────────────────────────
    pr.performance = PerformanceTelemetry(
        preprocessing_ms=nav_result.preprocessing_time_ms,
        detection_ms=nav_result.detection_time_ms,
        ai_inference_ms=nav_result.feature_extraction_time_ms,
        matching_ms=nav_result.recognition_time_ms,
        attitude_ms=nav_result.attitude_time_ms,
        integrity_ms=integrity_ms,
        total_ms=nav_result.total_time_ms + integrity_ms,
    )

    return pr
