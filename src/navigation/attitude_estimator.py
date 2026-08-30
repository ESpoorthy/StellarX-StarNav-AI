"""
attitude_estimator.py — Phase 5
================================
Spacecraft attitude determination from verified star correspondences.

Algorithm: Wahba's problem solved via weighted SVD (Markley 1988).
  Minimises: sum_i w_i * ||cat_i - R @ obs_i||^2
  Solution:  B = sum_i (w_i * outer(cat_i, obs_i))
             B = U S V^T  (SVD)
             R = U @ diag(1, 1, det(U @ V^T)) @ V^T

Why Wahba/SVD over QUEST or TRIAD:
- Numerically stable, well-studied, widely used in real star trackers
- Directly handles weighted correspondences (Phase 4 confidence scores)
- Produces the globally optimal rotation under Gaussian noise
- Already used in Phase 4 RANSAC refinement — consistent with existing code

Rotation convention (documented explicitly):
  R maps camera-frame vectors to inertial-frame vectors:
    v_inertial = R @ v_camera

Quaternion convention:
  [qw, qx, qy, qz]  (scalar-first)
  v_inertial = q * v_camera * q^{-1}  (quaternion rotation)

Camera frame:
  +X: right  (+image column direction)
  +Y: up     (-image row direction)
  +Z: boresight (pointing into scene)

Inertial frame (J2000 ICRS):
  +X: RA=0°, Dec=0°
  +Y: RA=90°, Dec=0°
  +Z: north celestial pole (Dec=90°)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttitudeEstimate:
    """Spacecraft attitude estimate from star correspondences.

    Attributes
    ----------
    quaternion : np.ndarray
        Unit quaternion [qw, qx, qy, qz], camera→inertial rotation.
        Convention: v_inertial = R @ v_camera.
    rotation_matrix : np.ndarray
        Equivalent 3×3 rotation matrix (camera to inertial).
        Satisfies R^T R = I, det(R) = +1.
    euler_angles_deg : np.ndarray
        [yaw, pitch, roll] in degrees, ZYX convention, for display only.
        Do NOT use Euler-angle differences as attitude error metric.
    residual_deg : float
        Mean angular residual over correspondences used, in degrees.
    num_correspondences : int
        Number of star correspondences used.
    is_valid : bool
        True when quality thresholds are met.
    attitude_confidence : float
        Confidence in [0, 1]. 1 = perfect fit.
    per_star_residuals_deg : np.ndarray
        Per-correspondence angular residuals in degrees. Shape (N,).
    inlier_mask : np.ndarray
        Boolean mask, True = inlier correspondence. Shape (N,).
    n_inliers : int
        Number of inlier correspondences after outlier rejection.
    n_outliers : int
        Number of rejected outlier correspondences.
    max_residual_deg : float
        Maximum per-star angular residual among inliers, degrees.
    """

    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    euler_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    residual_deg: float = float("nan")
    num_correspondences: int = 0
    is_valid: bool = False
    attitude_confidence: float = 0.0
    per_star_residuals_deg: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    inlier_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    n_inliers: int = 0
    n_outliers: int = 0
    max_residual_deg: float = float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_attitude(
    observed_directions: np.ndarray,
    catalog_directions: np.ndarray,
    config: dict,
    weights: Optional[np.ndarray] = None,
) -> AttitudeEstimate:
    """Estimate spacecraft attitude from star direction correspondences.

    Uses Wahba's problem / weighted SVD solution with iterative outlier
    rejection.

    Parameters
    ----------
    observed_directions : np.ndarray
        Shape (N, 3) — unit vectors in the camera/body frame.
    catalog_directions : np.ndarray
        Shape (N, 3) — corresponding unit vectors in the inertial frame.
    config : dict
        Project configuration dict. Reads from config['navigation'].
    weights : np.ndarray, optional
        Shape (N,) non-negative weights. If None, uniform weights used.
        Typically set to Phase 4 per-star confidence scores.

    Returns
    -------
    AttitudeEstimate

    Raises
    ------
    ValueError
        If observed and catalog arrays have different shapes.
    """
    if observed_directions.shape != catalog_directions.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed_directions.shape} "
            f"vs catalog {catalog_directions.shape}"
        )

    nav_cfg = config.get("navigation", {})
    min_corr = int(nav_cfg.get("min_correspondences", 2))
    max_residual_thresh = float(nav_cfg.get("max_residual_threshold_deg", 2.0))
    outlier_thresh = float(nav_cfg.get("outlier_rejection_threshold_deg",
                                        max_residual_thresh))
    max_iter = int(nav_cfg.get("outlier_rejection_max_iter", 3))

    n = len(observed_directions)

    if n < min_corr:
        return AttitudeEstimate(
            num_correspondences=n,
            is_valid=False,
            attitude_confidence=0.0,
            n_inliers=0,
            n_outliers=n,
        )

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).copy()
        weights = np.clip(weights, 1e-9, None)

    # Iterative outlier rejection + Wahba/SVD
    active = np.ones(n, dtype=bool)

    R = None
    for iteration in range(max_iter + 1):
        idx = np.where(active)[0]
        if len(idx) < min_corr:
            break

        obs_a = observed_directions[idx]
        cat_a = catalog_directions[idx]
        w_a = weights[idx]

        R_new = _wahba_svd(obs_a, cat_a, w_a)
        if R_new is None:
            break
        R = R_new

        # Compute per-correspondence residuals for all active
        residuals = _compute_residuals(R, obs_a, cat_a)

        # Reject outliers beyond threshold (not on final iteration)
        if iteration < max_iter:
            inlier_local = residuals <= outlier_thresh
            if inlier_local.sum() < min_corr:
                break
            # Update active mask
            active[idx] = inlier_local

    if R is None:
        return AttitudeEstimate(
            num_correspondences=n,
            is_valid=False,
            attitude_confidence=0.0,
            n_inliers=0,
            n_outliers=n,
        )

    # Final residuals for ALL correspondences
    all_residuals = _compute_residuals(R, observed_directions, catalog_directions)

    # Inlier mask based on final outlier threshold
    inlier_mask = all_residuals <= outlier_thresh
    n_inliers = int(inlier_mask.sum())
    n_outliers = n - n_inliers

    mean_residual = float(np.mean(all_residuals[inlier_mask])) if n_inliers > 0 else float("nan")
    max_residual = float(np.max(all_residuals[inlier_mask])) if n_inliers > 0 else float("nan")

    # Validate rotation matrix
    if not _is_valid_rotation(R):
        return AttitudeEstimate(
            num_correspondences=n,
            is_valid=False,
            attitude_confidence=0.0,
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            per_star_residuals_deg=all_residuals,
            inlier_mask=inlier_mask,
        )

    q = rotation_matrix_to_quaternion(R)
    euler = rotation_matrix_to_euler_deg(R)

    confidence = float(np.clip(
        1.0 - mean_residual / max(max_residual_thresh, 1e-9),
        0.0, 1.0
    )) if not math.isnan(mean_residual) else 0.0

    is_valid = (
        n_inliers >= min_corr
        and not math.isnan(mean_residual)
        and mean_residual < max_residual_thresh
    )

    return AttitudeEstimate(
        quaternion=q,
        rotation_matrix=R,
        euler_angles_deg=euler,
        residual_deg=mean_residual,
        num_correspondences=n,
        is_valid=is_valid,
        attitude_confidence=confidence,
        per_star_residuals_deg=all_residuals,
        inlier_mask=inlier_mask,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        max_residual_deg=max_residual,
    )


def estimate_attitude_weighted(
    observed_directions: np.ndarray,
    catalog_directions: np.ndarray,
    weights: np.ndarray,
    config: dict,
) -> AttitudeEstimate:
    """Convenience wrapper — passes weights as positional arg."""
    return estimate_attitude(observed_directions, catalog_directions, config, weights)


def angular_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angular error between two rotation matrices on SO(3).

    theta = arccos( (trace(R1^T @ R2) - 1) / 2 )

    This is the correct primary metric for attitude error.
    Do NOT use Euler-angle differences.

    Parameters
    ----------
    R1, R2 : np.ndarray
        Shape (3, 3) rotation matrices.

    Returns
    -------
    float
        Angular error in degrees in [0, 180].
    """
    R_rel = R1.T @ R2
    cos_angle = (float(np.trace(R_rel)) - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def validate_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> dict:
    """Check that R is a valid rotation matrix.

    Parameters
    ----------
    R : np.ndarray  Shape (3, 3).
    tol : float     Tolerance for orthogonality and determinant checks.

    Returns
    -------
    dict  Keys: 'is_valid', 'orthogonality_error', 'determinant', 'det_error'.
    """
    RtR = R.T @ R
    orth_err = float(np.max(np.abs(RtR - np.eye(3))))
    det = float(np.linalg.det(R))
    det_err = abs(det - 1.0)
    return {
        "is_valid": orth_err < tol and det_err < tol,
        "orthogonality_error": orth_err,
        "determinant": det,
        "det_error": det_err,
    }


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities
# ---------------------------------------------------------------------------


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to unit quaternion [qw, qx, qy, qz].

    Uses Shepperd method for numerical stability.
    """
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    return q


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qw, qx, qy, qz] to 3×3 rotation matrix."""
    q = q / max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def rotation_matrix_to_euler_deg(R: np.ndarray) -> np.ndarray:
    """ZYX Euler angles [yaw, pitch, roll] in degrees from rotation matrix.

    Convention: R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    For DISPLAY ONLY — do not use for attitude error computation.
    """
    sin_pitch = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = math.degrees(math.asin(sin_pitch))
    cos_pitch = math.cos(math.asin(sin_pitch))

    if abs(cos_pitch) > 1e-6:
        yaw = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
        roll = math.degrees(math.atan2(float(R[2, 1]), float(R[2, 2])))
    else:
        # Gimbal lock
        yaw = math.degrees(math.atan2(float(-R[0, 1]), float(R[1, 1])))
        roll = 0.0

    return np.array([yaw, pitch, roll], dtype=np.float64)


def quaternion_sign_canonical(q: np.ndarray) -> np.ndarray:
    """Return canonical quaternion with qw >= 0 (resolve sign equivalence).

    q and -q represent the same rotation. This function ensures a
    consistent sign convention for comparison.
    """
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wahba_svd(
    obs_vecs: np.ndarray,
    cat_vecs: np.ndarray,
    weights: np.ndarray,
) -> Optional[np.ndarray]:
    """Solve Wahba's problem via SVD.

    B = sum_i (w_i * outer(cat_i, obs_i))
    B = U S V^T
    R = U @ diag(1, 1, det(U @ V^T)) @ V^T
    """
    if len(obs_vecs) == 0:
        return None
    try:
        B = np.zeros((3, 3), dtype=np.float64)
        for i in range(len(obs_vecs)):
            B += weights[i] * np.outer(cat_vecs[i], obs_vecs[i])
        U, S, Vt = np.linalg.svd(B)
        det = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0, 1.0, det]) @ Vt
        return R
    except np.linalg.LinAlgError:
        return None


def _compute_residuals(
    R: np.ndarray,
    obs_vecs: np.ndarray,
    cat_vecs: np.ndarray,
) -> np.ndarray:
    """Angular residuals in degrees: arccos(dot(R@obs, cat)) for each pair."""
    residuals = np.zeros(len(obs_vecs), dtype=np.float64)
    for i in range(len(obs_vecs)):
        pred = R @ obs_vecs[i]
        norm = np.linalg.norm(pred)
        if norm > 1e-12:
            pred /= norm
        dot = float(np.dot(pred, cat_vecs[i]))
        dot = max(-1.0, min(1.0, dot))
        residuals[i] = math.degrees(math.acos(dot))
    return residuals


def _is_valid_rotation(R: np.ndarray, tol: float = 1e-5) -> bool:
    """Check R^T R ≈ I and det(R) ≈ +1."""
    v = validate_rotation_matrix(R, tol=tol)
    return bool(v["is_valid"])
