"""
attitude_estimator.py
=====================
Responsible for computing spacecraft attitude (orientation) from a verified
star catalog match.

Responsibility (planned)
------------------------
- Accept star correspondences: observed directions (camera frame) paired
  with catalog directions (inertial frame).
- Solve the attitude determination problem to produce a rotation estimate.
- Return the attitude as a quaternion or rotation matrix, along with an
  uncertainty or residual estimate.

Implementation note
-------------------
The estimation algorithm is to be selected during Phase 5.
Candidates include QUEST, the Davenport q-method, and SVD-based solvers.
All parameters must be sourced from config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttitudeEstimate:
    """Spacecraft attitude estimate.

    Attributes
    ----------
    quaternion : np.ndarray
        Unit quaternion [qw, qx, qy, qz] representing the rotation from
        the inertial frame to the camera/body frame.
    rotation_matrix : np.ndarray
        Equivalent 3×3 rotation matrix.
    residual_deg : float
        Mean angular residual of the fit in degrees.
    num_correspondences : int
        Number of star correspondences used in the estimate.
    is_valid : bool
        True if the estimate meets the configured quality threshold.
    """

    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    rotation_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )
    residual_deg: float = float("nan")
    num_correspondences: int = 0
    is_valid: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_attitude(
    observed_directions: np.ndarray,
    catalog_directions: np.ndarray,
    config: dict,
) -> AttitudeEstimate:
    """Estimate spacecraft attitude from star direction correspondences.

    Parameters
    ----------
    observed_directions:
        Array of shape (N, 3) — unit vectors in the camera/body frame,
        one per matched star.
    catalog_directions:
        Array of shape (N, 3) — corresponding unit vectors in the
        inertial (J2000) frame, from the star catalog.
    config:
        Navigation configuration dict (``navigation`` section of config.yaml).

    Returns
    -------
    AttitudeEstimate
        Estimated attitude with quality metrics.

    Raises
    ------
    ValueError
        If *observed_directions* and *catalog_directions* have different shapes.
    NotImplementedError
        Until this function is implemented in Phase 5.
    """
    if observed_directions.shape != catalog_directions.shape:
        raise ValueError(
            f"observed_directions shape {observed_directions.shape} does not match "
            f"catalog_directions shape {catalog_directions.shape}."
        )

    # TODO (Phase 5): implement attitude estimation.
    #   Algorithm candidates: QUEST, Davenport q-method, SVD/Wahba solution.
    raise NotImplementedError("estimate_attitude is not yet implemented.")
