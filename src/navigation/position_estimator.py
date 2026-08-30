"""
position_estimator.py — Phase 5
=================================
Spacecraft position estimation.

Scientific design decision
--------------------------
A single star-field image determines ATTITUDE (3 DoF orientation), not
absolute POSITION (3 DoF location in space).

Stars are effectively at infinite distance. Their observed angular positions
give camera pointing direction (attitude) but carry no distance information.
To determine position you need one of:
  1. Multi-image baseline triangulation (known separation between images)
  2. Orbital mechanics propagation from a known initial state
  3. Additional sensors (GPS, IMU, LIDAR, planetary limb sensing)
  4. Near-body navigation (planetary/lunar limb crossing)

This module returns PositionEstimate(is_valid=False) with an honest
explanation. The architecture is designed so that a future phase can
integrate orbital mechanics or multi-image data without changing the
function signature.

References
----------
Markley, F.L. & Crassidis, J.L. (2014). Fundamentals of Spacecraft
Attitude Determination and Control. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.navigation.attitude_estimator import AttitudeEstimate

_POSITION_UNAVAILABLE_NOTE = (
    "POSITION UNAVAILABLE: Single-image star tracking determines spacecraft "
    "ATTITUDE (3 DoF orientation) only. Absolute position requires "
    "multi-image triangulation, orbital mechanics propagation, or additional "
    "sensors (IMU, GPS, planetary limb). "
    "This is scientifically correct — star directions are angular measurements "
    "with no distance information."
)


@dataclass
class PositionEstimate:
    """Spacecraft position estimate.

    Attributes
    ----------
    position_vector : np.ndarray
        Shape (3,). NaN for single-image case (position not observable).
    velocity_vector : np.ndarray
        Shape (3,). NaN for single-image case.
    uncertainty : np.ndarray
        Shape (3, 3). NaN for single-image case.
    is_valid : bool
        False for single-image case. Position not observable from one image.
    method : str
        Algorithm name.
    notes : str
        Human-readable explanation of availability/unavailability.
    position_status : str
        "UNAVAILABLE", "ESTIMATED", or "FAILED".
    velocity_status : str
        "UNAVAILABLE", "ESTIMATED", or "FAILED".
    """

    position_vector: np.ndarray = field(
        default_factory=lambda: np.full(3, float("nan"))
    )
    velocity_vector: np.ndarray = field(
        default_factory=lambda: np.full(3, float("nan"))
    )
    uncertainty: np.ndarray = field(
        default_factory=lambda: np.full((3, 3), float("nan"))
    )
    is_valid: bool = False
    method: str = "single_image_attitude_only"
    notes: str = _POSITION_UNAVAILABLE_NOTE
    position_status: str = "UNAVAILABLE"
    velocity_status: str = "UNAVAILABLE"


def estimate_position(
    attitude_estimate: AttitudeEstimate,
    catalog_match_metadata: dict,
    config: dict,
) -> PositionEstimate:
    """Return PositionEstimate explaining why position cannot be determined.

    Single-image star tracking provides attitude only.
    This function never raises — it always returns a valid object with
    is_valid=False and an honest explanation.

    Parameters
    ----------
    attitude_estimate : AttitudeEstimate
        Attitude result from estimate_attitude().
    catalog_match_metadata : dict
        Catalog match metadata (unused for single-image case).
    config : dict
        Navigation config dict.

    Returns
    -------
    PositionEstimate  with is_valid=False.
    """
    return PositionEstimate(
        position_vector=np.full(3, float("nan")),
        velocity_vector=np.full(3, float("nan")),
        uncertainty=np.full((3, 3), float("nan")),
        is_valid=False,
        method=config.get("navigation", {}).get(
            "position_method", "single_image_attitude_only"
        ),
        notes=_POSITION_UNAVAILABLE_NOTE,
        position_status="UNAVAILABLE",
        velocity_status="UNAVAILABLE",
    )
