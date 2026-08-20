"""
position_estimator.py
=====================
Responsible for estimating spacecraft position from catalog matches,
where the chosen methodology supports it.

Responsibility (planned)
------------------------
- Accept a verified catalog match and attitude estimate.
- Apply a position estimation algorithm appropriate to the available data.
- Return a PositionEstimate with uncertainty metrics.

Implementation note
-------------------
Position estimation from star imagery alone is generally underdetermined
without additional information (e.g. known orbital mechanics, multi-camera
parallax, or magnitude-distance relationships).  Whether and how position
estimation is supported will be determined during Phase 5.

All parameters must be sourced from config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.navigation.attitude_estimator import AttitudeEstimate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PositionEstimate:
    """Spacecraft position estimate.

    Attributes
    ----------
    position_vector : np.ndarray
        Estimated position vector.  Units and reference frame are TBD
        (Phase 5) and depend on the chosen methodology.
    uncertainty : np.ndarray
        Uncertainty / covariance associated with the estimate.  Shape TBD.
    is_valid : bool
        True if the estimate meets the configured quality threshold.
    method : str
        Name of the algorithm used to produce this estimate.
    notes : str
        Human-readable notes, e.g. explaining why estimation was not possible.
    """

    position_vector: np.ndarray = field(
        default_factory=lambda: np.array([float("nan")] * 3)
    )
    uncertainty: np.ndarray = field(
        default_factory=lambda: np.full((3, 3), float("nan"))
    )
    is_valid: bool = False
    method: str = "undefined"
    notes: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_position(
    attitude_estimate: AttitudeEstimate,
    catalog_match_metadata: dict,
    config: dict,
) -> PositionEstimate:
    """Estimate spacecraft position from attitude and catalog match data.

    Parameters
    ----------
    attitude_estimate:
        Validated AttitudeEstimate from ``src.navigation.attitude_estimator``.
    catalog_match_metadata:
        Additional metadata from the catalog match that may support
        position estimation (content TBD, Phase 5).
    config:
        Navigation configuration dict (``navigation`` section of config.yaml).

    Returns
    -------
    PositionEstimate
        Estimated position.  ``is_valid`` will be False if estimation is
        not supported by the current methodology.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 5.
    """
    # TODO (Phase 5): determine feasibility of position estimation given
    #   the chosen methodology and implement accordingly.
    #   If position estimation is not supportable, return a PositionEstimate
    #   with is_valid=False and an explanatory note rather than raising.
    raise NotImplementedError("estimate_position is not yet implemented.")
