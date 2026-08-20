"""
star_detection.py
=================
Responsible for locating stars in a preprocessed image and extracting
per-star features for downstream neural network input.

Pipeline (planned)
------------------
1. Apply intensity threshold to identify candidate star regions.
2. Localise each candidate to sub-pixel precision (centroiding).
3. Measure integrated brightness per candidate.
4. Filter candidates by configurable quality criteria
   (min brightness, morphology, area).
5. Extract a feature representation suitable for neural network input.
6. Return a structured list of detected stars.

Implementation note
-------------------
Detection algorithm, centroiding method, and feature design are all
to be determined during Phase 2–3.  All threshold parameters must be
sourced from config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StarCandidate:
    """Represents a single detected star candidate.

    Attributes
    ----------
    x : float
        Sub-pixel centroid column coordinate (horizontal).
    y : float
        Sub-pixel centroid row coordinate (vertical).
    brightness : float
        Integrated pixel intensity within the detection aperture.
    features : np.ndarray
        Feature vector extracted for this star candidate.
        Shape and content are TBD (Phase 3).
    metadata : dict
        Optional extra fields (e.g. morphological descriptors).
    """

    x: float = 0.0
    y: float = 0.0
    brightness: float = 0.0
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_stars(
    image: np.ndarray,
    config: dict,
) -> list[StarCandidate]:
    """Locate stars in a preprocessed image.

    Parameters
    ----------
    image:
        2-D float32 preprocessed image array.
    config:
        Star detection configuration dict, typically the
        ``star_detection`` section of config.yaml.

    Returns
    -------
    list[StarCandidate]
        Detected star candidates, ordered by descending brightness.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 2.
    """
    # TODO (Phase 2): implement star detection.
    #   Algorithm candidates: blob detection (LoG/DoG), connected-component
    #   analysis after thresholding, matched filter / PSF fitting.
    raise NotImplementedError("detect_stars is not yet implemented.")


def compute_centroid(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    **kwargs: Any,
) -> tuple[float, float]:
    """Compute the sub-pixel centroid of a star within a bounding box.

    Parameters
    ----------
    image:
        2-D float32 image array.
    bbox:
        (row_min, col_min, row_max, col_max) bounding box of the candidate.
    **kwargs:
        Centroiding method keyword arguments (sourced from config).

    Returns
    -------
    tuple[float, float]
        (x, y) sub-pixel centroid coordinates in image space.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 2.
    """
    # TODO (Phase 2): implement centroiding.
    #   Candidates: intensity-weighted centroid, Gaussian PSF fit,
    #   iterative windowed centroid.
    raise NotImplementedError("compute_centroid is not yet implemented.")


def extract_features(
    stars: list[StarCandidate],
    config: dict,
) -> np.ndarray:
    """Extract a global feature representation from a list of detected stars.

    The resulting feature array is the input to the neural network.

    Parameters
    ----------
    stars:
        List of detected StarCandidate objects.
    config:
        Feature extraction configuration dict.

    Returns
    -------
    np.ndarray
        Feature array of shape (N_features,) or (N_stars, N_per_star_features).
        Exact shape is TBD (Phase 3).

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 3.
    """
    # TODO (Phase 3): design and implement the feature extraction step.
    #   Candidates: pairwise angular distances, brightness ratios,
    #   geometric descriptors (triangles, polygons), normalised positions.
    raise NotImplementedError("extract_features is not yet implemented.")
