"""
image_preprocessing.py
======================
Responsible for transforming raw star-field images into clean, normalised
arrays ready for star detection.

Pipeline (planned)
------------------
1. Load raw image from disk (FITS / PNG / TIFF — format TBD).
2. Estimate and subtract the background illumination component.
3. Apply noise reduction (algorithm TBD).
4. Normalise pixel intensities to a consistent range (method TBD).
5. Return a preprocessed image array and optional metadata.

Implementation note
-------------------
All algorithm choices and tunable parameters must be sourced from config.yaml.
No hard-coded thresholds or paths should appear in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a star-field image from disk and return a 2-D float32 array.

    Parameters
    ----------
    image_path:
        Path to the image file.  Supported formats are to be determined
        during Phase 1 dataset preparation.

    Returns
    -------
    np.ndarray
        2-D (H, W) float32 array of pixel intensities.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    NotImplementedError
        Until this function is implemented in Phase 2.
    """
    # TODO (Phase 2): implement image loading for the chosen format(s).
    #   Consider using:
    #     - astropy.io.fits for FITS images
    #     - cv2.imread / PIL.Image for standard formats
    raise NotImplementedError("load_image is not yet implemented.")


def subtract_background(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Estimate and subtract the background illumination from *image*.

    Parameters
    ----------
    image:
        2-D float32 array of pixel intensities.
    **kwargs:
        Algorithm-specific keyword arguments (sourced from config.yaml).

    Returns
    -------
    np.ndarray
        Background-subtracted image with the same shape as *image*.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 2.
    """
    # TODO (Phase 2): implement background estimation/subtraction.
    #   Algorithm to be selected (e.g. median filter, polynomial fit, sigma-clip).
    raise NotImplementedError("subtract_background is not yet implemented.")


def reduce_noise(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Apply noise reduction to *image*.

    Parameters
    ----------
    image:
        2-D float32 background-subtracted array.
    **kwargs:
        Algorithm-specific keyword arguments (sourced from config.yaml).

    Returns
    -------
    np.ndarray
        Noise-reduced image with the same shape as *image*.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 2.
    """
    # TODO (Phase 2): implement noise reduction.
    #   Candidates: Gaussian smoothing, median filter, wavelet denoising.
    raise NotImplementedError("reduce_noise is not yet implemented.")


def normalise(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Normalise pixel intensities to a consistent range.

    Parameters
    ----------
    image:
        2-D float32 array after background subtraction and noise reduction.
    **kwargs:
        Method-specific keyword arguments (sourced from config.yaml).
        Example key: ``method`` — one of ``"min_max"``, ``"z_score"``.

    Returns
    -------
    np.ndarray
        Normalised float32 array with the same shape as *image*.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 2.
    """
    # TODO (Phase 2): implement intensity normalisation.
    #   Normalisation method to be determined (min-max, z-score, percentile clip).
    raise NotImplementedError("normalise is not yet implemented.")


def preprocess(image_path: str | Path, config: dict) -> np.ndarray:
    """Run the full preprocessing pipeline for a single image.

    This is the primary entry point used by downstream components.

    Parameters
    ----------
    image_path:
        Path to the raw input image.
    config:
        Preprocessing configuration dict, typically the ``preprocessing``
        section of config.yaml.

    Returns
    -------
    np.ndarray
        Preprocessed 2-D float32 image array, ready for star detection.

    Raises
    ------
    NotImplementedError
        Until all constituent steps are implemented in Phase 2.
    """
    # TODO (Phase 2): wire together load_image → subtract_background →
    #   reduce_noise → normalise once each step is implemented.
    raise NotImplementedError("preprocess pipeline is not yet implemented.")
