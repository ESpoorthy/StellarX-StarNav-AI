"""
image_preprocessing.py
=======================
Loads synthetic (and future real) star-field images from disk and exposes
a preprocessing pipeline for downstream star detection.

Phase 1 implementation
-----------------------
- :func:`load_image` — reads 8-bit or 16-bit greyscale PNG / TIFF files
  and returns a normalised float32 array in [0, 1].
- :func:`subtract_background`, :func:`reduce_noise`, :func:`normalise`,
  and :func:`preprocess` remain as documented stubs; their algorithms will
  be selected in Phase 2 after star-detection experiments.

Supported formats
-----------------
- **PNG** (8-bit and 16-bit greyscale) — primary format for Phase 1
  synthetic images saved by :mod:`src.preprocessing.dataset_builder`.
- **TIFF** (8-bit and 16-bit greyscale) — alternative lossless format.
- **JPEG** is explicitly *not* supported because lossy compression
  corrupts the low-level photometric data needed for star detection.

All paths must be passed in by the caller; no hard-coded paths appear here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Phase 1 implementation
# ---------------------------------------------------------------------------


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a star-field image from disk and return a 2-D float32 array.

    Reads greyscale PNG or TIFF files (8-bit or 16-bit).  The pixel values
    are normalised to the range [0, 1] regardless of the source bit depth:

    - 8-bit  images are divided by 255.0
    - 16-bit images are divided by 65535.0

    RGB or RGBA images are converted to greyscale using the standard
    luminosity weights before normalisation.

    Parameters
    ----------
    image_path:
        Path to a PNG or TIFF image file.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape (H, W) with values in [0, 1].

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the file format is not a supported greyscale / colour image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError(
            f"Unsupported image format '{suffix}'. "
            "Only PNG and TIFF are supported in Phase 1."
        )

    pil_img = Image.open(image_path)

    # Convert to greyscale if needed
    if pil_img.mode in ("RGB", "RGBA"):
        pil_img = pil_img.convert("L")
    elif pil_img.mode == "I;16":
        # 16-bit greyscale — Pillow stores as raw 16-bit; convert correctly
        arr = np.frombuffer(pil_img.tobytes(), dtype=np.uint16).reshape(
            pil_img.height, pil_img.width
        )
        return (arr.astype(np.float32) / 65535.0)
    elif pil_img.mode == "I":
        # 32-bit signed integer (Pillow internal for some 16-bit PNGs)
        arr = np.array(pil_img, dtype=np.int32)
        # Values are in [0, 65535] for 16-bit source data
        arr = np.clip(arr, 0, 65535)
        return (arr.astype(np.float32) / 65535.0)

    arr = np.array(pil_img)

    # Determine normalisation divisor from dtype
    if arr.dtype == np.uint8:
        divisor = 255.0
    elif arr.dtype == np.uint16:
        divisor = 65535.0
    else:
        # Fallback: scale by max value if non-zero, else leave as-is
        max_val = arr.max()
        divisor = float(max_val) if max_val > 0 else 1.0

    return (arr.astype(np.float32) / divisor)


# ---------------------------------------------------------------------------
# Phase 2 stubs
# ---------------------------------------------------------------------------


def subtract_background(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Estimate and subtract the background illumination from *image*.

    Parameters
    ----------
    image:
        2-D float32 array of pixel intensities in [0, 1].
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

    Notes
    -----
    Algorithm candidates: median filter, polynomial fit, sigma-clipped mean.
    Decision deferred to Phase 2 after examining real background structure
    in generated images.
    """
    # TODO (Phase 2): implement background estimation/subtraction.
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

    Notes
    -----
    Algorithm candidates: Gaussian smoothing, median filter, wavelet
    denoising.  Choice will be validated against star-detection performance.
    """
    # TODO (Phase 2): implement noise reduction.
    raise NotImplementedError("reduce_noise is not yet implemented.")


def normalise(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Normalise pixel intensities to a consistent range.

    Parameters
    ----------
    image:
        2-D float32 array after background subtraction and noise reduction.
    **kwargs:
        Method-specific keyword arguments.
        Key ``method``: one of ``"min_max"`` (default), ``"z_score"``.

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
    raise NotImplementedError("normalise is not yet implemented.")


def preprocess(image_path: str | Path, config: dict) -> np.ndarray:
    """Run the full preprocessing pipeline for a single image.

    In Phase 1 this chains only :func:`load_image`; the remaining steps
    will be wired in once Phase 2 algorithms are selected.

    Parameters
    ----------
    image_path:
        Path to the raw input image.
    config:
        Preprocessing configuration dict (``preprocessing`` section of
        config.yaml).

    Returns
    -------
    np.ndarray
        Preprocessed 2-D float32 image array, ready for star detection.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    NotImplementedError
        If any Phase 2 step is called before it is implemented.
    """
    # Phase 1: only loading is implemented.
    # TODO (Phase 2): chain subtract_background → reduce_noise → normalise.
    return load_image(image_path)
