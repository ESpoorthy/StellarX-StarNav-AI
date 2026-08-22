"""
image_preprocessing.py
=======================
Loads star-field images from disk and applies a preprocessing pipeline
that prepares images for star detection.

Phase 2 implementation
-----------------------
All four pipeline stages are now implemented:

1. :func:`load_image`        — load 8/16-bit greyscale PNG / TIFF → float32 [0, 1]
2. :func:`subtract_background` — estimate and remove the background
3. :func:`reduce_noise`      — Gaussian smoothing to suppress read noise
4. :func:`normalise`         — rescale intensities for consistent thresholding
5. :func:`preprocess`        — convenience wrapper that chains all four steps

Algorithm choices (Phase 2)
----------------------------
- **Background subtraction**: large-kernel median filter.  The median of a
  large neighbourhood estimates the slowly-varying background without being
  pulled by the compact, bright star blobs.  Filter size is configurable
  (``background_filter_size`` in config.yaml, default 31 px).

- **Noise reduction**: Gaussian blur.  A small sigma (default 0.8 px) smooths
  sub-pixel read noise while barely widening star PSFs (FWHM ≈ 3.5 px).
  Setting ``noise_sigma: 0`` skips this step entirely.

- **Normalisation**: min–max rescaling maps the clipped dynamic range to
  [0, 1] so that the absolute threshold ``min_brightness`` in config.yaml
  has a consistent meaning regardless of image gain.

Approximations documented
--------------------------
- The median-filter background estimate is a coarse approximation: it works
  well when the image contains only a few compact stars against a flat
  background, which is true for the Phase 1 synthetic dataset.  For crowded
  fields or spatially structured backgrounds (e.g. stray light gradients in
  real spacecraft imagery), a more sophisticated estimator (sigma-clipped
  polynomial, or iterative 2-D background mesh) should be used.
- Gaussian smoothing is isotropic; real sensor noise may be anisotropic.

All parameters are read from config.yaml — no hard-coded values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter


# ---------------------------------------------------------------------------
# Phase 1 — load_image (unchanged)
# ---------------------------------------------------------------------------


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a star-field image from disk and return a 2-D float32 array.

    Reads greyscale PNG or TIFF files (8-bit or 16-bit).  Pixel values are
    normalised to [0, 1] regardless of source bit-depth.

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
            "Only PNG and TIFF are supported."
        )

    pil_img = Image.open(image_path)

    if pil_img.mode in ("RGB", "RGBA"):
        pil_img = pil_img.convert("L")
    elif pil_img.mode == "I;16":
        arr = np.frombuffer(pil_img.tobytes(), dtype=np.uint16).reshape(
            pil_img.height, pil_img.width
        )
        return arr.astype(np.float32) / 65535.0
    elif pil_img.mode == "I":
        arr = np.array(pil_img, dtype=np.int32)
        arr = np.clip(arr, 0, 65535)
        return arr.astype(np.float32) / 65535.0

    arr = np.array(pil_img)
    if arr.dtype == np.uint8:
        divisor = 255.0
    elif arr.dtype == np.uint16:
        divisor = 65535.0
    else:
        max_val = float(arr.max())
        divisor = max_val if max_val > 0 else 1.0

    return arr.astype(np.float32) / divisor


# ---------------------------------------------------------------------------
# Phase 2 — pipeline steps
# ---------------------------------------------------------------------------


def subtract_background(
    image: np.ndarray,
    method: str = "median_filter",
    filter_size: int = 31,
) -> np.ndarray:
    """Estimate and subtract the background illumination from *image*.

    Algorithm: **large-kernel median filter**.
    The median of a (filter_size × filter_size) neighbourhood provides a
    smooth estimate of the slowly-varying background.  Subtracting it
    removes DC bias and spatial gradients while leaving compact star signals
    intact.  The result is clipped to [0, 1].

    Parameters
    ----------
    image:
        2-D float32 array in [0, 1].
    method:
        Background estimation method.  Supported: ``"median_filter"``,
        ``"constant"`` (subtracts the global image median — fast but coarse).
    filter_size:
        Kernel size for the median filter (must be a positive odd integer).
        Larger values handle broader background structures but cost more CPU.
        Default: 31 px.

    Returns
    -------
    np.ndarray
        Background-subtracted float32 array, clipped to [0, 1], same shape
        as *image*.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method == "median_filter":
        # Ensure filter_size is odd
        fs = int(filter_size)
        if fs % 2 == 0:
            fs += 1
        background = median_filter(image.astype(np.float64), size=fs).astype(np.float32)
    elif method == "constant":
        background = np.full_like(image, float(np.median(image)))
    else:
        raise ValueError(
            f"Unknown background subtraction method '{method}'. "
            "Choose 'median_filter' or 'constant'."
        )

    result = image - background
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def reduce_noise(
    image: np.ndarray,
    method: str = "gaussian",
    sigma: float = 0.8,
) -> np.ndarray:
    """Apply noise reduction to *image*.

    Algorithm: **Gaussian smoothing**.
    A small-sigma Gaussian blur suppresses sub-pixel read noise without
    significantly broadening the star PSF.  Setting ``sigma=0`` (or
    ``method="none"``) skips the step and returns the image unchanged.

    Parameters
    ----------
    image:
        2-D float32 array in [0, 1].
    method:
        Noise-reduction method.  Supported: ``"gaussian"``, ``"none"``.
    sigma:
        Gaussian blur standard deviation in pixels.  Default: 0.8 px.

    Returns
    -------
    np.ndarray
        Noise-reduced float32 array in [0, 1], same shape as *image*.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method == "none" or sigma <= 0.0:
        return image.copy()
    elif method == "gaussian":
        smoothed = gaussian_filter(image.astype(np.float64), sigma=float(sigma))
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown noise reduction method '{method}'. "
            "Choose 'gaussian' or 'none'."
        )


def normalise(
    image: np.ndarray,
    method: str = "min_max",
    clip_percentile: float = 99.9,
) -> np.ndarray:
    """Normalise pixel intensities for consistent thresholding.

    Two methods are supported:

    - **min_max**: maps ``[0, p99.9]`` to ``[0, 1]``.  Using a high
      percentile instead of the true maximum makes the normalisation robust
      to hot pixels and cosmic-ray artifacts.
    - **z_score**: subtracts the mean and divides by the standard deviation.
      Useful for debugging; not recommended for thresholding because the
      output range is unbounded.

    Parameters
    ----------
    image:
        2-D float32 array.
    method:
        ``"min_max"`` (default) or ``"z_score"``.
    clip_percentile:
        Upper percentile used as the maximum in min_max mode.  Default 99.9.

    Returns
    -------
    np.ndarray
        Normalised float32 array, same shape as *image*.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method == "min_max":
        lo = float(image.min())
        hi = float(np.percentile(image, clip_percentile))
        if hi - lo < 1e-9:
            # Uniform or near-uniform image — return as-is
            return image.copy()
        result = (image.astype(np.float64) - lo) / (hi - lo)
        return np.clip(result, 0.0, 1.0).astype(np.float32)
    elif method == "z_score":
        mu = float(image.mean())
        sigma = float(image.std())
        if sigma < 1e-9:
            return np.zeros_like(image)
        return ((image.astype(np.float64) - mu) / sigma).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose 'min_max' or 'z_score'."
        )


def preprocess(image_path: str | Path, config: dict) -> np.ndarray:
    """Run the full preprocessing pipeline for a single image.

    Chains: load_image → subtract_background → reduce_noise → normalise.

    Parameters are read from ``config["preprocessing"]``.  Any key absent
    from the dict falls back to the function defaults.

    Parameters
    ----------
    image_path:
        Path to the raw input image (PNG or TIFF).
    config:
        Full project configuration dict (loaded from config.yaml).
        The ``preprocessing`` sub-dict is used.

    Returns
    -------
    np.ndarray
        Preprocessed 2-D float32 image array in [0, 1], ready for star
        detection.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    """
    pp = config.get("preprocessing", {})

    # Step 1 — load
    image = load_image(image_path)

    # Step 2 — background subtraction
    if pp.get("background_subtraction", True):
        image = subtract_background(
            image,
            method=pp.get("background_method", "median_filter"),
            filter_size=int(pp.get("background_filter_size", 31)),
        )

    # Step 3 — noise reduction
    if pp.get("noise_reduction", True):
        image = reduce_noise(
            image,
            method=pp.get("noise_method", "gaussian"),
            sigma=float(pp.get("noise_sigma", 0.8)),
        )

    # Step 4 — normalisation
    norm_method = pp.get("normalization", "min_max")
    if norm_method and norm_method != "none":
        image = normalise(image, method=norm_method)

    return image
