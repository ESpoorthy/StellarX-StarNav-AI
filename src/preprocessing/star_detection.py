"""
star_detection.py
=================
Locates stars in a preprocessed star-field image and extracts per-star
attributes for downstream neural-network input.

Phase 2 implementation
-----------------------
Three public functions are implemented:

1. :func:`detect_stars`     — threshold → connected-component labelling →
                               morphological filtering → centroiding
2. :func:`compute_centroid` — intensity-weighted centroid within a bounding
                               box (sub-pixel precision)
3. :func:`extract_features` — Phase 3 stub (not yet implemented)

Algorithm: connected-component thresholding
-------------------------------------------
The detection pipeline is:

    preprocessed image (float32, [0,1])
            ↓
    threshold mask  (pixels > min_brightness)
            ↓
    scipy.ndimage.label  → integer label map
            ↓
    per-label stats (bbox, peak, area)
            ↓
    filter by area and peak brightness
            ↓
    intensity-weighted centroid per surviving label
            ↓
    sort by descending brightness, cap at max_stars
            ↓
    list[StarCandidate]

This approach is robust, fast, and requires no dependencies beyond
numpy and scipy (both already installed).

Centroiding
-----------
Intensity-weighted centroid (also called "centre of mass" or "first moment"):

    x̄ = Σ(I_ij · j) / Σ(I_ij)
    ȳ = Σ(I_ij · i) / Σ(I_ij)

computed over a (2W+1) × (2W+1) window centred on the peak pixel, where
W = ``centroid_half_window`` from config.  This achieves sub-pixel accuracy
when the PSF is roughly symmetric and the SNR is high — both true for the
synthetic images.

Feature extraction (Phase 3)
-----------------------------
:func:`extract_features` remains a stub.  The feature representation
(inter-star angular distances, brightness ratios, geometric descriptors)
will be designed in Phase 3 once the detection pipeline is validated.

Scientific accuracy
-------------------
- The connected-component approach is equivalent to a 2-D binary
  segmentation followed by region labelling.  It is a standard technique
  in astronomical image processing (see e.g. SExtractor's first step).
- Intensity-weighted centroiding achieves < 0.1 px accuracy for Gaussian
  PSFs with SNR > 10 and window size ≥ 3σ (Berry & Burnell, 2005).
- All thresholds and window sizes are configurable; no hard-coded values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import label as nd_label


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StarCandidate:
    """A single detected star candidate.

    Attributes
    ----------
    x : float
        Sub-pixel centroid **column** coordinate (horizontal), 0-indexed.
    y : float
        Sub-pixel centroid **row** coordinate (vertical), 0-indexed.
    brightness : float
        Integrated (sum) pixel intensity within the detection aperture,
        in the same units as the preprocessed image (normalised [0, 1]).
    peak : float
        Maximum pixel value within the detection blob.
    area : int
        Number of pixels in the connected-component blob.
    bbox : tuple[int, int, int, int]
        Bounding box ``(row_min, col_min, row_max, col_max)`` of the blob
        (exclusive upper bounds, numpy-convention).
    features : np.ndarray
        Per-star feature vector for neural-network input.
        Shape is TBD (Phase 3); defaults to empty array.
    metadata : dict
        Optional extra fields (e.g. debug info from the detector).
    """

    x: float = 0.0
    y: float = 0.0
    brightness: float = 0.0
    peak: float = 0.0
    area: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_stars(
    image: np.ndarray,
    config: dict,
) -> list[StarCandidate]:
    """Locate stars in a preprocessed image using connected-component analysis.

    Pipeline
    --------
    1. Threshold the image at ``min_brightness`` (absolute) or
       ``median + k * sigma`` (sigma-clip) — from config.
    2. Label connected components using :func:`scipy.ndimage.label`.
    3. Filter blobs by pixel area and peak brightness.
    4. Compute an intensity-weighted centroid for each surviving blob.
    5. Return candidates sorted by descending brightness, capped at
       ``max_stars``.

    Parameters
    ----------
    image:
        2-D float32 preprocessed image array with values in [0, 1].
    config:
        Star-detection configuration dict — typically the
        ``star_detection`` section of config.yaml.

    Returns
    -------
    list[StarCandidate]
        Detected star candidates, sorted by descending ``brightness``.
        Empty list if no stars pass all filters.

    Raises
    ------
    ValueError
        If *image* is not a 2-D array.
    """
    if image.ndim != 2:
        raise ValueError(
            f"detect_stars expects a 2-D image array, got shape {image.shape}."
        )

    # ── Config parameters ────────────────────────────────────────────────────
    threshold_method  = config.get("threshold_method",    "absolute")
    min_brightness    = float(config.get("min_brightness",    0.05))
    sigma_clip_k      = float(config.get("sigma_clip_k",      5.0))
    min_area          = int(config.get("min_area_px",          1))
    max_area          = int(config.get("max_area_px",          200))
    min_peak          = float(config.get("min_peak_brightness", 0.04))
    max_stars         = int(config.get("max_stars",            50))
    centroid_method   = config.get("centroid_method",    "intensity_weighted")
    centroid_hw       = int(config.get("centroid_half_window", 5))

    # ── Step 1: threshold → binary mask ──────────────────────────────────────
    if threshold_method == "absolute":
        threshold = min_brightness
    elif threshold_method == "sigma_clip":
        median = float(np.median(image))
        sigma  = float(image.std())
        threshold = median + sigma_clip_k * sigma
    else:
        raise ValueError(
            f"Unknown threshold_method '{threshold_method}'. "
            "Choose 'absolute' or 'sigma_clip'."
        )

    mask = (image >= threshold).astype(np.int32)

    if mask.sum() == 0:
        return []

    # ── Step 2: connected-component labelling ─────────────────────────────────
    # 8-connectivity structure (diagonals count as connected)
    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n_labels = nd_label(mask, structure=struct)

    if n_labels == 0:
        return []

    # ── Step 3 & 4: filter blobs, compute centroids ───────────────────────────
    candidates: list[StarCandidate] = []

    for lbl in range(1, n_labels + 1):
        blob_mask = labeled == lbl
        rows, cols = np.where(blob_mask)

        area = int(blob_mask.sum())
        if area < min_area or area > max_area:
            continue

        # Bounding box
        r_min, r_max = int(rows.min()), int(rows.max()) + 1
        c_min, c_max = int(cols.min()), int(cols.max()) + 1

        blob_pixels = image[blob_mask]
        peak       = float(blob_pixels.max())
        brightness = float(blob_pixels.sum())

        if peak < min_peak:
            continue

        # ── Centroid ────────────────────────────────────────────────────────
        if centroid_method == "intensity_weighted":
            cx, cy = compute_centroid(
                image,
                bbox=(r_min, c_min, r_max, c_max),
                half_window=centroid_hw,
                peak_row=int(rows[np.argmax(blob_pixels)]),
                peak_col=int(cols[np.argmax(blob_pixels)]),
            )
        else:
            # "peak" fallback — use the pixel with the highest value
            peak_idx = np.argmax(blob_pixels)
            cx = float(cols[peak_idx])
            cy = float(rows[peak_idx])

        candidates.append(
            StarCandidate(
                x=cx,
                y=cy,
                brightness=brightness,
                peak=peak,
                area=area,
                bbox=(r_min, c_min, r_max, c_max),
            )
        )

    # ── Step 5: sort by brightness, cap ───────────────────────────────────────
    candidates.sort(key=lambda c: c.brightness, reverse=True)
    return candidates[:max_stars]


def compute_centroid(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    half_window: int = 5,
    peak_row: int | None = None,
    peak_col: int | None = None,
) -> tuple[float, float]:
    """Compute the intensity-weighted centroid of a star.

    The centroid is computed over a (2W+1) × (2W+1) window centred on the
    peak pixel.  Using the peak rather than the bounding-box centre makes
    the estimate more robust when blobs are asymmetric or partially masked.

    Parameters
    ----------
    image:
        2-D float32 preprocessed image array.
    bbox:
        ``(row_min, col_min, row_max, col_max)`` of the detection blob.
        Used to locate the peak when *peak_row* / *peak_col* are not given.
    half_window:
        Half-size of the centroiding window in pixels.  The window is
        ``(2*half_window + 1) × (2*half_window + 1)``.  Default: 5.
    peak_row:
        Row index of the peak pixel.  If ``None``, computed from *bbox*.
    peak_col:
        Column index of the peak pixel.  If ``None``, computed from *bbox*.

    Returns
    -------
    tuple[float, float]
        ``(x, y)`` — sub-pixel centroid as (column, row).

    Raises
    ------
    ValueError
        If *image* is not 2-D.
    """
    if image.ndim != 2:
        raise ValueError(
            f"compute_centroid expects a 2-D array, got shape {image.shape}."
        )

    h, w = image.shape
    r_min, c_min, r_max, c_max = bbox

    # Locate peak if not provided
    if peak_row is None or peak_col is None:
        sub = image[r_min:r_max, c_min:c_max]
        local_idx = np.unravel_index(sub.argmax(), sub.shape)
        peak_row = r_min + int(local_idx[0])
        peak_col = c_min + int(local_idx[1])

    # Centroiding window clamped to image bounds
    hw = int(half_window)
    wr_min = max(0, peak_row - hw)
    wr_max = min(h, peak_row + hw + 1)
    wc_min = max(0, peak_col - hw)
    wc_max = min(w, peak_col + hw + 1)

    window = image[wr_min:wr_max, wc_min:wc_max].astype(np.float64)

    total = window.sum()
    if total <= 0:
        # Degenerate case — return peak pixel coordinates
        return float(peak_col), float(peak_row)

    rows_idx = np.arange(wr_min, wr_max, dtype=np.float64)
    cols_idx = np.arange(wc_min, wc_max, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)

    cx = float((col_grid * window).sum() / total)
    cy = float((row_grid * window).sum() / total)

    return cx, cy


def extract_features(
    stars: list[StarCandidate],
    config: dict,
) -> np.ndarray:
    """Extract a feature representation from detected stars for the neural network.

    Phase 3 stub — not yet implemented.

    The feature representation design (inter-star angular distances,
    brightness ratios, geometric descriptors) will be defined in Phase 3
    after the detection pipeline has been validated end-to-end.

    Parameters
    ----------
    stars:
        List of :class:`StarCandidate` objects from :func:`detect_stars`.
    config:
        Feature-extraction configuration dict.

    Returns
    -------
    np.ndarray
        Feature array.  Shape is TBD (Phase 3).

    Raises
    ------
    NotImplementedError
        Until Phase 3 implementation.
    """
    # TODO (Phase 3): design and implement feature extraction.
    #   Candidates:
    #   - Pairwise angular distances between star centroids (normalised by FoV)
    #   - Brightness ratios between pairs / triplets
    #   - Geometric descriptors (triangle side ratios, polygon angles)
    #   - Normalised (x, y) positions within the image frame
    raise NotImplementedError("extract_features is not yet implemented (Phase 3).")
