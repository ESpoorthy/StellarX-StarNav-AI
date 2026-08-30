"""
pattern_builder.py — Phase 4
=============================
Pixel-to-unit-vector conversion and angular pattern construction.

Camera model (matches star_field_generator.py):
  focal_px = (image_width/2) / tan(radians(fov_deg/2))
  principal point: cx=width/2, cy=height/2

  Pixel to camera-frame unit vector:
    x_cam = (col - cx) / focal_px
    y_cam = -(row - cy) / focal_px    # y-flip: image rows increase downward
    z_cam = 1.0
    unit_vec = normalize([x_cam, y_cam, z_cam])

  Camera frame convention:
    +Z: boresight (pointing out of camera)
    +X: right (image column direction)
    +Y: up (opposite to image row direction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.preprocessing.star_detection import StarCandidate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StarPattern:
    """Angular pattern built from detected stars.

    Attributes
    ----------
    unit_vectors : np.ndarray
        Shape (N, 3) camera-frame unit vectors.
    pixel_coords : np.ndarray
        Shape (N, 2) pixel (col, row) centroids.
    brightnesses : np.ndarray
        Shape (N,) brightness values.
    pairwise_angles_deg : np.ndarray
        Shape (N, N) symmetric angular separations in degrees.
    n_stars : int
        Number of stars in the pattern.
    focal_px : float
        Focal length in pixels.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.
    fov_deg : float
        Full field of view in degrees.
    """

    unit_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pixel_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    brightnesses: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pairwise_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_stars: int = 0
    focal_px: float = 0.0
    image_width: int = 512
    image_height: int = 512
    fov_deg: float = 20.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pattern(
    stars: list[StarCandidate],
    config: dict,
) -> StarPattern:
    """Build a StarPattern from detected star candidates.

    Sorts stars by brightness (descending), caps at max_stars,
    converts pixel coordinates to camera-frame unit vectors,
    and computes the pairwise angular separation matrix.

    Parameters
    ----------
    stars : list[StarCandidate]
        Detected star candidates from detect_stars().
    config : dict
        Project configuration dict. Reads from:
        - config['features']['max_stars'] (default 10)
        - config['dataset']['field_of_view_deg'] (default 20.0)
        - config['dataset']['image_width'] (default 512)
        - config['dataset']['image_height'] (default 512)

    Returns
    -------
    StarPattern
        Pattern with unit vectors and pairwise angles.
        If fewer than 1 star, returns an empty StarPattern with n_stars=0.
    """
    dataset_cfg = config.get("dataset", {})
    feat_cfg = config.get("features", {})

    max_stars = int(feat_cfg.get("max_stars", 10))
    fov_deg = float(dataset_cfg.get("field_of_view_deg", 20.0))
    image_width = int(dataset_cfg.get("image_width", 512))
    image_height = int(dataset_cfg.get("image_height", 512))

    focal_px = _compute_focal_px(image_width, fov_deg)

    # Sort by brightness descending, cap at max_stars
    sorted_stars = sorted(stars, key=lambda s: s.brightness, reverse=True)
    top_stars = sorted_stars[:max_stars]

    n = len(top_stars)

    if n == 0:
        return StarPattern(
            unit_vectors=np.zeros((0, 3), dtype=np.float64),
            pixel_coords=np.zeros((0, 2), dtype=np.float64),
            brightnesses=np.zeros(0, dtype=np.float64),
            pairwise_angles_deg=np.zeros((0, 0), dtype=np.float64),
            n_stars=0,
            focal_px=focal_px,
            image_width=image_width,
            image_height=image_height,
            fov_deg=fov_deg,
        )

    # Convert pixel coords to unit vectors
    unit_vecs = np.zeros((n, 3), dtype=np.float64)
    pixel_coords = np.zeros((n, 2), dtype=np.float64)
    brightnesses = np.zeros(n, dtype=np.float64)

    for i, star in enumerate(top_stars):
        uv = pixels_to_unit_vector(
            star.x, star.y, image_width, image_height, fov_deg
        )
        unit_vecs[i] = uv
        pixel_coords[i] = [star.x, star.y]
        brightnesses[i] = star.brightness

    # Compute NxN pairwise angle matrix
    pairwise_angles = _compute_pairwise_angles(unit_vecs)

    return StarPattern(
        unit_vectors=unit_vecs,
        pixel_coords=pixel_coords,
        brightnesses=brightnesses,
        pairwise_angles_deg=pairwise_angles,
        n_stars=n,
        focal_px=focal_px,
        image_width=image_width,
        image_height=image_height,
        fov_deg=fov_deg,
    )


def pixels_to_unit_vector(
    col: float,
    row: float,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> np.ndarray:
    """Convert a pixel coordinate to a camera-frame unit vector.

    Uses the pinhole camera model:
      - cx = image_width / 2, cy = image_height / 2
      - focal_px = (image_width / 2) / tan(fov_deg / 2)
      - x_cam = (col - cx) / focal_px
      - y_cam = -(row - cy) / focal_px   [y-flip for image convention]
      - z_cam = 1.0
      - unit_vec = normalize([x_cam, y_cam, z_cam])

    Parameters
    ----------
    col, row : float
        Pixel coordinate (column, row).
    image_width, image_height : int
        Image dimensions in pixels.
    fov_deg : float
        Full field-of-view in degrees.

    Returns
    -------
    np.ndarray
        Shape (3,) unit vector in camera frame.
    """
    focal_px = _compute_focal_px(image_width, fov_deg)
    cx = image_width / 2.0
    cy = image_height / 2.0

    x_cam = (col - cx) / focal_px
    y_cam = -(row - cy) / focal_px
    z_cam = 1.0

    vec = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / norm


def _compute_focal_px(image_width: int, fov_deg: float) -> float:
    """Compute focal length in pixels from image width and full FoV.

    focal_px = (image_width / 2) / tan(fov_deg / 2)

    Parameters
    ----------
    image_width : int
        Image width in pixels.
    fov_deg : float
        Full field-of-view in degrees.

    Returns
    -------
    float
        Focal length in pixels.
    """
    half_fov_rad = math.radians(fov_deg / 2.0)
    return (image_width / 2.0) / math.tan(half_fov_rad)


def _compute_pairwise_angles(unit_vecs: np.ndarray) -> np.ndarray:
    """Compute NxN matrix of pairwise angular separations in degrees.

    Parameters
    ----------
    unit_vecs : np.ndarray
        Shape (N, 3) array of unit vectors.

    Returns
    -------
    np.ndarray
        Shape (N, N) matrix. Entry [i, j] is the angle in degrees between
        unit_vecs[i] and unit_vecs[j]. Diagonal is 0.
    """
    n = len(unit_vecs)
    angles = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dot = float(np.dot(unit_vecs[i], unit_vecs[j]))
            dot = max(-1.0, min(1.0, dot))
            angle_deg = math.degrees(math.acos(dot))
            angles[i, j] = angle_deg
            angles[j, i] = angle_deg

    return angles
