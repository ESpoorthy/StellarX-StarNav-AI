"""
camera_model.py — Phase 5
==========================
Pinhole camera model for star-tracker sensor.

Coordinate conventions (documented precisely):
  Image frame: origin top-left, +x right, +y down (standard image coordinates)
  Camera frame (right-handed 3D):
    +Z: boresight (pointing into scene)
    +X: aligned with image +x (right)
    +Y: aligned with image -y (up, opposite image row direction)
  Inertial frame: J2000 ICRS
    +X: (RA=0°, Dec=0°)
    +Y: (RA=90°, Dec=0°)
    +Z: north celestial pole (Dec=90°)

  Attitude rotation R: maps camera-frame vectors to inertial-frame vectors
    v_inertial = R @ v_camera
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


@dataclass
class CameraModel:
    """Pinhole camera model for a star-tracker sensor.

    Attributes
    ----------
    image_width : int
        Sensor width in pixels.
    image_height : int
        Sensor height in pixels.
    fov_deg : float
        Full field-of-view in degrees.
    cx : float
        Principal point column (pixels). Defaults to image_width/2.
    cy : float
        Principal point row (pixels). Defaults to image_height/2.
    focal_px : float
        Focal length in pixels. Computed from fov_deg if 0.

    Notes
    -----
    The +Y camera axis is UP (opposite the image row direction).
    When converting pixel (col, row) to a unit vector:
      y_cam = -(row - cy) / focal_px   [minus sign for y-flip]
    """

    image_width: int = 512
    image_height: int = 512
    fov_deg: float = 20.0
    cx: float = 0.0
    cy: float = 0.0
    focal_px: float = 0.0

    def __post_init__(self) -> None:
        """Compute derived parameters if not set."""
        if self.cx == 0.0:
            self.cx = self.image_width / 2.0
        if self.cy == 0.0:
            self.cy = self.image_height / 2.0
        if self.focal_px == 0.0:
            half_fov_rad = math.radians(self.fov_deg / 2.0)
            self.focal_px = (self.image_width / 2.0) / math.tan(half_fov_rad)

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def pixel_to_unit_vector(self, col: float, row: float) -> np.ndarray:
        """Convert pixel coordinates to a camera-frame unit vector.

        Parameters
        ----------
        col : float
            Pixel column coordinate.
        row : float
            Pixel row coordinate.

        Returns
        -------
        np.ndarray
            Shape (3,) unit vector [x, y, z] in camera frame.
            - +x: right (image +col direction)
            - +y: up (image -row direction)
            - +z: boresight
        """
        x = (col - self.cx) / self.focal_px
        y = -(row - self.cy) / self.focal_px  # y-flip
        z = 1.0

        vec = np.array([x, y, z], dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return vec / norm

    def unit_vector_to_pixel(self, unit_vec: np.ndarray) -> tuple[float, float]:
        """Convert a camera-frame unit vector to pixel coordinates.

        Parameters
        ----------
        unit_vec : np.ndarray
            Shape (3,) camera-frame unit vector.

        Returns
        -------
        tuple[float, float]
            (col, row) pixel coordinates. Returns (nan, nan) if z <= 0
            (star behind camera).
        """
        x, y, z = float(unit_vec[0]), float(unit_vec[1]), float(unit_vec[2])

        if z <= 0:
            return float("nan"), float("nan")

        col = (x / z) * self.focal_px + self.cx
        row = -(y / z) * self.focal_px + self.cy  # y-flip inverse

        return col, row

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "CameraModel":
        """Create a CameraModel from the project configuration dict.

        Reads parameters from config['dataset']:
        - image_width (default 512)
        - image_height (default 512)
        - field_of_view_deg (default 20.0)

        Parameters
        ----------
        config : dict
            Project configuration dict.

        Returns
        -------
        CameraModel
        """
        dataset_cfg = config.get("dataset", {})
        image_width = int(dataset_cfg.get("image_width", 512))
        image_height = int(dataset_cfg.get("image_height", 512))
        fov_deg = float(dataset_cfg.get("field_of_view_deg", 20.0))

        return cls(
            image_width=image_width,
            image_height=image_height,
            fov_deg=fov_deg,
            cx=0.0,    # will be set to image_width/2 in __post_init__
            cy=0.0,    # will be set to image_height/2 in __post_init__
            focal_px=0.0,  # will be computed from fov_deg in __post_init__
        )

    def __repr__(self) -> str:
        return (
            f"CameraModel(W={self.image_width}, H={self.image_height}, "
            f"fov={self.fov_deg}°, focal={self.focal_px:.1f}px)"
        )
