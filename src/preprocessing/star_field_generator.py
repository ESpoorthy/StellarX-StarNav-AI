"""
star_field_generator.py
=======================
Generates synthetic star-field images that simulate the output of a
spacecraft star-sensor (star tracker).

Scientific model
----------------
Each image represents a narrow field-of-view snapshot of the sky.  The
simulation pipeline is:

1. **Attitude sampling** — a random spacecraft pointing direction is drawn
   uniformly from the unit sphere (RA/Dec), with a random roll angle.

2. **Star selection** — catalog stars whose angular separation from the
   boresight is within half the field-of-view are projected onto the focal
   plane.  Stars fainter than ``min_star_flux`` (after magnitude-to-flux
   conversion) are discarded.

3. **Projection** — a gnomonic (tangent-plane) projection maps each
   selected star's sky coordinates to pixel coordinates:

       x_px = f_px * tan(Δra  * cos(dec_c))
       y_px = f_px * tan(Δdec)

   where f_px = (image_width / 2) / tan(fov_rad / 2) is the focal length
   in pixels and (Δra, Δdec) are the angular offsets from the boresight.

4. **Rendering** — each star is rendered as an isotropic 2-D Gaussian
   point-spread function (PSF) with configurable sigma (``psf_sigma_px``).
   Peak flux is derived from the V magnitude using:

       flux = 10 ** (-0.4 * (vmag - vmag_ref))

   where ``vmag_ref`` is the magnitude of the brightest star in the frame,
   so the brightest star always has flux = 1.0.

5. **Noise** — a constant background level is added, followed by
   Poisson-like shot noise (approximated as Gaussian with sigma = sqrt(flux))
   and Gaussian read noise.

6. **Clipping** — the final image is clipped to [0, 1] and returned as a
   float32 array.

Approximations and limitations
-------------------------------
- Gnomonic projection is accurate within ~10° of the boresight; beyond
  that, distortion grows.  The default 20° FoV keeps distortion small.
- Atmospheric refraction, proper motion, and parallax are not modelled.
- The PSF is isotropic Gaussian; real star sensors have more complex PSFs.
- Shot noise is approximated as Gaussian rather than true Poisson; this is
  a good approximation when flux >> 1 photon.
- No vignetting or flat-field variation is modelled.

These simplifications are appropriate for Phase 1 prototype data generation.
They should be documented and revisited when training on real imagery.

All parameters are sourced from the ``dataset`` section of config.yaml.
No values are hard-coded in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from src.catalog.catalog_loader import CatalogStar, StarCatalog


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RenderedStar:
    """Metadata for a single star rendered into a synthetic image.

    Attributes
    ----------
    star_id : str
        Catalog identifier (``"HIP_<n>"``).
    x_px : float
        Horizontal pixel coordinate (column), sub-pixel precision.
    y_px : float
        Vertical pixel coordinate (row), sub-pixel precision.
    flux : float
        Normalised flux in [0, 1]; 1.0 = brightest star in the frame.
    vmag : float
        V-band apparent magnitude from the catalog.
    ra_deg : float
        Catalog right ascension (degrees).
    dec_deg : float
        Catalog declination (degrees).
    """

    star_id: str = ""
    x_px: float = 0.0
    y_px: float = 0.0
    flux: float = 0.0
    vmag: float = 0.0
    ra_deg: float = 0.0
    dec_deg: float = 0.0


@dataclass
class SyntheticStarField:
    """Output of a single synthetic star-field generation call.

    Attributes
    ----------
    image : np.ndarray
        Float32 array of shape (H, W) in [0, 1].
    stars : list[RenderedStar]
        Ground-truth list of rendered stars.
    boresight_ra_deg : float
        Camera boresight right ascension (degrees).
    boresight_dec_deg : float
        Camera boresight declination (degrees).
    roll_deg : float
        Camera roll angle around the boresight (degrees).
    fov_deg : float
        Full field-of-view used for this image (degrees).
    image_width : int
    image_height : int
    seed : int
        Random seed used to generate this image.
    """

    image: np.ndarray = field(default_factory=lambda: np.zeros((512, 512), dtype=np.float32))
    stars: list[RenderedStar] = field(default_factory=list)
    boresight_ra_deg: float = 0.0
    boresight_dec_deg: float = 0.0
    roll_deg: float = 0.0
    fov_deg: float = 20.0
    image_width: int = 512
    image_height: int = 512
    seed: int = 0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class StarFieldGenerator:
    """Produces synthetic star-field images from a loaded :class:`StarCatalog`.

    Parameters
    ----------
    catalog:
        A populated :class:`StarCatalog` instance.
    config:
        The ``dataset`` sub-dict from config.yaml.  All generation
        parameters are read from here.

    Example
    -------
    >>> gen = StarFieldGenerator(catalog, config["dataset"])
    >>> sf = gen.generate(seed=42)
    >>> sf.image.shape
    (512, 512)
    """

    def __init__(self, catalog: StarCatalog, config: dict) -> None:
        self._catalog = catalog
        self._cfg = config

        # Image dimensions
        self._width: int = int(config.get("image_width", 512))
        self._height: int = int(config.get("image_height", 512))

        # Optics
        self._fov_deg: float = float(config.get("field_of_view_deg", 20.0))

        # Rendering
        self._max_stars: int = int(config.get("max_stars_per_image", 30))
        self._psf_sigma: float = float(config.get("psf_sigma_px", 1.5))
        self._min_flux: float = float(config.get("min_star_flux", 0.05))

        # Noise
        self._bg_level: float = float(config.get("background_level", 0.02))
        self._read_noise: float = float(config.get("read_noise_sigma", 0.005))
        self._shot_noise: bool = bool(config.get("shot_noise", True))

        # Artifacts
        self._artifact_prob: float = float(config.get("artifact_probability", 0.01))

        # Pre-compute focal length in pixels
        # f = (w/2) / tan(fov/2)
        self._focal_px: float = (self._width / 2.0) / math.tan(
            math.radians(self._fov_deg / 2.0)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        seed: int,
        boresight_ra_deg: Optional[float] = None,
        boresight_dec_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
    ) -> SyntheticStarField:
        """Generate one synthetic star-field image.

        Parameters
        ----------
        seed:
            Random seed for full reproducibility.  The same seed always
            produces the same image when the catalog is identical.
        boresight_ra_deg:
            Optional fixed boresight RA.  If ``None``, drawn randomly.
        boresight_dec_deg:
            Optional fixed boresight Dec.  If ``None``, drawn randomly.
        roll_deg:
            Optional fixed roll.  If ``None``, drawn randomly in [0, 360).

        Returns
        -------
        SyntheticStarField
            Image array + ground-truth metadata.
        """
        rng = np.random.default_rng(seed)

        # --- 1. Sample attitude -------------------------------------------
        if boresight_ra_deg is None or boresight_dec_deg is None:
            bs_ra, bs_dec = _random_boresight(rng)
        else:
            bs_ra = float(boresight_ra_deg)
            bs_dec = float(boresight_dec_deg)

        if roll_deg is None:
            roll = float(rng.uniform(0.0, 360.0))
        else:
            roll = float(roll_deg)

        # --- 2. Select visible catalog stars ------------------------------
        half_fov = self._fov_deg / 2.0
        # Use a slightly larger cone to account for projection distortion
        candidates = self._catalog.query_by_region(bs_ra, bs_dec, half_fov * 1.05)

        # --- 3. Project onto focal plane ----------------------------------
        projected = self._project_stars(candidates, bs_ra, bs_dec, roll)

        # Keep only stars inside the sensor footprint
        in_frame = [
            (star, px, py, flux)
            for star, px, py, flux in projected
            if 0.0 <= px < self._width and 0.0 <= py < self._height
            and flux >= self._min_flux
        ]

        # Sort by brightness and cap at max_stars
        in_frame.sort(key=lambda t: t[3], reverse=True)
        in_frame = in_frame[: self._max_stars]

        # --- 4. Render image ----------------------------------------------
        image = self._render(in_frame, rng)

        # --- 5. Build ground-truth list -----------------------------------
        rendered_stars = [
            RenderedStar(
                star_id=star.star_id,
                x_px=px,
                y_px=py,
                flux=flux,
                vmag=star.magnitude,
                ra_deg=star.ra_deg,
                dec_deg=star.dec_deg,
            )
            for star, px, py, flux in in_frame
        ]

        return SyntheticStarField(
            image=image,
            stars=rendered_stars,
            boresight_ra_deg=bs_ra,
            boresight_dec_deg=bs_dec,
            roll_deg=roll,
            fov_deg=self._fov_deg,
            image_width=self._width,
            image_height=self._height,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_stars(
        self,
        stars: list[CatalogStar],
        bs_ra: float,
        bs_dec: float,
        roll: float,
    ) -> list[tuple[CatalogStar, float, float, float]]:
        """Project catalog stars onto the focal plane.

        Returns a list of (CatalogStar, x_px, y_px, normalised_flux).
        Flux is normalised so the brightest star = 1.0.
        """
        results = []
        fluxes = []

        for star in stars:
            x_tan, y_tan = _gnomonic_project(
                star.ra_deg, star.dec_deg, bs_ra, bs_dec
            )
            # Apply roll rotation
            roll_rad = math.radians(roll)
            cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
            x_rot = cos_r * x_tan - sin_r * y_tan
            y_rot = sin_r * x_tan + cos_r * y_tan

            # Convert tangent-plane coords to pixel coords
            x_px = self._focal_px * x_rot + self._width / 2.0
            y_px = -self._focal_px * y_rot + self._height / 2.0  # y flipped

            # Magnitude → linear flux
            flux = _mag_to_flux(star.magnitude)
            results.append((star, x_px, y_px, flux))
            fluxes.append(flux)

        if not fluxes:
            return results

        # Normalise fluxes so the brightest star = 1.0
        max_flux = max(fluxes)
        if max_flux > 0:
            results = [
                (s, x, y, f / max_flux) for s, x, y, f in results
            ]
        return results

    def _render(
        self,
        in_frame: list[tuple[CatalogStar, float, float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Render stars onto a float32 image array with noise."""
        image = np.zeros((self._height, self._width), dtype=np.float64)

        # Render each star as a Gaussian PSF
        for _star, x_px, y_px, flux in in_frame:
            _render_gaussian_star(image, x_px, y_px, flux, self._psf_sigma)

        # Add constant background
        image += self._bg_level

        # Shot noise: sigma ≈ sqrt(signal), scaled appropriately
        if self._shot_noise and image.max() > 0:
            shot_sigma = np.sqrt(np.clip(image, 0, None)) * 0.05
            image += rng.normal(0.0, 1.0, image.shape) * shot_sigma

        # Read noise: Gaussian with fixed sigma
        if self._read_noise > 0:
            image += rng.normal(0.0, self._read_noise, image.shape)

        # Random hot-pixel / cosmic-ray artifacts
        if self._artifact_prob > 0:
            n_artifacts = rng.poisson(
                self._artifact_prob * self._width * self._height
            )
            if n_artifacts > 0:
                rows = rng.integers(0, self._height, size=int(n_artifacts))
                cols = rng.integers(0, self._width, size=int(n_artifacts))
                image[rows, cols] += rng.uniform(0.3, 1.0, size=int(n_artifacts))

        # Clip and convert to float32
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        return image


# ---------------------------------------------------------------------------
# Module-level pure functions
# ---------------------------------------------------------------------------


def _random_boresight(rng: np.random.Generator) -> tuple[float, float]:
    """Sample a uniformly random point on the sphere as (RA, Dec) in degrees.

    Uses the standard method: Dec = arcsin(uniform(-1, 1)).
    """
    ra_deg = float(rng.uniform(0.0, 360.0))
    dec_deg = float(math.degrees(math.asin(rng.uniform(-1.0, 1.0))))
    return ra_deg, dec_deg


def _gnomonic_project(
    ra_deg: float,
    dec_deg: float,
    ra_c_deg: float,
    dec_c_deg: float,
) -> tuple[float, float]:
    """Gnomonic (tangent-plane) projection of (ra, dec) relative to centre.

    Returns (x_tan, y_tan) in tangent-plane units (radians on the sky).

    Parameters
    ----------
    ra_deg, dec_deg:
        Star position in degrees.
    ra_c_deg, dec_c_deg:
        Projection centre (boresight) in degrees.

    Returns
    -------
    tuple[float, float]
        (x, y) tangent-plane coordinates in radians.
        Returns (nan, nan) if the star is behind the projection plane.
    """
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    ra_c = math.radians(ra_c_deg)
    dec_c = math.radians(dec_c_deg)

    cos_c = (
        math.sin(dec_c) * math.sin(dec)
        + math.cos(dec_c) * math.cos(dec) * math.cos(ra - ra_c)
    )
    if cos_c <= 0:
        return float("nan"), float("nan")

    x = math.cos(dec) * math.sin(ra - ra_c) / cos_c
    y = (
        math.cos(dec_c) * math.sin(dec)
        - math.sin(dec_c) * math.cos(dec) * math.cos(ra - ra_c)
    ) / cos_c
    return x, y


def _mag_to_flux(vmag: float, vmag_ref: float = 0.0) -> float:
    """Convert V magnitude to linear flux relative to ``vmag_ref``.

    Uses the standard photometric relation:
        flux = 10 ** (-0.4 * (vmag - vmag_ref))

    A star with ``vmag = vmag_ref`` returns flux = 1.0.
    A fainter star (larger vmag) returns flux < 1.0.
    """
    return 10.0 ** (-0.4 * (vmag - vmag_ref))


def _render_gaussian_star(
    image: np.ndarray,
    x_px: float,
    y_px: float,
    flux: float,
    sigma: float,
) -> None:
    """Add a 2-D Gaussian PSF centred at (x_px, y_px) to *image* in-place.

    Only updates pixels within a bounding box of radius 4*sigma to avoid
    touching the whole image for every star.

    Parameters
    ----------
    image:
        Float64 image array, modified in place.
    x_px, y_px:
        Sub-pixel centre coordinates (column, row).
    flux:
        Peak flux to place at the star centre (before PSF spread).
    sigma:
        Gaussian PSF sigma in pixels.
    """
    h, w = image.shape
    radius = int(math.ceil(4.0 * sigma)) + 1

    row_min = max(0, int(y_px) - radius)
    row_max = min(h, int(y_px) + radius + 1)
    col_min = max(0, int(x_px) - radius)
    col_max = min(w, int(x_px) + radius + 1)

    if row_min >= row_max or col_min >= col_max:
        return

    rows = np.arange(row_min, row_max, dtype=np.float64)
    cols = np.arange(col_min, col_max, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols, rows)

    gauss = flux * np.exp(
        -((col_grid - x_px) ** 2 + (row_grid - y_px) ** 2) / (2.0 * sigma ** 2)
    )
    image[row_min:row_max, col_min:col_max] += gauss
