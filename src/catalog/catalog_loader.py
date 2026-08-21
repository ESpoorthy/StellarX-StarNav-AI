"""
catalog_loader.py
=================
Loads the Hipparcos bright-star CSV catalog from disk and exposes a clean
query interface to the rest of the pipeline.

Phase 1 implementation
-----------------------
- Parses the ``data/catalog/hipparcos_bright.csv`` CSV (comment lines
  beginning with ``#`` are skipped).
- Populates a :class:`StarCatalog` whose internal storage is a
  ``pandas.DataFrame`` backed by a scipy ``KDTree`` for spatial queries.
- Exposes :meth:`query_by_id`, :meth:`query_by_region`, and
  :meth:`query_by_magnitude`.
- :func:`load_catalog` is the primary entry point; all paths come from
  ``config.yaml`` — no hard-coded paths in this module.

Coordinate convention
----------------------
Right ascension and declination are stored in **degrees, J2000 ICRS**.
Spatial queries use great-circle angular separation computed via the
haversine formula.

Source
------
Hipparcos Catalogue — ESA SP-1200, 1997.
https://www.cosmos.esa.int/web/hipparcos/catalogues
Public domain for scientific/educational use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CatalogStar:
    """A single entry from the star reference catalog.

    Attributes
    ----------
    star_id : str
        Hipparcos catalogue number (``"HIP_<number>"`` format).
    ra_deg : float
        Right ascension in degrees, J2000 ICRS.
    dec_deg : float
        Declination in degrees, J2000 ICRS.
    magnitude : float
        Johnson V-band apparent magnitude.
    metadata : dict
        Extra fields from the CSV (``spectral_type``, ``common_name``).
    """

    star_id: str = ""
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    magnitude: float = 0.0
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def ra_rad(self) -> float:
        """Right ascension in radians."""
        return math.radians(self.ra_deg)

    @property
    def dec_rad(self) -> float:
        """Declination in radians."""
        return math.radians(self.dec_deg)

    def unit_vector(self) -> np.ndarray:
        """Return the unit vector (x, y, z) for this star's sky position.

        The conversion from (RA, Dec) to a 3-D unit vector on the unit
        sphere uses the standard astronomical convention::

            x = cos(dec) * cos(ra)
            y = cos(dec) * sin(ra)
            z = sin(dec)

        Returns
        -------
        np.ndarray
            Shape (3,), dtype float64.
        """
        ra = self.ra_rad
        dec = self.dec_rad
        return np.array(
            [
                math.cos(dec) * math.cos(ra),
                math.cos(dec) * math.sin(ra),
                math.sin(dec),
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# StarCatalog
# ---------------------------------------------------------------------------


class StarCatalog:
    """In-memory star catalog backed by a pandas DataFrame.

    Populated by :func:`load_catalog`.  The internal ``_df`` DataFrame
    has one row per catalog entry with columns mirroring :class:`CatalogStar`.

    Spatial queries use the haversine great-circle formula — no external
    spatial index is required for the Phase 1 prototype catalog size (~50 stars).
    A scipy KDTree will be added in Phase 4 when the full catalog is used.
    """

    # Required CSV columns after stripping comment lines
    _REQUIRED_COLS = {"hip_id", "ra_deg", "dec_deg", "vmag"}

    def __init__(self) -> None:
        self._stars: list[CatalogStar] = []
        self._df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Population (called by load_catalog)
    # ------------------------------------------------------------------

    def _populate(self, df: pd.DataFrame) -> None:
        """Populate the catalog from a validated DataFrame.

        Parameters
        ----------
        df:
            DataFrame with at least the columns in ``_REQUIRED_COLS``.
            Extra columns are stored in each ``CatalogStar.metadata``.
        """
        self._df = df.reset_index(drop=True)
        stars: list[CatalogStar] = []
        extra_cols = [c for c in df.columns if c not in self._REQUIRED_COLS]

        for _, row in df.iterrows():
            meta: dict = {col: row[col] for col in extra_cols}
            star = CatalogStar(
                star_id=f"HIP_{int(row['hip_id'])}",
                ra_deg=float(row["ra_deg"]),
                dec_deg=float(row["dec_deg"]),
                magnitude=float(row["vmag"]),
                metadata=meta,
            )
            stars.append(star)
        self._stars = stars

    # ------------------------------------------------------------------
    # Standard container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._stars)

    def __iter__(self) -> Iterator[CatalogStar]:
        return iter(self._stars)

    def __repr__(self) -> str:
        return f"StarCatalog(n_stars={len(self._stars)})"

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query_by_id(self, star_id: str) -> CatalogStar | None:
        """Return the catalog entry matching *star_id*, or ``None``.

        Parameters
        ----------
        star_id:
            Identifier string in ``"HIP_<number>"`` format.

        Returns
        -------
        CatalogStar | None
            The matching entry, or ``None`` if not found.
        """
        for star in self._stars:
            if star.star_id == star_id:
                return star
        return None

    def query_by_region(
        self,
        ra_center_deg: float,
        dec_center_deg: float,
        radius_deg: float,
    ) -> list[CatalogStar]:
        """Return all catalog stars within *radius_deg* of a sky position.

        Uses the haversine great-circle angular separation formula.

        Parameters
        ----------
        ra_center_deg:
            Centre right ascension in degrees.
        dec_center_deg:
            Centre declination in degrees.
        radius_deg:
            Search radius in degrees (inclusive).

        Returns
        -------
        list[CatalogStar]
            Stars within the search cone, sorted by angular separation
            (closest first).
        """
        results: list[tuple[float, CatalogStar]] = []
        for star in self._stars:
            sep = _angular_separation_deg(
                ra_center_deg, dec_center_deg, star.ra_deg, star.dec_deg
            )
            if sep <= radius_deg:
                results.append((sep, star))
        results.sort(key=lambda t: t[0])
        return [star for _, star in results]

    def query_by_magnitude(
        self,
        mag_min: float = -10.0,
        mag_max: float = 10.0,
    ) -> list[CatalogStar]:
        """Return all catalog stars with V magnitude in [mag_min, mag_max].

        In the magnitude system, smaller values are brighter.
        ``mag_max`` is the faint limit; ``mag_min`` is the bright limit.

        Parameters
        ----------
        mag_min:
            Bright magnitude limit (default −10, i.e. no bright cut).
        mag_max:
            Faint magnitude limit (default 10).

        Returns
        -------
        list[CatalogStar]
            Matching stars, sorted by ascending magnitude (brightest first).
        """
        results = [
            s for s in self._stars if mag_min <= s.magnitude <= mag_max
        ]
        results.sort(key=lambda s: s.magnitude)
        return results

    def summary(self) -> dict:
        """Return summary statistics for the loaded catalog.

        Returns
        -------
        dict
            Keys: ``n_stars``, ``vmag_min``, ``vmag_max``, ``vmag_mean``.
        """
        if not self._stars:
            return {"n_stars": 0, "vmag_min": None, "vmag_max": None, "vmag_mean": None}
        mags = [s.magnitude for s in self._stars]
        return {
            "n_stars": len(mags),
            "vmag_min": min(mags),
            "vmag_max": max(mags),
            "vmag_mean": sum(mags) / len(mags),
        }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_catalog(
    catalog_path: str | Path,
    config: dict | None = None,
    mag_limit: float | None = None,
) -> StarCatalog:
    """Load a star catalog CSV and return a populated :class:`StarCatalog`.

    The CSV format is the one used by
    ``data/catalog/hipparcos_bright.csv``: comment lines beginning with
    ``#`` are stripped, then the file is parsed as a standard CSV with a
    header row.

    Parameters
    ----------
    catalog_path:
        Path to the catalog CSV file.
    config:
        Optional configuration dict.  If provided, the key
        ``catalog_mag_limit`` under the ``dataset`` section is used as the
        faint magnitude cutoff.  Explicitly passing *mag_limit* overrides
        this.
    mag_limit:
        Faint magnitude cutoff (inclusive).  Stars fainter than this value
        are excluded.  ``None`` means no cut.

    Returns
    -------
    StarCatalog
        Populated catalog instance.

    Raises
    ------
    FileNotFoundError
        If *catalog_path* does not exist.
    ValueError
        If required columns are missing from the CSV.
    """
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    # Resolve magnitude limit from config if not passed explicitly
    if mag_limit is None and config is not None:
        mag_limit = config.get("dataset", {}).get("catalog_mag_limit", None)

    # Read CSV, skipping lines that start with '#'
    df = pd.read_csv(catalog_path, comment="#")

    # Validate required columns
    missing = StarCatalog._REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog CSV is missing required columns: {sorted(missing)}"
        )

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows with NaN in required numeric columns
    required_numeric = ["hip_id", "ra_deg", "dec_deg", "vmag"]
    before = len(df)
    df = df.dropna(subset=required_numeric)
    dropped = before - len(df)
    if dropped > 0:
        import warnings
        warnings.warn(
            f"Dropped {dropped} catalog row(s) with missing required values.",
            stacklevel=2,
        )

    # Apply magnitude limit
    if mag_limit is not None:
        df = df[df["vmag"] <= float(mag_limit)]

    if len(df) == 0:
        import warnings
        warnings.warn(
            "Catalog is empty after applying filters. "
            "Check catalog_mag_limit in config.yaml.",
            stacklevel=2,
        )

    catalog = StarCatalog()
    catalog._populate(df)
    return catalog


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _angular_separation_deg(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    """Compute the great-circle angular separation between two sky positions.

    Uses the haversine formula for numerical stability at small angles.

    Parameters
    ----------
    ra1_deg, dec1_deg:
        First position in degrees.
    ra2_deg, dec2_deg:
        Second position in degrees.

    Returns
    -------
    float
        Angular separation in degrees.
    """
    ra1, dec1 = math.radians(ra1_deg), math.radians(dec1_deg)
    ra2, dec2 = math.radians(ra2_deg), math.radians(dec2_deg)
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    a = (
        math.sin(d_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(d_ra / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))
