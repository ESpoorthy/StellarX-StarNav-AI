"""
catalog_loader.py
=================
Responsible for loading the star reference catalog from disk and exposing
a clean query interface to the rest of the pipeline.

Responsibility (planned)
------------------------
- Load a star catalog from the path specified in config.yaml.
- Parse catalog entries into structured Python objects.
- Provide query methods (by magnitude range, sky region, identifier, etc.).
- Abstract over the on-disk storage format (CSV, HDF5, SQLite — TBD).

Implementation note
-------------------
The catalog source and file format are to be determined during Phase 1.
No hard-coded catalog data or file paths should appear in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CatalogStar:
    """Represents a single entry in the star reference catalog.

    Attributes
    ----------
    star_id : str
        Unique identifier from the catalog (e.g. HD number, HIP number).
    ra_deg : float
        Right ascension in degrees (J2000 or epoch TBD).
    dec_deg : float
        Declination in degrees (J2000 or epoch TBD).
    magnitude : float
        Apparent visual magnitude.
    metadata : dict
        Optional additional fields from the catalog source.
    """

    star_id: str = ""
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    magnitude: float = 0.0
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StarCatalog:
    """In-memory representation of the loaded star catalog.

    Provides query methods used by the pattern matcher and navigation
    estimator.  The internal data structure (NumPy arrays, Pandas DataFrame,
    k-d tree, etc.) is to be determined during Phase 4 based on the
    required query patterns.
    """

    def __init__(self) -> None:
        # TODO (Phase 4): define the internal storage structure.
        self._stars: list[CatalogStar] = []

    def __len__(self) -> int:
        return len(self._stars)

    def __iter__(self) -> Iterator[CatalogStar]:
        return iter(self._stars)

    def query_by_id(self, star_id: str) -> CatalogStar | None:
        """Return the catalog entry for *star_id*, or None if not found.

        Parameters
        ----------
        star_id:
            Catalog identifier string.

        Returns
        -------
        CatalogStar | None

        Raises
        ------
        NotImplementedError
            Until implemented in Phase 4.
        """
        # TODO (Phase 4): implement identifier lookup.
        raise NotImplementedError("query_by_id is not yet implemented.")

    def query_by_region(
        self,
        ra_center_deg: float,
        dec_center_deg: float,
        radius_deg: float,
    ) -> list[CatalogStar]:
        """Return all catalog stars within *radius_deg* of a sky coordinate.

        Parameters
        ----------
        ra_center_deg:
            Centre right ascension in degrees.
        dec_center_deg:
            Centre declination in degrees.
        radius_deg:
            Search radius in degrees.

        Returns
        -------
        list[CatalogStar]

        Raises
        ------
        NotImplementedError
            Until implemented in Phase 4.
        """
        # TODO (Phase 4): implement spatial query (k-d tree or HEALPix).
        raise NotImplementedError("query_by_region is not yet implemented.")

    def query_by_magnitude(
        self,
        mag_min: float,
        mag_max: float,
    ) -> list[CatalogStar]:
        """Return all catalog stars with apparent magnitude in [mag_min, mag_max].

        Parameters
        ----------
        mag_min:
            Faint magnitude limit (larger value = fainter).
        mag_max:
            Bright magnitude limit (smaller value = brighter).

        Returns
        -------
        list[CatalogStar]

        Raises
        ------
        NotImplementedError
            Until implemented in Phase 4.
        """
        # TODO (Phase 4): implement magnitude-range query.
        raise NotImplementedError("query_by_magnitude is not yet implemented.")


def load_catalog(catalog_path: str | Path, config: dict) -> StarCatalog:
    """Load the star catalog from disk and return a StarCatalog instance.

    Parameters
    ----------
    catalog_path:
        Path to the catalog file (format TBD).
    config:
        Data configuration dict (``data`` section of config.yaml).

    Returns
    -------
    StarCatalog
        Populated catalog instance.

    Raises
    ------
    FileNotFoundError
        If *catalog_path* does not exist.
    NotImplementedError
        Until this function is implemented in Phase 4.
    """
    # TODO (Phase 4): implement catalog loading.
    #   Consider supporting multiple formats via a format key in config.
    raise NotImplementedError("load_catalog is not yet implemented.")
