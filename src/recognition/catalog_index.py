"""
catalog_index.py — Phase 4
==========================
KD-tree indexed catalog for O(log n) angular-distance queries.

Uses scipy.spatial.KDTree on precomputed unit vectors.
Chord-distance formula: chord = 2*sin(angle/2), so angle = 2*arcsin(chord/2).
Precomputes all pairwise catalog angular distances at construction
(50 stars = 1225 pairs, fast).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from src.catalog.catalog_loader import CatalogStar, StarCatalog, load_catalog


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IndexedCatalogStar:
    """A catalog star with precomputed unit vector.

    Attributes
    ----------
    star : CatalogStar
        The original catalog entry.
    unit_vec : np.ndarray
        Shape (3,) unit vector in inertial frame, computed from (RA, Dec).
    index : int
        Position in the CatalogIndex star list.
    """

    star: CatalogStar
    unit_vec: np.ndarray
    index: int = 0


# ---------------------------------------------------------------------------
# CatalogIndex
# ---------------------------------------------------------------------------


class CatalogIndex:
    """KD-tree indexed catalog for efficient angular-distance queries.

    At construction, the index:
    - Builds a scipy KDTree on the unit-vector array (shape N×3).
    - Precomputes all pairwise angular distances (N*(N-1)/2 pairs).

    Queries use chord distance: for angular separation θ,
        chord = 2 * sin(θ/2)
    and the inverse:
        θ = 2 * arcsin(chord / 2)

    Parameters
    ----------
    catalog : StarCatalog
        A populated star catalog.
    """

    def __init__(self, catalog: StarCatalog) -> None:
        stars_list = list(catalog)
        self._indexed: list[IndexedCatalogStar] = []

        unit_vecs = []
        for i, star in enumerate(stars_list):
            uv = star.unit_vector()
            indexed = IndexedCatalogStar(star=star, unit_vec=uv, index=i)
            self._indexed.append(indexed)
            unit_vecs.append(uv)

        if len(unit_vecs) == 0:
            self._unit_vecs = np.zeros((0, 3), dtype=np.float64)
            self._kdtree = None
            self._pair_angles: dict[tuple[int, int], float] = {}
            return

        self._unit_vecs = np.array(unit_vecs, dtype=np.float64)
        self._kdtree = KDTree(self._unit_vecs)

        # Precompute all pairwise angles
        self._pair_angles: dict[tuple[int, int], float] = {}
        n = len(self._indexed)
        for i in range(n):
            for j in range(i + 1, n):
                dot = float(np.dot(self._unit_vecs[i], self._unit_vecs[j]))
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))
                self._pair_angles[(i, j)] = angle_deg

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def query_cone(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
    ) -> list[IndexedCatalogStar]:
        """Return all indexed stars within a cone around (ra_deg, dec_deg).

        Parameters
        ----------
        ra_deg, dec_deg : float
            Query centre in degrees.
        radius_deg : float
            Search cone half-angle in degrees.

        Returns
        -------
        list[IndexedCatalogStar]
            Stars inside the cone, sorted by angular separation (closest first).
        """
        if self._kdtree is None:
            return []

        # Build query unit vector
        ra_rad = math.radians(ra_deg)
        dec_rad = math.radians(dec_deg)
        q_vec = np.array([
            math.cos(dec_rad) * math.cos(ra_rad),
            math.cos(dec_rad) * math.sin(ra_rad),
            math.sin(dec_rad),
        ], dtype=np.float64)

        # Convert radius to chord distance threshold
        chord_thresh = 2.0 * math.sin(math.radians(radius_deg) / 2.0)

        indices = self._kdtree.query_ball_point(q_vec, r=chord_thresh)

        # Compute actual angular separations and sort
        results: list[tuple[float, IndexedCatalogStar]] = []
        for idx in indices:
            dot = float(np.dot(q_vec, self._unit_vecs[idx]))
            dot = max(-1.0, min(1.0, dot))
            angle_deg = math.degrees(math.acos(dot))
            results.append((angle_deg, self._indexed[idx]))

        results.sort(key=lambda t: t[0])
        return [star for _, star in results]

    def find_pairs_by_angle(
        self,
        angle_deg: float,
        tolerance_deg: float,
    ) -> list[tuple[int, int, float]]:
        """Find catalog star pairs with angular separation near angle_deg.

        Parameters
        ----------
        angle_deg : float
            Target angular separation in degrees.
        tolerance_deg : float
            Acceptable deviation from angle_deg in degrees.

        Returns
        -------
        list[tuple[int, int, float]]
            Each element is (index_i, index_j, actual_angle_deg) for pairs
            within tolerance. Sorted by |actual - target|.
        """
        lo = angle_deg - tolerance_deg
        hi = angle_deg + tolerance_deg
        results: list[tuple[float, int, int, float]] = []

        for (i, j), actual in self._pair_angles.items():
            if lo <= actual <= hi:
                diff = abs(actual - angle_deg)
                results.append((diff, i, j, actual))

        results.sort(key=lambda t: t[0])
        return [(i, j, actual) for _, i, j, actual in results]

    def query_angular_distance(self, i: int, j: int) -> float:
        """Return precomputed angular distance between catalog stars i and j.

        Parameters
        ----------
        i, j : int
            Indices into the catalog star list.

        Returns
        -------
        float
            Angular separation in degrees.
        """
        if i == j:
            return 0.0
        key = (min(i, j), max(i, j))
        return self._pair_angles.get(key, float("nan"))

    def get_by_catalog_index(self, i: int) -> IndexedCatalogStar:
        """Return the IndexedCatalogStar at position i.

        Parameters
        ----------
        i : int
            Index into the catalog list.

        Returns
        -------
        IndexedCatalogStar
        """
        return self._indexed[i]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stars(self) -> list[IndexedCatalogStar]:
        """All indexed catalog stars."""
        return self._indexed

    @property
    def unit_vectors(self) -> np.ndarray:
        """Unit vector array, shape (N, 3)."""
        return self._unit_vecs

    def __len__(self) -> int:
        return len(self._indexed)

    def __repr__(self) -> str:
        return f"CatalogIndex(n_stars={len(self._indexed)}, n_pairs={len(self._pair_angles)})"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_catalog_index(
    catalog_path: str | Path,
    config: Optional[dict] = None,
    mag_limit: Optional[float] = None,
) -> CatalogIndex:
    """Load a star catalog and build a CatalogIndex from it.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the Hipparcos CSV file.
    config : dict, optional
        Project configuration dict. Used to resolve mag_limit if not given.
    mag_limit : float, optional
        Faint magnitude cutoff. Overrides config value.

    Returns
    -------
    CatalogIndex
        Populated catalog index ready for queries.
    """
    catalog = load_catalog(catalog_path, config=config, mag_limit=mag_limit)
    return CatalogIndex(catalog)
