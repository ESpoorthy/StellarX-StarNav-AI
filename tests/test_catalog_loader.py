"""
test_catalog_loader.py
======================
Unit tests for src.catalog.catalog_loader.

Covers:
- CSV loading with comment lines
- Required field validation
- Magnitude filtering
- CatalogStar properties (ra_rad, dec_rad, unit_vector)
- StarCatalog query methods (by_id, by_region, by_magnitude)
- load_catalog error cases (missing file, missing columns)
- StarCatalog.summary()

Run with:
    pytest tests/test_catalog_loader.py -v
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.catalog.catalog_loader import (
    CatalogStar,
    StarCatalog,
    _angular_separation_deg,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")


@pytest.fixture(scope="module")
def real_catalog() -> StarCatalog:
    """Load the real bundled Hipparcos catalog once for the whole module."""
    return load_catalog(CATALOG_PATH)


@pytest.fixture()
def minimal_csv(tmp_path: Path) -> Path:
    """Write a minimal 3-star catalog CSV to a temp file and return its path."""
    content = textwrap.dedent("""\
        # comment line — should be ignored
        hip_id,ra_deg,dec_deg,vmag,spectral_type,common_name
        32349,101.287,-16.716,-1.46,A1Vm,Sirius
        91262,279.235,38.784,0.03,A0Va,Vega
        97649,297.696,8.868,0.77,A7IV,Altair
    """)
    p = tmp_path / "test_catalog.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def minimal_catalog(minimal_csv: Path) -> StarCatalog:
    return load_catalog(minimal_csv)


# ---------------------------------------------------------------------------
# load_catalog — happy path
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_returns_star_catalog_instance(self, minimal_catalog):
        assert isinstance(minimal_catalog, StarCatalog)

    def test_correct_number_of_stars(self, minimal_catalog):
        assert len(minimal_catalog) == 3

    def test_real_catalog_has_many_stars(self, real_catalog):
        """The bundled catalog must have at least 30 entries."""
        assert len(real_catalog) >= 30

    def test_comment_lines_skipped(self, minimal_catalog):
        """No CatalogStar should have 'comment' in its star_id."""
        for star in minimal_catalog:
            assert "comment" not in star.star_id.lower()

    def test_star_ids_use_hip_prefix(self, minimal_catalog):
        for star in minimal_catalog:
            assert star.star_id.startswith("HIP_"), (
                f"Expected 'HIP_' prefix, got '{star.star_id}'"
            )

    def test_magnitude_filter(self, minimal_csv: Path):
        """Only stars with vmag <= mag_limit should be returned."""
        catalog = load_catalog(minimal_csv, mag_limit=0.5)
        for star in catalog:
            assert star.magnitude <= 0.5

    def test_magnitude_filter_excludes_all(self, minimal_csv: Path):
        """A very low mag_limit should return an empty (but valid) catalog."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            catalog = load_catalog(minimal_csv, mag_limit=-5.0)
        assert len(catalog) == 0

    def test_config_mag_limit(self, minimal_csv: Path):
        """mag_limit from config dict should be applied."""
        config = {"dataset": {"catalog_mag_limit": 0.5}}
        catalog = load_catalog(minimal_csv, config=config)
        for star in catalog:
            assert star.magnitude <= 0.5


# ---------------------------------------------------------------------------
# load_catalog — error cases
# ---------------------------------------------------------------------------


class TestLoadCatalogErrors:
    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "nonexistent.csv")

    def test_missing_required_column(self, tmp_path: Path):
        """A CSV missing 'vmag' should raise ValueError."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("hip_id,ra_deg,dec_deg\n1,0.0,0.0\n")
        with pytest.raises(ValueError, match="vmag"):
            load_catalog(bad_csv)


# ---------------------------------------------------------------------------
# CatalogStar properties
# ---------------------------------------------------------------------------


class TestCatalogStarProperties:
    def test_ra_rad_conversion(self):
        star = CatalogStar(ra_deg=180.0, dec_deg=0.0)
        assert math.isclose(star.ra_rad, math.pi, rel_tol=1e-9)

    def test_dec_rad_conversion(self):
        star = CatalogStar(ra_deg=0.0, dec_deg=90.0)
        assert math.isclose(star.dec_rad, math.pi / 2.0, rel_tol=1e-9)

    def test_unit_vector_shape(self):
        star = CatalogStar(ra_deg=0.0, dec_deg=0.0)
        uv = star.unit_vector()
        assert uv.shape == (3,)

    def test_unit_vector_magnitude(self):
        """Unit vector must have magnitude 1."""
        for ra, dec in [(0, 0), (90, 45), (180, -30), (270, 60)]:
            star = CatalogStar(ra_deg=ra, dec_deg=dec)
            mag = float(np.linalg.norm(star.unit_vector()))
            assert math.isclose(mag, 1.0, rel_tol=1e-9), (
                f"Unit vector magnitude {mag} != 1 for RA={ra}, Dec={dec}"
            )

    def test_unit_vector_equator_ra0(self):
        """RA=0, Dec=0 should give unit vector (1, 0, 0)."""
        star = CatalogStar(ra_deg=0.0, dec_deg=0.0)
        uv = star.unit_vector()
        assert np.allclose(uv, [1.0, 0.0, 0.0], atol=1e-9)

    def test_unit_vector_north_pole(self):
        """Dec=90 should give unit vector (0, 0, 1)."""
        star = CatalogStar(ra_deg=0.0, dec_deg=90.0)
        uv = star.unit_vector()
        assert np.allclose(uv, [0.0, 0.0, 1.0], atol=1e-9)

    def test_metadata_defaults_to_empty_dict(self):
        star = CatalogStar()
        assert star.metadata == {}

    def test_extra_metadata_stored(self, minimal_catalog):
        """Spectral type and common name must be in metadata."""
        sirius = minimal_catalog.query_by_id("HIP_32349")
        assert sirius is not None
        assert "spectral_type" in sirius.metadata
        assert "common_name" in sirius.metadata


# ---------------------------------------------------------------------------
# StarCatalog — basic container behaviour
# ---------------------------------------------------------------------------


class TestStarCatalogContainer:
    def test_len(self, minimal_catalog):
        assert len(minimal_catalog) == 3

    def test_iter_yields_catalog_stars(self, minimal_catalog):
        for star in minimal_catalog:
            assert isinstance(star, CatalogStar)

    def test_repr_contains_n_stars(self, minimal_catalog):
        assert "3" in repr(minimal_catalog)


# ---------------------------------------------------------------------------
# StarCatalog.query_by_id
# ---------------------------------------------------------------------------


class TestQueryById:
    def test_known_star_found(self, minimal_catalog):
        star = minimal_catalog.query_by_id("HIP_32349")
        assert star is not None
        assert star.star_id == "HIP_32349"

    def test_unknown_id_returns_none(self, minimal_catalog):
        assert minimal_catalog.query_by_id("HIP_99999") is None

    def test_sirius_magnitude(self, minimal_catalog):
        sirius = minimal_catalog.query_by_id("HIP_32349")
        assert sirius is not None
        assert math.isclose(sirius.magnitude, -1.46, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# StarCatalog.query_by_region
# ---------------------------------------------------------------------------


class TestQueryByRegion:
    def test_returns_list(self, minimal_catalog):
        result = minimal_catalog.query_by_region(0.0, 0.0, 180.0)
        assert isinstance(result, list)

    def test_large_radius_returns_all(self, minimal_catalog):
        """A 180° radius must encompass the whole sky."""
        result = minimal_catalog.query_by_region(0.0, 0.0, 180.0)
        assert len(result) == len(minimal_catalog)

    def test_zero_radius_may_return_empty_or_exact(self, minimal_catalog):
        """A 0° radius returns at most one star (unlikely exact match)."""
        result = minimal_catalog.query_by_region(0.0, 0.0, 0.0)
        assert len(result) <= 1

    def test_result_sorted_by_separation(self, minimal_catalog):
        """Results must be ordered nearest-first."""
        results = minimal_catalog.query_by_region(279.235, 38.784, 180.0)
        # Vega (HIP_91262) is at (279.235, 38.784), so it should be first
        assert results[0].star_id == "HIP_91262"

    def test_stars_within_radius_are_actually_close(self, minimal_catalog):
        """Every returned star must be within the requested radius."""
        ra_c, dec_c, radius = 279.235, 38.784, 30.0
        for star in minimal_catalog.query_by_region(ra_c, dec_c, radius):
            sep = _angular_separation_deg(ra_c, dec_c, star.ra_deg, star.dec_deg)
            assert sep <= radius + 1e-9, (
                f"Star {star.star_id} at sep={sep:.4f}° exceeds radius={radius}°"
            )


# ---------------------------------------------------------------------------
# StarCatalog.query_by_magnitude
# ---------------------------------------------------------------------------


class TestQueryByMagnitude:
    def test_returns_list(self, minimal_catalog):
        assert isinstance(minimal_catalog.query_by_magnitude(), list)

    def test_all_returned_within_range(self, minimal_catalog):
        for star in minimal_catalog.query_by_magnitude(mag_min=-2.0, mag_max=0.5):
            assert -2.0 <= star.magnitude <= 0.5

    def test_sorted_brightest_first(self, minimal_catalog):
        results = minimal_catalog.query_by_magnitude()
        mags = [s.magnitude for s in results]
        assert mags == sorted(mags), "Results should be sorted by ascending magnitude"

    def test_no_results_for_impossible_range(self, minimal_catalog):
        result = minimal_catalog.query_by_magnitude(mag_min=50.0, mag_max=60.0)
        assert result == []


# ---------------------------------------------------------------------------
# StarCatalog.summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_keys(self, real_catalog):
        s = real_catalog.summary()
        assert {"n_stars", "vmag_min", "vmag_max", "vmag_mean"} <= s.keys()

    def test_summary_n_stars(self, real_catalog):
        s = real_catalog.summary()
        assert s["n_stars"] == len(real_catalog)

    def test_summary_vmag_min_le_max(self, real_catalog):
        s = real_catalog.summary()
        assert s["vmag_min"] <= s["vmag_max"]

    def test_summary_empty_catalog(self):
        empty = StarCatalog()
        s = empty.summary()
        assert s["n_stars"] == 0
        assert s["vmag_min"] is None


# ---------------------------------------------------------------------------
# _angular_separation_deg
# ---------------------------------------------------------------------------


class TestAngularSeparation:
    def test_same_point_is_zero(self):
        sep = _angular_separation_deg(45.0, 30.0, 45.0, 30.0)
        assert math.isclose(sep, 0.0, abs_tol=1e-9)

    def test_poles_are_90_from_equator(self):
        sep = _angular_separation_deg(0.0, 0.0, 0.0, 90.0)
        assert math.isclose(sep, 90.0, rel_tol=1e-6)

    def test_antipodal_points_are_180(self):
        sep = _angular_separation_deg(0.0, 0.0, 180.0, 0.0)
        assert math.isclose(sep, 180.0, rel_tol=1e-6)

    def test_symmetric(self):
        sep1 = _angular_separation_deg(10.0, 20.0, 30.0, 40.0)
        sep2 = _angular_separation_deg(30.0, 40.0, 10.0, 20.0)
        assert math.isclose(sep1, sep2, rel_tol=1e-9)

    def test_known_value(self):
        """Sirius–Vega separation should be roughly 158° (verified via haversine)."""
        sep = _angular_separation_deg(101.287, -16.716, 279.235, 38.784)
        assert 150.0 < sep < 165.0
