"""
test_pattern_matching.py
========================
Unit tests for src.catalog.catalog_loader (query methods).

Note: src/catalog/pattern_matcher.py was a dead stub and has been removed.
The live pattern matching pipeline is in src/recognition/pattern_matcher.py
and is tested in tests/test_phase4_recognition.py.

Run with:
    pytest tests/test_pattern_matching.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.catalog.catalog_loader import CatalogStar, StarCatalog, load_catalog
from src.models.inference import RecognitionResult


CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_catalog() -> StarCatalog:
    """Full bundled Hipparcos catalog, loaded once per module."""
    return load_catalog(CATALOG_PATH)


@pytest.fixture()
def empty_catalog() -> StarCatalog:
    return StarCatalog()


@pytest.fixture()
def sample_star() -> CatalogStar:
    return CatalogStar(
        star_id="HIP_32349",
        ra_deg=101.287,
        dec_deg=-16.716,
        magnitude=-1.46,
        metadata={"common_name": "Sirius"},
    )


@pytest.fixture()
def evaluation_config() -> dict:
    return {"confidence_threshold": 0.5, "top_k": 5}


# ---------------------------------------------------------------------------
# StarCatalog — basic container (Phase 1 implemented)
# ---------------------------------------------------------------------------


class TestStarCatalogInit:
    def test_empty_catalog_has_zero_length(self, empty_catalog):
        assert len(empty_catalog) == 0

    def test_empty_catalog_is_iterable(self, empty_catalog):
        assert list(empty_catalog) == []

    def test_real_catalog_nonzero_length(self, real_catalog):
        assert len(real_catalog) > 0

    def test_real_catalog_contains_catalog_stars(self, real_catalog):
        for star in real_catalog:
            assert isinstance(star, CatalogStar)


# ---------------------------------------------------------------------------
# StarCatalog.query_by_id (Phase 1 implemented)
# ---------------------------------------------------------------------------


class TestCatalogQueryById:
    def test_sirius_found_by_id(self, real_catalog):
        """Sirius (HIP 32349) must be present in the bundled catalog."""
        star = real_catalog.query_by_id("HIP_32349")
        assert star is not None

    def test_returned_star_has_correct_id(self, real_catalog):
        star = real_catalog.query_by_id("HIP_32349")
        assert star.star_id == "HIP_32349"

    def test_unknown_id_returns_none(self, real_catalog):
        assert real_catalog.query_by_id("HIP_999999") is None

    def test_unknown_id_returns_none_on_empty(self, empty_catalog):
        assert empty_catalog.query_by_id("HIP_32349") is None

    # Phase 4 — additional lookup tests to be added:
    # TODO (Phase 4): verify that lookup is O(1) or O(log n) via index


# ---------------------------------------------------------------------------
# StarCatalog.query_by_region (Phase 1 implemented)
# ---------------------------------------------------------------------------


class TestCatalogQueryByRegion:
    def test_large_radius_returns_results(self, real_catalog):
        """A 180° cone must return at least some stars."""
        result = real_catalog.query_by_region(0.0, 0.0, 180.0)
        assert len(result) > 0

    def test_zero_radius_returns_at_most_one(self, real_catalog):
        result = real_catalog.query_by_region(0.0, 0.0, 0.0)
        assert len(result) <= 1

    def test_empty_catalog_returns_empty_list(self, empty_catalog):
        result = empty_catalog.query_by_region(0.0, 0.0, 90.0)
        assert result == []

    def test_returns_list(self, real_catalog):
        assert isinstance(real_catalog.query_by_region(0.0, 0.0, 10.0), list)

    # Phase 4 — precision tests to be added once the k-d tree is in place:
    # TODO (Phase 4): verify separation threshold with sub-arcsecond precision


# ---------------------------------------------------------------------------
# load_catalog — Phase 1 implemented
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_loads_bundled_catalog(self):
        catalog = load_catalog(CATALOG_PATH)
        assert isinstance(catalog, StarCatalog)
        assert len(catalog) >= 30

    def test_required_fields_present(self):
        catalog = load_catalog(CATALOG_PATH)
        for star in catalog:
            assert star.star_id.startswith("HIP_")
            assert isinstance(star.ra_deg, float)
            assert isinstance(star.dec_deg, float)
            assert isinstance(star.magnitude, float)

    def test_ra_in_valid_range(self):
        catalog = load_catalog(CATALOG_PATH)
        for star in catalog:
            assert 0.0 <= star.ra_deg < 360.0, (
                f"{star.star_id} has invalid RA={star.ra_deg}"
            )

    def test_dec_in_valid_range(self):
        catalog = load_catalog(CATALOG_PATH)
        for star in catalog:
            assert -90.0 <= star.dec_deg <= 90.0, (
                f"{star.star_id} has invalid Dec={star.dec_deg}"
            )

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "missing.csv")

    def test_missing_column_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("hip_id,ra_deg,dec_deg\n1,0.0,0.0\n")
        with pytest.raises(ValueError):
            load_catalog(bad)

    def test_mag_limit_filters_faint_stars(self):
        catalog = load_catalog(CATALOG_PATH, mag_limit=1.0)
        for star in catalog:
            assert star.magnitude <= 1.0

    def test_sirius_is_brightest(self):
        """Sirius (vmag ≈ -1.46) should be the brightest star in the catalog."""
        catalog = load_catalog(CATALOG_PATH)
        brightest = min(catalog, key=lambda s: s.magnitude)
        assert brightest.star_id == "HIP_32349", (
            f"Expected Sirius (HIP_32349) to be brightest, got {brightest.star_id}"
        )


# ---------------------------------------------------------------------------
# Note: src/catalog/pattern_matcher.py was a dead stub and has been removed.
# The live pattern matching pipeline is in src/recognition/pattern_matcher.py
# and is tested in tests/test_phase4_recognition.py.
# ---------------------------------------------------------------------------
