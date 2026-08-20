"""
test_pattern_matching.py
========================
Unit tests for src.catalog.catalog_loader and src.catalog.pattern_matcher.

Tests are structured to cover the public API of each module.
Implementation of test bodies is deferred until Phase 4, when the
corresponding source functions are implemented.

Run with:
    pytest tests/test_pattern_matching.py
"""

from __future__ import annotations

import pytest

from src.catalog.catalog_loader import CatalogStar, StarCatalog
from src.models.inference import RecognitionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_catalog() -> StarCatalog:
    """Return an empty StarCatalog instance."""
    return StarCatalog()


@pytest.fixture()
def sample_star() -> CatalogStar:
    """Return a sample CatalogStar for use in tests."""
    return CatalogStar(
        star_id="HIP_001",
        ra_deg=83.82,
        dec_deg=-5.39,
        magnitude=0.42,
    )


@pytest.fixture()
def recognition_result_high_confidence() -> RecognitionResult:
    """Return a RecognitionResult with a high-confidence pattern ID."""
    import numpy as np

    return RecognitionResult(
        pattern_id="PAT_001",
        confidence=0.95,
        raw_output=None,
        latency_ms=12.5,
    )


@pytest.fixture()
def recognition_result_low_confidence() -> RecognitionResult:
    """Return a RecognitionResult with sub-threshold confidence."""
    import numpy as np

    return RecognitionResult(
        pattern_id=None,
        confidence=0.10,
        raw_output=None,
        latency_ms=11.0,
    )


@pytest.fixture()
def evaluation_config() -> dict:
    """Return a minimal evaluation configuration dict."""
    return {"confidence_threshold": 0.5, "top_k": 5}


# ---------------------------------------------------------------------------
# StarCatalog tests
# ---------------------------------------------------------------------------


class TestStarCatalogInit:
    """Tests for StarCatalog initialisation."""

    def test_empty_catalog_has_zero_length(self, empty_catalog):
        """A freshly created StarCatalog should have length 0."""
        assert len(empty_catalog) == 0

    def test_empty_catalog_is_iterable(self, empty_catalog):
        """Iterating over an empty catalog should yield no items."""
        assert list(empty_catalog) == []


class TestCatalogQueryById:
    """Tests for StarCatalog.query_by_id."""

    def test_raises_not_implemented(self, empty_catalog):
        """query_by_id should raise NotImplementedError until Phase 4."""
        with pytest.raises(NotImplementedError):
            empty_catalog.query_by_id("HIP_001")

    # TODO (Phase 4): add tests for:
    #   - known ID returns the correct CatalogStar
    #   - unknown ID returns None
    #   - lookup is case-sensitive / case-insensitive (TBD)


class TestCatalogQueryByRegion:
    """Tests for StarCatalog.query_by_region."""

    def test_raises_not_implemented(self, empty_catalog):
        """query_by_region should raise NotImplementedError until Phase 4."""
        with pytest.raises(NotImplementedError):
            empty_catalog.query_by_region(
                ra_center_deg=0.0, dec_center_deg=0.0, radius_deg=5.0
            )

    # TODO (Phase 4): add tests for:
    #   - stars within radius are returned
    #   - stars outside radius are excluded
    #   - edge case: star exactly on the radius boundary


class TestLoadCatalog:
    """Tests for catalog_loader.load_catalog."""

    def test_raises_not_implemented(self, tmp_path, evaluation_config):
        """load_catalog should raise NotImplementedError until Phase 4."""
        from src.catalog.catalog_loader import load_catalog

        with pytest.raises(NotImplementedError):
            load_catalog(tmp_path / "catalog.csv", evaluation_config)

    # TODO (Phase 4): add tests for:
    #   - valid catalog file is loaded and length > 0
    #   - FileNotFoundError on missing path
    #   - all required fields (star_id, ra_deg, dec_deg, magnitude) are present


# ---------------------------------------------------------------------------
# pattern_matcher tests
# ---------------------------------------------------------------------------


class TestMatchPattern:
    """Tests for pattern_matcher.match_pattern."""

    def test_raises_not_implemented(
        self,
        recognition_result_high_confidence,
        empty_catalog,
        evaluation_config,
    ):
        """match_pattern should raise NotImplementedError until Phase 4."""
        from src.catalog.pattern_matcher import match_pattern

        with pytest.raises(NotImplementedError):
            match_pattern(
                recognition_result_high_confidence,
                empty_catalog,
                evaluation_config,
            )

    # TODO (Phase 4): add tests for:
    #   - high-confidence result with matching catalog entry → is_confident=True
    #   - low-confidence result → is_confident=False
    #   - result with pattern_id=None → MatchResult with matched_star=None
    #   - match confidence score is within [0.0, 1.0]
