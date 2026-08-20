"""
test_star_detection.py
======================
Unit tests for src.preprocessing.star_detection and
src.preprocessing.image_preprocessing.

Tests are structured to cover the public API of each module.
Implementation of test bodies is deferred until Phase 2, when the
corresponding source functions are implemented.

Run with:
    pytest tests/test_star_detection.py
"""

from __future__ import annotations

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def blank_image() -> np.ndarray:
    """Return a 128×128 zero-filled float32 image."""
    return np.zeros((128, 128), dtype=np.float32)


@pytest.fixture()
def synthetic_star_image() -> np.ndarray:
    """Return a synthetic star-field image with known star positions.

    TODO (Phase 2): generate a realistic synthetic image with Gaussian
    point-spread-function blobs at known (x, y) coordinates so that
    detection tests can verify correctness quantitatively.
    """
    # Placeholder — replace with proper synthetic image generator in Phase 2.
    image = np.zeros((128, 128), dtype=np.float32)
    return image


@pytest.fixture()
def detection_config() -> dict:
    """Return a minimal star-detection configuration dict."""
    return {
        "min_brightness": 0.1,
        "min_area_px": 1,
        "max_stars": 50,
    }


@pytest.fixture()
def preprocessing_config() -> dict:
    """Return a minimal preprocessing configuration dict."""
    return {
        "noise_reduction": None,
        "normalization": None,
        "background_subtraction": None,
    }


# ---------------------------------------------------------------------------
# image_preprocessing tests
# ---------------------------------------------------------------------------


class TestLoadImage:
    """Tests for image_preprocessing.load_image."""

    def test_raises_not_implemented(self, tmp_path):
        """load_image should raise NotImplementedError until Phase 2."""
        from src.preprocessing.image_preprocessing import load_image

        with pytest.raises(NotImplementedError):
            load_image(tmp_path / "nonexistent.fits")

    # TODO (Phase 2): add tests for:
    #   - successful loading of a valid image file
    #   - FileNotFoundError on missing path
    #   - correct output dtype (float32)
    #   - correct output shape (2-D)


class TestSubtractBackground:
    """Tests for image_preprocessing.subtract_background."""

    def test_raises_not_implemented(self, blank_image):
        """subtract_background should raise NotImplementedError until Phase 2."""
        from src.preprocessing.image_preprocessing import subtract_background

        with pytest.raises(NotImplementedError):
            subtract_background(blank_image)

    # TODO (Phase 2): add tests for:
    #   - output shape matches input shape
    #   - uniform image background is reduced to near-zero
    #   - output dtype preserved


class TestNormalise:
    """Tests for image_preprocessing.normalise."""

    def test_raises_not_implemented(self, blank_image):
        """normalise should raise NotImplementedError until Phase 2."""
        from src.preprocessing.image_preprocessing import normalise

        with pytest.raises(NotImplementedError):
            normalise(blank_image)

    # TODO (Phase 2): add tests for:
    #   - output range within [0, 1] for min-max normalisation
    #   - mean ≈ 0 and std ≈ 1 for z-score normalisation
    #   - output shape matches input shape


# ---------------------------------------------------------------------------
# star_detection tests
# ---------------------------------------------------------------------------


class TestDetectStars:
    """Tests for star_detection.detect_stars."""

    def test_raises_not_implemented(self, blank_image, detection_config):
        """detect_stars should raise NotImplementedError until Phase 2."""
        from src.preprocessing.star_detection import detect_stars

        with pytest.raises(NotImplementedError):
            detect_stars(blank_image, detection_config)

    # TODO (Phase 2): add tests for:
    #   - blank image returns empty list
    #   - synthetic image with N known stars returns N candidates
    #   - detected positions are within tolerance of ground-truth centroids
    #   - max_stars config limit is respected
    #   - candidates are ordered by descending brightness


class TestComputeCentroid:
    """Tests for star_detection.compute_centroid."""

    def test_raises_not_implemented(self, blank_image):
        """compute_centroid should raise NotImplementedError until Phase 2."""
        from src.preprocessing.star_detection import compute_centroid

        with pytest.raises(NotImplementedError):
            compute_centroid(blank_image, bbox=(0, 0, 10, 10))

    # TODO (Phase 2): add tests for:
    #   - centroid of symmetric Gaussian blob is at its centre
    #   - sub-pixel precision within acceptable tolerance


class TestExtractFeatures:
    """Tests for star_detection.extract_features."""

    def test_raises_not_implemented(self, detection_config):
        """extract_features should raise NotImplementedError until Phase 3."""
        from src.preprocessing.star_detection import extract_features

        with pytest.raises(NotImplementedError):
            extract_features([], detection_config)

    # TODO (Phase 3): add tests for:
    #   - output shape matches expected feature dimensionality
    #   - features are invariant to known transformations (rotation, scale)
    #   - empty star list returns zero-length or sentinel feature array
