"""
test_star_detection.py
======================
Unit tests for:
  - src.preprocessing.image_preprocessing  (Phase 2 pipeline steps)
  - src.preprocessing.star_detection       (Phase 2 detection + centroiding)

All tests use synthetic data produced in-process — no disk images are
required except for the load_image round-trip tests.

Run with:
    pytest tests/test_star_detection.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.image_preprocessing import (
    load_image,
    subtract_background,
    reduce_noise,
    normalise,
    preprocess,
)
from src.preprocessing.star_detection import (
    StarCandidate,
    compute_centroid,
    detect_stars,
    extract_features,
)
from src.preprocessing.star_field_generator import (
    StarFieldGenerator,
    _render_gaussian_star,
)
from src.catalog.catalog_loader import load_catalog


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")

# Config matching config.yaml values — used across multiple test classes
DETECTION_CONFIG = {
    "threshold_method":     "absolute",
    "min_brightness":       0.05,
    "sigma_clip_k":         5.0,
    "min_area_px":          1,
    "max_area_px":          200,
    "min_peak_brightness":  0.04,
    "max_stars":            50,
    "centroid_method":      "intensity_weighted",
    "centroid_half_window": 5,
}

PREPROCESS_CONFIG = {
    "preprocessing": {
        "background_subtraction": True,
        "background_method":      "median_filter",
        "background_filter_size": 31,
        "noise_reduction":        True,
        "noise_method":           "gaussian",
        "noise_sigma":            0.8,
        "normalization":          "min_max",
    }
}


def _make_star_image(
    height: int = 128,
    width: int = 128,
    stars: list[tuple[float, float, float]] | None = None,
    background: float = 0.02,
    noise_sigma: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Create a synthetic greyscale star-field image with known star positions.

    Parameters
    ----------
    height, width : int
        Image dimensions.
    stars : list of (x, y, flux) or None
        If None, three stars are placed at fixed positions.
    background : float
        Constant background level.
    noise_sigma : float
        Read-noise sigma.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    image : np.ndarray
        float32 (H, W) array in [0, 1].
    stars : list of (x, y, flux)
        Ground-truth star positions and fluxes.
    """
    if stars is None:
        stars = [
            (32.0,  32.0,  1.0),   # bright
            (80.0,  64.0,  0.5),   # medium
            (100.0, 100.0, 0.25),  # faint
        ]

    rng = np.random.default_rng(seed)
    image = np.full((height, width), background, dtype=np.float64)

    for x, y, flux in stars:
        _render_gaussian_star(image, x, y, flux, sigma=1.5)

    if noise_sigma > 0:
        image += rng.normal(0.0, noise_sigma, image.shape)

    return np.clip(image, 0.0, 1.0).astype(np.float32), stars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def blank_image() -> np.ndarray:
    return np.zeros((128, 128), dtype=np.float32)


@pytest.fixture()
def background_image() -> np.ndarray:
    """Uniform background with no stars."""
    return np.full((128, 128), 0.02, dtype=np.float32)


@pytest.fixture()
def star_image() -> tuple[np.ndarray, list]:
    return _make_star_image()


@pytest.fixture()
def single_star_image() -> tuple[np.ndarray, tuple[float, float, float]]:
    """One star at (64, 64) with flux=1.0."""
    img, stars = _make_star_image(stars=[(64.0, 64.0, 1.0)], noise_sigma=0.0)
    return img, stars[0]


# ===========================================================================
# load_image
# ===========================================================================

class TestLoadImage:

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.png")

    def test_raises_value_error_unsupported_format(self, tmp_path):
        bad = tmp_path / "img.jpg"
        bad.write_bytes(b"\xff\xd8\xff")
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(bad)

    def test_round_trip_shape(self, tmp_path):
        import cv2
        img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        path = tmp_path / "t.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        loaded = load_image(path)
        assert loaded.shape == img.shape

    def test_round_trip_dtype(self, tmp_path):
        import cv2
        img = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        path = tmp_path / "t.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        assert load_image(path).dtype == np.float32

    def test_round_trip_values_close(self, tmp_path):
        import cv2
        img = np.random.default_rng(2).random((64, 64)).astype(np.float32)
        path = tmp_path / "t.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        loaded = load_image(path)
        assert float(np.abs(loaded - img).max()) < 0.002

    def test_values_in_unit_range(self, tmp_path):
        import cv2
        img = np.random.default_rng(3).random((64, 64)).astype(np.float32)
        path = tmp_path / "t.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        loaded = load_image(path)
        assert loaded.min() >= 0.0
        assert loaded.max() <= 1.0


# ===========================================================================
# subtract_background
# ===========================================================================

class TestSubtractBackground:

    def test_output_shape_preserved(self, star_image):
        img, _ = star_image
        result = subtract_background(img)
        assert result.shape == img.shape

    def test_output_dtype_float32(self, star_image):
        img, _ = star_image
        assert subtract_background(img).dtype == np.float32

    def test_output_clipped_to_unit_range(self, star_image):
        img, _ = star_image
        result = subtract_background(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_background_reduced(self, background_image):
        """A pure background image should be reduced close to zero."""
        result = subtract_background(background_image, method="median_filter", filter_size=31)
        # After median-filter subtraction of a uniform field, result should be ≈ 0
        assert float(result.mean()) < 0.005

    def test_constant_method_works(self, star_image):
        img, _ = star_image
        result = subtract_background(img, method="constant")
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_stars_survive_background_subtraction(self):
        """Bright stars should still be present after background removal."""
        img, stars = _make_star_image(
            stars=[(64.0, 64.0, 1.0)], background=0.05, noise_sigma=0.0
        )
        result = subtract_background(img, method="median_filter", filter_size=15)
        # Peak near the star should be well above zero after subtraction
        cx, cy = int(64), int(64)
        peak = float(result[cy - 3:cy + 4, cx - 3:cx + 4].max())
        assert peak > 0.3, f"Star peak after background subtraction too low: {peak}"

    def test_unknown_method_raises(self, star_image):
        img, _ = star_image
        with pytest.raises(ValueError, match="Unknown"):
            subtract_background(img, method="fft_magic")


# ===========================================================================
# reduce_noise
# ===========================================================================

class TestReduceNoise:

    def test_output_shape_preserved(self, star_image):
        img, _ = star_image
        assert reduce_noise(img).shape == img.shape

    def test_output_dtype_float32(self, star_image):
        img, _ = star_image
        assert reduce_noise(img).dtype == np.float32

    def test_gaussian_reduces_noise_variance(self):
        """Gaussian smoothing must reduce pixel variance."""
        rng = np.random.default_rng(42)
        noisy = rng.random((128, 128)).astype(np.float32)
        smoothed = reduce_noise(noisy, method="gaussian", sigma=2.0)
        assert float(smoothed.var()) < float(noisy.var())

    def test_none_method_returns_copy(self, star_image):
        img, _ = star_image
        result = reduce_noise(img, method="none")
        assert np.array_equal(result, img)
        assert result is not img  # must be a copy

    def test_zero_sigma_returns_copy(self, star_image):
        img, _ = star_image
        result = reduce_noise(img, sigma=0.0)
        assert np.array_equal(result, img)

    def test_output_clipped_to_unit_range(self, star_image):
        img, _ = star_image
        result = reduce_noise(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_unknown_method_raises(self, star_image):
        img, _ = star_image
        with pytest.raises(ValueError, match="Unknown"):
            reduce_noise(img, method="wavelet_magic")


# ===========================================================================
# normalise
# ===========================================================================

class TestNormalise:

    def test_output_shape_preserved(self, star_image):
        img, _ = star_image
        assert normalise(img).shape == img.shape

    def test_output_dtype_float32(self, star_image):
        img, _ = star_image
        assert normalise(img).dtype == np.float32

    def test_min_max_output_range(self):
        """Min-max normalisation must produce output with min ≈ 0 and max ≈ 1."""
        img, _ = _make_star_image(noise_sigma=0.0)
        result = normalise(img, method="min_max")
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_min_max_max_near_one(self):
        img, _ = _make_star_image(noise_sigma=0.0)
        result = normalise(img, method="min_max")
        assert result.max() > 0.9

    def test_z_score_mean_near_zero(self):
        img, _ = _make_star_image()
        result = normalise(img, method="z_score")
        assert math.isclose(float(result.mean()), 0.0, abs_tol=0.05)

    def test_z_score_std_near_one(self):
        img, _ = _make_star_image()
        result = normalise(img, method="z_score")
        assert math.isclose(float(result.std()), 1.0, rel_tol=0.05)

    def test_uniform_image_returns_copy(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        result = normalise(img, method="min_max")
        assert result.shape == img.shape

    def test_unknown_method_raises(self, star_image):
        img, _ = star_image
        with pytest.raises(ValueError, match="Unknown"):
            normalise(img, method="histogram_equalization")


# ===========================================================================
# preprocess  (integration)
# ===========================================================================

class TestPreprocess:

    def test_returns_float32_array(self, tmp_path):
        import cv2
        img, _ = _make_star_image()
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        result = preprocess(path, PREPROCESS_CONFIG)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self, tmp_path):
        import cv2
        img, _ = _make_star_image()
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        result = preprocess(path, PREPROCESS_CONFIG)
        assert result.shape == img.shape

    def test_output_in_unit_range(self, tmp_path):
        import cv2
        img, _ = _make_star_image()
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        result = preprocess(path, PREPROCESS_CONFIG)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess(tmp_path / "missing.png", PREPROCESS_CONFIG)

    def test_background_subtraction_disabled(self, tmp_path):
        import cv2
        img, _ = _make_star_image()
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), (img * 65535).astype(np.uint16))
        cfg = {"preprocessing": {
            "background_subtraction": False,
            "noise_reduction": False,
            "normalization": "none",
        }}
        result = preprocess(path, cfg)
        # Without any processing, result should be close to loaded image
        loaded = load_image(path)
        assert float(np.abs(result - loaded).max()) < 0.002


# ===========================================================================
# compute_centroid
# ===========================================================================

class TestComputeCentroid:

    def test_output_is_two_floats(self, single_star_image):
        img, _ = single_star_image
        cx, cy = compute_centroid(img, bbox=(55, 55, 73, 73))
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_centroid_near_known_position(self, single_star_image):
        """Star at (64, 64) — centroid should be within 0.5 px."""
        img, (x, y, _) = single_star_image
        cx, cy = compute_centroid(
            img,
            bbox=(int(y) - 8, int(x) - 8, int(y) + 9, int(x) + 9),
            peak_row=int(y), peak_col=int(x),
        )
        assert abs(cx - x) < 0.5, f"x centroid error {abs(cx - x):.3f} > 0.5 px"
        assert abs(cy - y) < 0.5, f"y centroid error {abs(cy - y):.3f} > 0.5 px"

    def test_symmetric_gaussian_centroid_at_peak(self):
        """For a perfectly symmetric Gaussian blob, centroid == peak."""
        img = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img, 32.0, 32.0, 1.0, sigma=2.0)
        img = img.astype(np.float32)
        cx, cy = compute_centroid(img, bbox=(24, 24, 40, 40), peak_row=32, peak_col=32)
        assert abs(cx - 32.0) < 0.1, f"x centroid error: {abs(cx - 32.0):.4f}"
        assert abs(cy - 32.0) < 0.1, f"y centroid error: {abs(cy - 32.0):.4f}"

    def test_centroid_window_clamps_to_image_boundary(self):
        """Star at image edge — must not raise, result must be inside image."""
        img = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img, 0.0, 0.0, 1.0, sigma=1.5)
        img = img.astype(np.float32)
        cx, cy = compute_centroid(img, bbox=(0, 0, 5, 5), peak_row=0, peak_col=0)
        assert 0.0 <= cx < 64.0
        assert 0.0 <= cy < 64.0

    def test_raises_on_non_2d_image(self):
        img3d = np.zeros((3, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            compute_centroid(img3d, bbox=(0, 0, 10, 10))

    def test_zero_window_returns_peak(self):
        """All-zero window falls back to peak pixel."""
        img = np.zeros((64, 64), dtype=np.float32)
        cx, cy = compute_centroid(img, bbox=(20, 20, 30, 30), peak_row=25, peak_col=25)
        assert cx == 25.0
        assert cy == 25.0

    def test_subpixel_accuracy(self):
        """Centroid of a Gaussian placed at a non-integer position."""
        img = np.zeros((64, 64), dtype=np.float64)
        true_x, true_y = 31.7, 32.3
        _render_gaussian_star(img, true_x, true_y, 1.0, sigma=2.0)
        img = img.astype(np.float32)
        cx, cy = compute_centroid(img, bbox=(24, 24, 40, 40))
        assert abs(cx - true_x) < 0.15, f"sub-pixel x error: {abs(cx - true_x):.4f}"
        assert abs(cy - true_y) < 0.15, f"sub-pixel y error: {abs(cy - true_y):.4f}"


# ===========================================================================
# detect_stars
# ===========================================================================

class TestDetectStars:

    def test_returns_list(self, star_image):
        img, _ = star_image
        result = detect_stars(img, DETECTION_CONFIG)
        assert isinstance(result, list)

    def test_blank_image_returns_empty(self, blank_image):
        result = detect_stars(blank_image, DETECTION_CONFIG)
        assert result == []

    def test_background_only_returns_empty(self):
        """Image that is entirely at background level — nothing above threshold."""
        img = np.full((128, 128), 0.02, dtype=np.float32)
        result = detect_stars(img, DETECTION_CONFIG)
        assert result == []

    def test_detects_expected_number_of_stars(self):
        """Three bright stars should each produce exactly one detection."""
        star_positions = [(30.0, 30.0, 1.0), (80.0, 50.0, 0.7), (100.0, 100.0, 0.5)]
        img, _ = _make_star_image(stars=star_positions, noise_sigma=0.0)
        result = detect_stars(img, DETECTION_CONFIG)
        assert len(result) == 3, f"Expected 3 detections, got {len(result)}"

    def test_single_star_detected(self, single_star_image):
        img, _ = single_star_image
        result = detect_stars(img, DETECTION_CONFIG)
        assert len(result) >= 1

    def test_star_candidates_are_starcandidate_instances(self, star_image):
        img, _ = star_image
        for cand in detect_stars(img, DETECTION_CONFIG):
            assert isinstance(cand, StarCandidate)

    def test_sorted_by_descending_brightness(self):
        """Candidates must be ordered brightest first."""
        img, _ = _make_star_image(
            stars=[(30.0, 30.0, 0.3), (80.0, 80.0, 1.0), (60.0, 60.0, 0.6)],
            noise_sigma=0.0,
        )
        result = detect_stars(img, DETECTION_CONFIG)
        brightnesses = [c.brightness for c in result]
        assert brightnesses == sorted(brightnesses, reverse=True)

    def test_max_stars_cap_respected(self):
        """max_stars=1 must return at most 1 candidate."""
        img, _ = _make_star_image(noise_sigma=0.0)
        cfg = {**DETECTION_CONFIG, "max_stars": 1}
        result = detect_stars(img, cfg)
        assert len(result) <= 1

    def test_detected_positions_close_to_ground_truth(self):
        """Detected centroid must be within 1.5 px of the known star position."""
        truth = [(40.0, 40.0, 1.0), (90.0, 70.0, 0.6)]
        img, _ = _make_star_image(stars=truth, noise_sigma=0.0)
        detections = detect_stars(img, DETECTION_CONFIG)

        assert len(detections) == len(truth), (
            f"Expected {len(truth)} detections, got {len(detections)}"
        )

        # Match detections to truth by nearest centroid
        truth_pts = [(x, y) for x, y, _ in truth]
        for det in detections:
            nearest_dist = min(
                math.hypot(det.x - tx, det.y - ty) for tx, ty in truth_pts
            )
            assert nearest_dist < 1.5, (
                f"Detection at ({det.x:.2f},{det.y:.2f}) "
                f"is {nearest_dist:.2f} px from nearest ground-truth star"
            )

    def test_brightness_is_positive(self, star_image):
        img, _ = star_image
        for cand in detect_stars(img, DETECTION_CONFIG):
            assert cand.brightness > 0.0

    def test_peak_is_positive(self, star_image):
        img, _ = star_image
        for cand in detect_stars(img, DETECTION_CONFIG):
            assert cand.peak > 0.0

    def test_peak_ge_min_peak_brightness(self):
        """Every detected candidate must exceed the min_peak_brightness filter."""
        img, _ = _make_star_image()
        cfg = {**DETECTION_CONFIG, "min_peak_brightness": 0.1}
        for cand in detect_stars(img, cfg):
            assert cand.peak >= 0.1

    def test_area_within_bounds(self):
        """Every blob must satisfy min_area_px ≤ area ≤ max_area_px."""
        img, _ = _make_star_image(noise_sigma=0.0)
        cfg = {**DETECTION_CONFIG, "min_area_px": 1, "max_area_px": 200}
        for cand in detect_stars(img, cfg):
            assert cfg["min_area_px"] <= cand.area <= cfg["max_area_px"]

    def test_candidates_within_image_bounds(self, star_image):
        """All detected centroids must lie inside the image dimensions."""
        img, _ = star_image
        h, w = img.shape
        for cand in detect_stars(img, DETECTION_CONFIG):
            assert 0.0 <= cand.x < w, f"x={cand.x} outside [0, {w})"
            assert 0.0 <= cand.y < h, f"y={cand.y} outside [0, {h})"

    def test_bbox_stored_correctly(self, single_star_image):
        img, _ = single_star_image
        result = detect_stars(img, DETECTION_CONFIG)
        assert len(result) >= 1
        r_min, c_min, r_max, c_max = result[0].bbox
        assert r_min < r_max and c_min < c_max

    def test_sigma_clip_threshold_works(self, star_image):
        """sigma_clip method must still detect stars."""
        img, _ = star_image
        cfg = {**DETECTION_CONFIG, "threshold_method": "sigma_clip", "sigma_clip_k": 5.0}
        result = detect_stars(img, cfg)
        assert isinstance(result, list)

    def test_unknown_threshold_method_raises(self, star_image):
        img, _ = star_image
        cfg = {**DETECTION_CONFIG, "threshold_method": "magic_threshold"}
        with pytest.raises(ValueError, match="Unknown"):
            detect_stars(img, cfg)

    def test_raises_on_non_2d_image(self):
        img3d = np.zeros((3, 128, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            detect_stars(img3d, DETECTION_CONFIG)

    def test_deterministic_for_same_input(self, star_image):
        """Same image + config must produce identical results."""
        img, _ = star_image
        r1 = detect_stars(img, DETECTION_CONFIG)
        r2 = detect_stars(img, DETECTION_CONFIG)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.x == b.x and a.y == b.y


# ===========================================================================
# detect_stars  — end-to-end with real synthetic pipeline
# ===========================================================================

def _build_pipeline_result():
    """Build one end-to-end pipeline result used by TestDetectStarsEndToEnd."""
    catalog = load_catalog(CATALOG_PATH)
    ds_cfg = {
        "image_width": 256, "image_height": 256,
        "field_of_view_deg": 20.0, "max_stars_per_image": 30,
        "psf_sigma_px": 1.5, "min_star_flux": 0.05,
        "background_level": 0.02, "read_noise_sigma": 0.005,
        "shot_noise": False, "artifact_probability": 0.0,
    }
    gen = StarFieldGenerator(catalog, ds_cfg)
    # Find a seed that yields ≥2 stars so the median-filter background
    # estimation is stable (single-star frames can be over-subtracted)
    sf = None
    for seed in range(500):
        candidate = gen.generate(seed=seed)
        if len(candidate.stars) >= 2:
            sf = candidate
            break
    if sf is None:
        # Fallback: point near a known dense region of the catalog
        sf = gen.generate(seed=0, boresight_ra_deg=83.0, boresight_dec_deg=5.0)
    preprocessed = _preprocess_array(sf.image)
    detections = detect_stars(preprocessed, DETECTION_CONFIG)
    return sf, preprocessed, detections


# Module-level cache so the expensive setup runs only once
_E2E_RESULT = None


def _get_e2e_result():
    global _E2E_RESULT
    if _E2E_RESULT is None:
        _E2E_RESULT = _build_pipeline_result()
    return _E2E_RESULT


class TestDetectStarsEndToEnd:
    """Generate images via StarFieldGenerator, run full preprocess + detect."""

    def test_preprocessed_is_float32(self):
        _, preprocessed, _ = _get_e2e_result()
        assert preprocessed.dtype == np.float32

    def test_preprocessed_in_unit_range(self):
        _, preprocessed, _ = _get_e2e_result()
        assert preprocessed.min() >= 0.0
        assert preprocessed.max() <= 1.0 + 1e-6

    def test_detections_are_star_candidates(self):
        _, _, detections = _get_e2e_result()
        for d in detections:
            assert isinstance(d, StarCandidate)

    def test_ground_truth_stars_detected(self):
        """Every ground-truth rendered star should have a detection within 3 px."""
        sf, _, detections = _get_e2e_result()
        if not sf.stars or not detections:
            pytest.skip("No stars in this frame or no detections — skip match test")
        for gt in sf.stars:
            nearest = min(
                math.hypot(d.x - gt.x_px, d.y - gt.y_px) for d in detections
            )
            assert nearest < 3.0, (
                f"Ground-truth star {gt.star_id} at ({gt.x_px:.1f},{gt.y_px:.1f}) "
                f"not matched within 3 px (nearest: {nearest:.2f} px)"
            )


def _preprocess_array(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing steps directly to a numpy array (no disk I/O)."""
    from src.preprocessing.image_preprocessing import (
        subtract_background, reduce_noise, normalise
    )
    img = subtract_background(image, method="median_filter", filter_size=31)
    img = reduce_noise(img, method="gaussian", sigma=0.8)
    img = normalise(img, method="min_max")
    return img


# ===========================================================================
# extract_features  (Phase 3 — implemented)
# ===========================================================================

# Feature config matching config.yaml values
FEATURE_CONFIG = {
    "features": {
        "max_stars":    10,
        "descriptor":   "pairwise_distances_and_ratios",
        "image_width":  128,
        "image_height": 128,
    }
}

_MAX_N      = 10
_N_PAIRS    = _MAX_N * (_MAX_N - 1) // 2   # 45
_FEAT_DIM   = 2 * _N_PAIRS                  # 90


class TestExtractFeatures:

    def test_returns_ndarray(self, star_image):
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        feat = extract_features(stars, FEATURE_CONFIG)
        assert isinstance(feat, np.ndarray)

    def test_output_dtype_float32(self, star_image):
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        feat = extract_features(stars, FEATURE_CONFIG)
        assert feat.dtype == np.float32

    def test_output_length_distances_and_ratios(self, star_image):
        """With max_stars=10 and distances+ratios: 2*45 = 90 features."""
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        feat = extract_features(stars, FEATURE_CONFIG)
        assert feat.shape == (_FEAT_DIM,), (
            f"Expected ({_FEAT_DIM},), got {feat.shape}"
        )

    def test_output_length_distances_only(self, star_image):
        """With descriptor=pairwise_distances: 45 features."""
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        cfg = {
            "features": {
                **FEATURE_CONFIG["features"],
                "descriptor": "pairwise_distances",
            }
        }
        feat = extract_features(stars, cfg)
        assert feat.shape == (_N_PAIRS,)

    def test_empty_star_list_returns_zero_vector(self):
        """No detections → zero feature vector of correct length."""
        feat = extract_features([], FEATURE_CONFIG)
        assert feat.shape == (_FEAT_DIM,)
        assert np.all(feat == 0.0)

    def test_single_star_returns_zero_vector(self, single_star_image):
        """Only 1 star → can't form pairs → zero vector."""
        img, _ = single_star_image
        stars = detect_stars(img, {**DETECTION_CONFIG, "max_stars": 1})
        stars = stars[:1]   # force single star
        feat = extract_features(stars, FEATURE_CONFIG)
        assert feat.shape == (_FEAT_DIM,)
        assert np.all(feat == 0.0)

    def test_distances_in_zero_one_range(self, star_image):
        """Normalised distances must be in [0, 1]."""
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        feat = extract_features(stars, FEATURE_CONFIG)
        distances = feat[:_N_PAIRS]
        assert distances.min() >= 0.0
        assert distances.max() <= 1.0 + 1e-6

    def test_ratios_in_zero_one_range(self, star_image):
        """Brightness ratios must be in (0, 1)."""
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        feat = extract_features(stars, FEATURE_CONFIG)
        ratios = feat[_N_PAIRS:]
        for r in ratios[ratios > 0]:
            assert 0.0 < r <= 1.0, f"Ratio {r} out of (0, 1]"

    def test_deterministic_same_stars(self, star_image):
        """Same star list → identical feature vector."""
        img, _ = star_image
        stars = detect_stars(img, DETECTION_CONFIG)
        f1 = extract_features(stars, FEATURE_CONFIG)
        f2 = extract_features(stars, FEATURE_CONFIG)
        assert np.array_equal(f1, f2)

    def test_different_images_different_features(self):
        """Two clearly different star patterns must produce different features."""
        img1, _ = _make_star_image(
            stars=[(20.0, 20.0, 1.0), (40.0, 40.0, 0.5)], noise_sigma=0.0
        )
        img2, _ = _make_star_image(
            stars=[(100.0, 100.0, 1.0), (110.0, 110.0, 0.5)], noise_sigma=0.0
        )
        s1 = detect_stars(img1, DETECTION_CONFIG)
        s2 = detect_stars(img2, DETECTION_CONFIG)
        f1 = extract_features(s1, FEATURE_CONFIG)
        f2 = extract_features(s2, FEATURE_CONFIG)
        assert not np.array_equal(f1, f2)

    def test_max_stars_cap_applied(self):
        """Only top max_stars=2 are used regardless of how many are detected."""
        img, _ = _make_star_image(
            stars=[(20.0, 20.0, 1.0), (50.0, 50.0, 0.8), (90.0, 90.0, 0.6)],
            noise_sigma=0.0,
        )
        stars_all = detect_stars(img, DETECTION_CONFIG)
        cfg_2 = {
            "features": {
                "max_stars": 2,
                "descriptor": "pairwise_distances_and_ratios",
                "image_width": 128, "image_height": 128,
            }
        }
        feat = extract_features(stars_all, cfg_2)
        # With max_stars=2: 1 pair → 2 features (1 dist + 1 ratio)
        assert feat.shape == (2,)

    def test_feature_uses_full_config_for_dim(self):
        """feature_dim is always max_stars*(max_stars-1)/2 * n_groups."""
        for max_n in [3, 5, 8]:
            n_p = max_n * (max_n - 1) // 2
            cfg = {
                "features": {
                    "max_stars": max_n,
                    "descriptor": "pairwise_distances_and_ratios",
                    "image_width": 128, "image_height": 128,
                }
            }
            feat = extract_features([], cfg)
            assert feat.shape == (2 * n_p,), (
                f"max_stars={max_n}: expected ({2*n_p},), got {feat.shape}"
            )
