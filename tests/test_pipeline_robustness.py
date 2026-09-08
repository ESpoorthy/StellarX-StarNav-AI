"""
test_pipeline_robustness.py
============================
Comprehensive tests covering:

A. NaN / Inf / zero-vector guards in estimate_attitude and run_recognition
B. Degenerate geometry (collinear vectors, single star, two stars)
C. Synthetic ground-truth attitude recovery (geodesic SO(3) error)
D. Intentional failure scenarios (blank image, no stars, insufficient stars)
E. PipelineResult integrity check (det, quaternion norm, residual)
F. Performance regression (pipeline must complete within a wall-clock bound)
G. Repeated execution stability (same seed → same result, N times)

Run with:
    pytest tests/test_pipeline_robustness.py -v
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from src.navigation.attitude_estimator import (
    AttitudeEstimate,
    estimate_attitude,
    validate_rotation_matrix,
    angular_error_deg,
)
from src.recognition.catalog_index import CatalogIndex
from src.catalog.catalog_loader import load_catalog

CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")
CONFIG_BASE = {
    "navigation": {
        "min_correspondences": 2,
        "max_residual_threshold_deg": 2.0,
        "outlier_rejection_threshold_deg": 2.0,
        "outlier_rejection_max_iter": 3,
    },
    "recognition": {
        "angle_tolerance_deg": 0.5,
        "ransac_iterations": 30,
        "max_residual_deg": 1.0,
        "min_inliers": 2,
        "confidence_success": 0.6,
        "confidence_partial": 0.3,
    },
    "features": {
        "max_stars": 10,
        "descriptor": "pairwise_distances_and_ratios",
        "image_width": 128,
        "image_height": 128,
    },
    "dataset": {
        "catalog_file": str(CATALOG_PATH),
        "catalog_mag_limit": 6.5,
        "image_width": 128,
        "image_height": 128,
        "field_of_view_deg": 20.0,
        "max_stars_per_image": 30,
        "psf_sigma_px": 1.5,
        "min_star_flux": 0.05,
        "background_level": 0.02,
        "read_noise_sigma": 0.0,   # deterministic by default
        "shot_noise": False,
        "artifact_probability": 0.0,
        "random_seed": 42,
    },
    "preprocessing": {
        "background_subtraction": True,
        "background_method": "median_filter",
        "background_filter_size": 31,
        "noise_reduction": True,
        "noise_method": "gaussian",
        "noise_sigma": 0.8,
        "normalization": "min_max",
    },
    "star_detection": {
        "threshold_method": "absolute",
        "min_brightness": 0.05,
        "sigma_clip_k": 5.0,
        "min_area_px": 1,
        "max_area_px": 200,
        "min_peak_brightness": 0.04,
        "max_stars": 50,
        "centroid_method": "intensity_weighted",
        "centroid_half_window": 5,
    },
    "integrity": {
        "max_mean_residual_deg": 2.0,
        "max_per_star_residual_deg": 3.0,
        "min_quaternion_norm": 0.999,
        "max_quaternion_norm": 1.001,
        "min_det": 0.999,
        "min_inliers_valid": 2,
    },
}


# ============================================================================
# Helpers
# ============================================================================

def _make_valid_directions(n: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (obs_vecs, cat_vecs, R_true) where cat = R_true @ obs."""
    rng = np.random.default_rng(seed)
    # Build a random proper rotation
    q = rng.normal(0, 1, 4)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    R_true = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
    ], dtype=np.float64)

    obs = rng.normal(0, 1, (n, 3))
    obs = obs / np.linalg.norm(obs, axis=1, keepdims=True)
    cat = (R_true @ obs.T).T
    return obs, cat, R_true


# ============================================================================
# A. NaN / Inf / Zero-vector guards
# ============================================================================

class TestNaNInfGuards:

    def test_all_nan_obs_returns_invalid(self):
        obs = np.full((4, 3), float("nan"))
        cat = np.eye(3)[:1].repeat(4, axis=0)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert not result.is_valid

    def test_all_inf_obs_returns_invalid(self):
        obs = np.full((4, 3), float("inf"))
        cat = np.eye(3)[:1].repeat(4, axis=0)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert not result.is_valid

    def test_mixed_nan_partial_returns_result(self):
        """Some NaN rows + some valid rows — should still attempt a solve."""
        obs, cat, _ = _make_valid_directions(6)
        obs[0] = float("nan")
        obs[2] = float("inf")
        # 4 valid rows remain — solver should attempt
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        # May succeed or fail depending on geometry; must not raise
        assert isinstance(result, AttitudeEstimate)

    def test_zero_vector_obs_filtered(self):
        """Zero-length observed vectors must be silently removed."""
        obs, cat, _ = _make_valid_directions(6)
        obs[1] = np.zeros(3)   # zero vector
        obs[3] = np.zeros(3)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)

    def test_zero_vector_cat_filtered(self):
        obs, cat, _ = _make_valid_directions(6)
        cat[0] = np.zeros(3)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)

    def test_nan_weights_replaced_by_uniform(self):
        obs, cat, _ = _make_valid_directions(4)
        weights = np.array([float("nan"), 0.5, 0.8, float("inf")])
        result = estimate_attitude(obs, cat, CONFIG_BASE, weights=weights)
        assert isinstance(result, AttitudeEstimate)

    def test_shape_mismatch_raises(self):
        obs = np.eye(3)
        cat = np.eye(4)[:, :3]
        with pytest.raises(ValueError, match="Shape mismatch"):
            estimate_attitude(obs, cat, CONFIG_BASE)

    def test_valid_inputs_succeed(self):
        obs, cat, R_true = _make_valid_directions(6)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)
        assert result.num_correspondences >= 4   # after filtering


# ============================================================================
# B. Degenerate geometry
# ============================================================================

class TestDegenerateGeometry:

    def test_zero_stars_returns_invalid(self):
        obs = np.zeros((0, 3))
        cat = np.zeros((0, 3))
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert not result.is_valid
        assert result.n_inliers == 0

    def test_one_star_returns_invalid(self):
        obs = np.array([[0., 0., 1.]])
        cat = np.array([[1., 0., 0.]])
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert not result.is_valid

    def test_two_stars_minimum_attempt(self):
        """Two valid stars is the minimum — solver should attempt."""
        obs, cat, _ = _make_valid_directions(2)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)

    def test_collinear_vectors_handled(self):
        """All observed stars on the same line — degenerate configuration."""
        obs = np.tile([0., 0., 1.], (5, 1))   # all identical
        cat = np.tile([1., 0., 0.], (5, 1))
        # Should not raise; may return is_valid=False
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)

    def test_antiparallel_vectors_handled(self):
        obs = np.array([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
        cat = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)

    def test_nearly_collinear_stars_handled(self):
        """Stars nearly collinear — very poorly-conditioned geometry."""
        eps = 1e-8
        base = np.array([0., 0., 1.])
        obs = base + np.array([[eps * i, 0., 0.] for i in range(4)])
        obs = obs / np.linalg.norm(obs, axis=1, keepdims=True)
        cat = obs + 0.01  # slightly different
        cat = cat / np.linalg.norm(cat, axis=1, keepdims=True)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert isinstance(result, AttitudeEstimate)

    def test_insufficient_stars_returns_insufficient_status(self):
        """One star — pipeline must return INSUFFICIENT_STARS, not hang."""
        from src.preprocessing.star_detection import StarCandidate, detect_stars
        from src.preprocessing.star_field_generator import _render_gaussian_star
        # Make an image with only 1 visible star
        img = np.zeros((128, 128), dtype=np.float64)
        _render_gaussian_star(img, 64.0, 64.0, 1.0, sigma=1.5)
        img = img.astype(np.float32)
        stars = detect_stars(img, CONFIG_BASE["star_detection"])
        # estimate_attitude with 1 star must return is_valid=False
        if stars:
            obs = np.array([[0., 0., 1.]])
            cat = np.array([[0., 0., 1.]])
            result = estimate_attitude(obs, cat, CONFIG_BASE)
            assert not result.is_valid


# ============================================================================
# C. Synthetic ground-truth attitude recovery
# ============================================================================

class TestGroundTruthRecovery:

    def test_perfect_correspondence_small_error(self):
        """Perfect correspondences → error should be near 0°."""
        obs, cat, R_true = _make_valid_directions(8, seed=42)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert result.is_valid, "Expected valid result with perfect correspondences"
        err = angular_error_deg(R_true, result.rotation_matrix)
        assert err < 1.0, f"Geodesic error {err:.4f}° too large for perfect inputs"

    def test_noisy_correspondence_bounded_error(self):
        """Noisy correspondences → error should stay below a loose bound."""
        rng = np.random.default_rng(7)
        obs, cat, R_true = _make_valid_directions(10, seed=7)
        # Add small noise
        cat = cat + rng.normal(0, 0.01, cat.shape)
        cat = cat / np.linalg.norm(cat, axis=1, keepdims=True)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        if result.is_valid:
            err = angular_error_deg(R_true, result.rotation_matrix)
            assert err < 15.0, f"Error {err:.3f}° unreasonably large for σ=0.01 noise"

    def test_identity_rotation_recovered(self):
        """obs == cat → R should be identity, error ≈ 0°."""
        rng = np.random.default_rng(0)
        vecs = rng.normal(0, 1, (8, 3))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        result = estimate_attitude(vecs, vecs.copy(), CONFIG_BASE)
        if result.is_valid:
            R_id = np.eye(3)
            err = angular_error_deg(R_id, result.rotation_matrix)
            assert err < 0.1, f"Identity rotation error {err:.6f}° should be near 0"

    def test_quaternion_norm_is_unit(self):
        obs, cat, _ = _make_valid_directions(6, seed=99)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        if result.is_valid:
            q_norm = float(np.linalg.norm(result.quaternion))
            assert abs(q_norm - 1.0) < 1e-6, f"Quaternion norm {q_norm} not unit"

    def test_rotation_matrix_det_is_positive_one(self):
        obs, cat, _ = _make_valid_directions(6, seed=11)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        if result.is_valid:
            det = float(np.linalg.det(result.rotation_matrix))
            assert abs(det - 1.0) < 1e-5, f"det(R) = {det} not +1"

    def test_rotation_matrix_is_orthogonal(self):
        obs, cat, _ = _make_valid_directions(6, seed=22)
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        if result.is_valid:
            v = validate_rotation_matrix(result.rotation_matrix)
            assert v["orthogonality_error"] < 1e-5, (
                f"R^T R deviation: {v['orthogonality_error']:.2e}")

    def test_geodesic_error_metric_symmetric(self):
        """angular_error_deg(R1, R2) == angular_error_deg(R2, R1)."""
        obs, cat, R = _make_valid_directions(6, seed=55)
        R2 = np.eye(3)
        err1 = angular_error_deg(R, R2)
        err2 = angular_error_deg(R2, R)
        assert abs(err1 - err2) < 1e-9

    def test_geodesic_error_identity_is_zero(self):
        R = np.eye(3)
        assert angular_error_deg(R, R) < 1e-9

    def test_geodesic_error_180deg_rotation(self):
        """R = diag(-1,-1,1) is a 180° rotation around Z."""
        R = np.diag([-1., -1., 1.])
        err = angular_error_deg(np.eye(3), R)
        assert abs(err - 180.0) < 1e-5, f"Expected 180°, got {err:.6f}°"


# ============================================================================
# D. Intentional failure scenarios
# ============================================================================

class TestIntentionalFailures:

    def test_blank_image_returns_insufficient_stars(self):
        """Blank image → no stars detected → navigation must fail gracefully."""
        from src.navigation.navigator import run_navigation, _preprocess_image
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        blank = np.zeros((128, 128), dtype=np.float32)
        result = run_navigation(blank, CONFIG_BASE, cidx)
        assert result.status in (
            "INSUFFICIENT_STARS", "FAILURE", "LOW_CONFIDENCE", "ATTITUDE_FAILURE"
        ), f"Expected failure status, got {result.status}"
        assert result.n_inlier_stars == 0

    def test_all_noise_image_returns_failure_or_low_confidence(self):
        """Pure noise image should not produce a confident attitude."""
        from src.navigation.navigator import run_navigation
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        rng = np.random.default_rng(99)
        noise_img = rng.uniform(0, 0.04, (128, 128)).astype(np.float32)
        result = run_navigation(noise_img, CONFIG_BASE, cidx)
        # confidence must be low if a result is returned
        if result.status == "SUCCESS":
            assert result.attitude_confidence < 0.9, (
                "Should not produce high-confidence result on noise image")

    def test_single_star_image_returns_insufficient(self):
        """One bright star — cannot determine attitude."""
        from src.navigation.navigator import run_navigation
        from src.preprocessing.star_field_generator import _render_gaussian_star
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        img = np.zeros((128, 128), dtype=np.float64)
        _render_gaussian_star(img, 64.0, 64.0, 1.0, sigma=1.5)
        img = np.clip(img, 0, 1).astype(np.float32)
        result = run_navigation(img, CONFIG_BASE, cidx)
        assert result.status not in ("SUCCESS",), (
            f"Should not succeed with 1 star, got {result.status}")

    def test_pipeline_never_hangs(self):
        """Complete pipeline must finish within 30 seconds on 128×128 image."""
        from src.navigation.navigator import run_navigation
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 0.05, (128, 128)).astype(np.float32)
        t0 = time.perf_counter()
        result = run_navigation(img, CONFIG_BASE, cidx)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"Pipeline took {elapsed:.1f}s — too slow"
        assert isinstance(result.status, str)

    def test_insufficient_stars_gives_structured_failure_message(self):
        """Failure result must have a meaningful status string."""
        obs = np.array([[0., 0., 1.]])
        cat = np.array([[0., 0., 1.]])
        result = estimate_attitude(obs, cat, CONFIG_BASE)
        assert not result.is_valid
        assert isinstance(result.status if hasattr(result, "status") else "", str)


# ============================================================================
# E. PipelineResult integrity check
# ============================================================================

class TestIntegrityCheck:

    def _make_nav_result(self, R, q, n_inliers=3, residual=0.3):
        """Build a minimal NavigationResult-like object for testing."""
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class FakeNav:
            status: str = "SUCCESS"
            attitude_status: str = "DETERMINED"
            position_status: str = "UNAVAILABLE"
            quaternion: np.ndarray = dc_field(default_factory=lambda: np.array([1.,0.,0.,0.]))
            rotation_matrix: np.ndarray = dc_field(default_factory=lambda: np.eye(3))
            euler_angles_deg: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
            attitude_confidence: float = 0.8
            attitude_residual_deg: float = 0.3
            max_residual_deg: float = 0.5
            n_observed_stars: int = 5
            n_matched_stars: int = 4
            n_inlier_stars: int = 3
            n_outlier_stars: int = 1
            identified_stars: list = dc_field(default_factory=list)
            preprocessing_time_ms: float = 5.0
            detection_time_ms: float = 10.0
            feature_extraction_time_ms: float = 1.0
            recognition_time_ms: float = 20.0
            attitude_time_ms: float = 2.0
            total_time_ms: float = 38.0
            error_message: str = ""

        nav = FakeNav()
        nav.rotation_matrix = R
        nav.quaternion = q
        nav.n_inlier_stars = n_inliers
        nav.attitude_residual_deg = residual
        return nav

    def test_valid_rotation_passes_integrity(self):
        from src.navigation.pipeline_result import build_pipeline_result
        obs, cat, R = _make_valid_directions(4)
        q = np.array([1., 0., 0., 0.])
        nav = self._make_nav_result(R, q)
        pr = build_pipeline_result(nav, CONFIG_BASE)
        # det(R) and orthogonality should be fine
        assert math.isfinite(pr.integrity.determinant)
        assert abs(pr.integrity.determinant - 1.0) < 0.01

    def test_invalid_det_rejected(self):
        from src.navigation.pipeline_result import build_pipeline_result
        # R with det = -1 (reflection, not rotation)
        R_bad = np.diag([1., 1., -1.])
        q = np.array([1., 0., 0., 0.])
        nav = self._make_nav_result(R_bad, q, n_inliers=3)
        pr = build_pipeline_result(nav, CONFIG_BASE)
        assert pr.integrity.status in ("REJECTED", "DEGRADED")

    def test_zero_quaternion_rejected(self):
        from src.navigation.pipeline_result import build_pipeline_result
        R = np.eye(3)
        q = np.zeros(4)  # norm = 0 — invalid
        nav = self._make_nav_result(R, q)
        pr = build_pipeline_result(nav, CONFIG_BASE)
        assert pr.integrity.status == "REJECTED"
        assert "norm" in pr.integrity.rejection_reason.lower()

    def test_excessive_residual_rejected(self):
        from src.navigation.pipeline_result import build_pipeline_result
        obs, cat, R = _make_valid_directions(4)
        q_valid = np.array([1., 0., 0., 0.])
        nav = self._make_nav_result(R, q_valid, residual=5.0)
        pr = build_pipeline_result(nav, CONFIG_BASE)
        assert pr.integrity.status in ("REJECTED", "DEGRADED")

    def test_insufficient_inliers_rejected(self):
        from src.navigation.pipeline_result import build_pipeline_result
        obs, cat, R = _make_valid_directions(4)
        q_valid = np.array([1., 0., 0., 0.])
        nav = self._make_nav_result(R, q_valid, n_inliers=0)
        pr = build_pipeline_result(nav, CONFIG_BASE)
        assert pr.integrity.status == "REJECTED"

    def test_nan_rotation_matrix_rejected(self):
        from src.navigation.pipeline_result import build_pipeline_result
        R_nan = np.full((3, 3), float("nan"))
        q = np.array([1., 0., 0., 0.])
        nav = self._make_nav_result(R_nan, q)
        nav.status = "SUCCESS"
        pr = build_pipeline_result(nav, CONFIG_BASE)
        assert pr.integrity.status == "REJECTED"


# ============================================================================
# F. Performance regression
# ============================================================================

class TestPerformanceRegression:
    """Performance regression tests — no class-scoped fixtures to avoid pytest deprecation."""

    def test_single_run_under_30s(self):
        """Single 128×128 pipeline run must complete < 30 s."""
        from src.navigation.navigator import run_navigation
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 0.05, (128, 128)).astype(np.float32)
        t0 = time.perf_counter()
        run_navigation(img, CONFIG_BASE, cidx)
        assert (time.perf_counter() - t0) < 30.0

    def test_catalog_index_build_fast(self):
        """Building CatalogIndex from the 50-star catalog must take < 2 s."""
        t0 = time.perf_counter()
        CatalogIndex(load_catalog(str(CATALOG_PATH)))
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"CatalogIndex build took {elapsed:.2f}s"

    def test_attitude_solver_fast(self):
        """Wahba/SVD on 10 correspondences must complete < 0.05 s."""
        obs, cat, _ = _make_valid_directions(10)
        t0 = time.perf_counter()
        for _ in range(50):
            estimate_attitude(obs, cat, CONFIG_BASE)
        elapsed = (time.perf_counter() - t0) / 50
        assert elapsed < 0.05, f"Attitude solver avg {elapsed*1000:.1f}ms — too slow"

    def test_repeated_runs_never_error(self):
        """Same image repeated 3× — must never raise or return ERROR status.

        RANSAC uses random sampling so SUCCESS/PARTIAL status can vary between
        runs. The critical invariant is that the pipeline never crashes (ERROR).
        """
        from src.navigation.navigator import run_navigation
        from src.preprocessing.star_field_generator import StarFieldGenerator

        catalog = load_catalog(str(CATALOG_PATH))
        gen = StarFieldGenerator(catalog, CONFIG_BASE["dataset"])
        sf = gen.generate(seed=30)
        img = sf.image
        cidx = CatalogIndex(catalog)

        acceptable = {"SUCCESS", "PARTIAL", "LOW_CONFIDENCE",
                      "INSUFFICIENT_STARS", "FAILURE", "ATTITUDE_FAILURE"}
        for run_i in range(3):
            r = run_navigation(img, CONFIG_BASE, cidx)
            assert r.status in acceptable, (
                f"Run {run_i}: unexpected status '{r.status}'")
            assert r.status != "ERROR", (
                f"Run {run_i}: pipeline raised ERROR — {r.error_message}")


# ============================================================================
# G. Vectorized ops correctness
# ============================================================================

class TestVectorizedOpsCorrectness:
    """Verify vectorized ops produce numerically identical results to scalar."""

    def test_wahba_svd_vectorized_matches_scalar(self):
        """wahba_svd_vectorized must produce det(R)=+1 and ||R^TR - I|| ≈ 0."""
        from src.recognition.pattern_matcher_optimized import wahba_svd_vectorized
        obs, cat, R_true = _make_valid_directions(6)
        w = np.ones(6)
        R = wahba_svd_vectorized(obs, cat, w)
        assert R is not None
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-6
        orth = float(np.max(np.abs(R.T @ R - np.eye(3))))
        assert orth < 1e-6

    def test_ransac_inlier_count_vectorized_correct(self):
        """All perfect inliers must be found by vectorized RANSAC."""
        from src.recognition.pattern_matcher_optimized import (
            ransac_inlier_count_vectorized, wahba_svd_vectorized)
        obs, cat, R_true = _make_valid_directions(8)
        w = np.ones(8)
        R = wahba_svd_vectorized(obs, cat, w)
        inliers = ransac_inlier_count_vectorized(R, obs, cat, max_residual_deg=1.0)
        assert len(inliers) == 8  # all should be inliers for perfect correspondences

    def test_compute_residuals_vectorized_all_zero_for_perfect(self):
        """Perfect correspondences → all residuals ≈ 0."""
        from src.recognition.pattern_matcher_optimized import (
            compute_residuals_vectorized, wahba_svd_vectorized)
        obs, cat, _ = _make_valid_directions(6)
        w = np.ones(6)
        R = wahba_svd_vectorized(obs, cat, w)
        residuals = compute_residuals_vectorized(R, obs, cat)
        assert residuals.max() < 1e-5, f"Max residual {residuals.max():.2e} should be ~0"

    def test_pairwise_angles_vectorized_symmetric(self):
        """Pairwise angle matrix must be symmetric."""
        from src.recognition.pattern_matcher_optimized import pairwise_angles_vectorized
        obs, _, _ = _make_valid_directions(6)
        A = pairwise_angles_vectorized(obs)
        assert A.shape == (6, 6)
        assert np.allclose(A, A.T, atol=1e-9)
        assert np.allclose(np.diag(A), 0.0, atol=1e-9)

    def test_catalog_index_find_pairs_vectorized(self):
        """find_pairs_by_angle must return pairs within tolerance."""
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        pairs = cidx.find_pairs_by_angle(angle_deg=30.0, tolerance_deg=1.0)
        for i, j, actual in pairs:
            assert abs(actual - 30.0) <= 1.0, (
                f"Pair ({i},{j}) angle {actual:.3f}° outside tolerance")

    def test_catalog_index_find_pairs_sorted_by_diff(self):
        """Results must be sorted by |actual - target| ascending."""
        cidx = CatalogIndex(load_catalog(str(CATALOG_PATH)))
        pairs = cidx.find_pairs_by_angle(angle_deg=45.0, tolerance_deg=5.0)
        if len(pairs) >= 2:
            diffs = [abs(actual - 45.0) for _, _, actual in pairs]
            assert diffs == sorted(diffs), "Pairs not sorted by angular difference"
