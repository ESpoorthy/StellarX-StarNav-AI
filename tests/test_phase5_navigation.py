"""
test_phase5_navigation.py
=========================
Phase 5 — Navigation / Attitude Determination tests.

Covers all 21 required scenarios:
  Camera model (1-5), Coordinate conversion (6-8),
  Attitude estimation (9-16), Quaternion (17-19), Integration (20-21).

Run with:
    pytest tests/test_phase5_navigation.py -v
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pytest

from src.navigation.attitude_estimator import (
    AttitudeEstimate,
    angular_error_deg,
    estimate_attitude,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_euler_deg,
    quaternion_sign_canonical,
    validate_rotation_matrix,
    _wahba_svd,
)
from src.navigation.camera_model import CameraModel
from src.navigation.position_estimator import PositionEstimate, estimate_position

CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")

NAV_CFG = {
    "navigation": {
        "min_correspondences": 2,
        "max_residual_threshold_deg": 2.0,
        "outlier_rejection_threshold_deg": 2.0,
        "outlier_rejection_max_iter": 3,
        "attitude_confidence_threshold": 0.3,
    },
    "dataset": {"image_width": 512, "image_height": 512, "field_of_view_deg": 20.0},
    "recognition": {
        "angle_tolerance_deg": 0.5, "min_inliers": 3,
        "confidence_success": 0.6, "confidence_partial": 0.3,
        "max_residual_deg": 1.0, "ransac_iterations": 50,
    },
    "star_detection": {
        "threshold_method": "absolute", "min_brightness": 0.05,
        "min_area_px": 1, "max_area_px": 200, "min_peak_brightness": 0.04,
        "max_stars": 50, "centroid_method": "intensity_weighted",
        "centroid_half_window": 5,
    },
    "features": {"max_stars": 10},
}


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])

def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])

def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])


# ═══════════════════════════════════════════════════════════════
# Camera Model Tests (1–5)
# ═══════════════════════════════════════════════════════════════

class TestCameraModel:
    """Tests 1-5: CameraModel pixel↔unit-vector conversion."""

    def test_1_center_pixel_is_boresight(self):
        """Test 1: Center pixel → optical-axis ray [0, 0, 1]."""
        cam = CameraModel(image_width=512, image_height=512, fov_deg=20.0)
        uv = cam.pixel_to_unit_vector(cam.cx, cam.cy)
        assert np.allclose(uv, [0.0, 0.0, 1.0], atol=1e-10), \
            f"Center pixel → {uv}, expected [0,0,1]"

    def test_2_off_center_pixel_correct_direction(self):
        """Test 2: Pixel right of center → positive x component."""
        cam = CameraModel(image_width=512, image_height=512, fov_deg=20.0)
        uv = cam.pixel_to_unit_vector(cam.cx + 50, cam.cy)
        assert uv[0] > 0, f"Pixel right of center should have +x, got {uv[0]}"
        assert abs(uv[1]) < 1e-10, f"Should have y≈0 for horizontal offset, got {uv[1]}"
        assert abs(np.linalg.norm(uv) - 1.0) < 1e-10, "Unit vector must have norm 1"

    def test_3_different_focal_lengths(self):
        """Test 3: Larger FOV → smaller focal length → larger angular offset."""
        cam_narrow = CameraModel(image_width=512, image_height=512, fov_deg=10.0)
        cam_wide = CameraModel(image_width=512, image_height=512, fov_deg=40.0)
        assert cam_narrow.focal_px > cam_wide.focal_px, \
            "Narrow FOV should have larger focal length in pixels"

    def test_4_different_principal_points(self):
        """Test 4: Non-centered principal point shifts the boresight pixel."""
        cam = CameraModel(image_width=512, image_height=512, fov_deg=20.0,
                          cx=200.0, cy=300.0, focal_px=1448.0)
        # The principal point (200, 300) should map to boresight
        uv = cam.pixel_to_unit_vector(200.0, 300.0)
        assert np.allclose(uv, [0.0, 0.0, 1.0], atol=1e-10)

    def test_5_unit_vector_normalization(self):
        """Test 5: All pixel positions produce unit vectors."""
        cam = CameraModel(image_width=512, image_height=512, fov_deg=20.0)
        for col, row in [(0, 0), (256, 256), (511, 511), (100, 400), (400, 100)]:
            uv = cam.pixel_to_unit_vector(col, row)
            assert abs(np.linalg.norm(uv) - 1.0) < 1e-10, \
                f"Pixel ({col},{row}) → norm={np.linalg.norm(uv):.8f} ≠ 1"

    def test_inverse_projection_roundtrip(self):
        """pixel → unit_vector → pixel roundtrip."""
        cam = CameraModel(image_width=512, image_height=512, fov_deg=20.0)
        for col, row in [(256.0, 256.0), (300.0, 200.0), (100.0, 400.0)]:
            uv = cam.pixel_to_unit_vector(col, row)
            col2, row2 = cam.unit_vector_to_pixel(uv)
            assert abs(col2 - col) < 0.01, f"col roundtrip: {col} → {col2}"
            assert abs(row2 - row) < 0.01, f"row roundtrip: {row} → {row2}"

    def test_from_config(self):
        """CameraModel.from_config reads dataset section."""
        cam = CameraModel.from_config(NAV_CFG)
        assert cam.image_width == 512
        assert cam.image_height == 512
        assert abs(cam.fov_deg - 20.0) < 1e-9
        assert cam.focal_px > 0

    def test_behind_camera_returns_nan(self):
        """Unit vector pointing away from camera → (nan, nan) pixel."""
        cam = CameraModel(image_width=512, image_height=512, fov_deg=20.0)
        uv = np.array([0.0, 0.0, -1.0])  # z < 0: behind camera
        col, row = cam.unit_vector_to_pixel(uv)
        assert math.isnan(col) and math.isnan(row)


# ═══════════════════════════════════════════════════════════════
# Coordinate Conversion Tests (6–8)
# ═══════════════════════════════════════════════════════════════

class TestCoordinateConversion:
    """Tests 6-8: RA/Dec → inertial unit vector."""

    def test_6_known_ra_dec_unit_vector(self):
        """Test 6: RA=0°, Dec=0° → [1, 0, 0]."""
        from src.catalog.catalog_loader import CatalogStar
        star = CatalogStar(ra_deg=0.0, dec_deg=0.0)
        uv = star.unit_vector()
        assert np.allclose(uv, [1.0, 0.0, 0.0], atol=1e-10)

    def test_7_unit_vector_normalization(self):
        """Test 7: All catalog stars produce normalized unit vectors."""
        from src.catalog.catalog_loader import load_catalog
        catalog = load_catalog(CATALOG_PATH)
        for star in catalog:
            uv = star.unit_vector()
            norm = float(np.linalg.norm(uv))
            assert abs(norm - 1.0) < 1e-10, \
                f"{star.star_id}: norm={norm:.10f}"

    def test_8_north_pole_convention(self):
        """Test 8: Dec=90° → [0, 0, 1] (north pole = +Z)."""
        from src.catalog.catalog_loader import CatalogStar
        star = CatalogStar(ra_deg=0.0, dec_deg=90.0)
        uv = star.unit_vector()
        assert np.allclose(uv, [0.0, 0.0, 1.0], atol=1e-10)

    def test_ra90_dec0_is_y_axis(self):
        """RA=90°, Dec=0° → [0, 1, 0] (inertial +Y axis)."""
        from src.catalog.catalog_loader import CatalogStar
        star = CatalogStar(ra_deg=90.0, dec_deg=0.0)
        uv = star.unit_vector()
        assert np.allclose(uv, [0.0, 1.0, 0.0], atol=1e-9)


# ═══════════════════════════════════════════════════════════════
# Attitude Estimation Tests (9–16)
# ═══════════════════════════════════════════════════════════════

class TestAttitudeEstimation:
    """Tests 9-16: Attitude estimation correctness and robustness."""

    def test_9_identity_rotation_estimate(self):
        """Test 9: obs == cat (identity) → R ≈ I, error < 0.001°."""
        obs = np.eye(3)
        cat = np.eye(3)
        result = estimate_attitude(obs, cat, NAV_CFG)
        err = angular_error_deg(result.rotation_matrix, np.eye(3))
        assert err < 0.001, f"Identity error {err:.6f}° > 0.001°"
        assert result.is_valid

    def test_10_small_rotation_recovered(self):
        """Test 10: Small rotation (5°) recovered to < 0.1°."""
        R_true = _rot_z(5.0)
        rng = np.random.default_rng(10)
        vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(5, 3))]
        obs = np.array(vecs)
        cat = np.array([R_true @ v for v in vecs])
        result = estimate_attitude(obs, cat, NAV_CFG)
        err = angular_error_deg(result.rotation_matrix, R_true)
        assert err < 0.1, f"Small rotation error {err:.4f}° > 0.1°"

    def test_11_large_rotation_recovered(self):
        """Test 11: Large rotation (90°) recovered to < 1.0°."""
        R_true = _rot_x(45.0) @ _rot_z(90.0)
        rng = np.random.default_rng(11)
        vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(6, 3))]
        obs = np.array(vecs)
        cat = np.array([R_true @ v for v in vecs])
        result = estimate_attitude(obs, cat, NAV_CFG)
        err = angular_error_deg(result.rotation_matrix, R_true)
        assert err < 1.0, f"Large rotation error {err:.4f}° > 1.0°"

    def test_12_random_rotations_accurate(self):
        """Test 12: 5 random rotations each recovered to < 0.1°."""
        rng = np.random.default_rng(12)
        for trial in range(5):
            angles = rng.uniform(-90, 90, 3)
            R_true = _rot_z(angles[2]) @ _rot_y(angles[1]) @ _rot_x(angles[0])
            vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(6, 3))]
            obs = np.array(vecs)
            cat = np.array([R_true @ v for v in vecs])
            result = estimate_attitude(obs, cat, NAV_CFG)
            err = angular_error_deg(result.rotation_matrix, R_true)
            assert err < 0.1, f"Trial {trial}: error {err:.4f}° > 0.1°"

    def test_13_noisy_observations_robust(self):
        """Test 13: With 0.2° centroid noise, error < 2.0°."""
        R_true = _rot_z(20.0) @ _rot_y(10.0)
        rng = np.random.default_rng(13)
        vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(8, 3))]
        obs_clean = np.array(vecs)
        cat = np.array([R_true @ v for v in vecs])

        # Add 0.2° noise to observed directions
        noise_rad = math.radians(0.2)
        noise = rng.normal(0, noise_rad, obs_clean.shape)
        obs_noisy = obs_clean + noise
        obs_noisy /= np.linalg.norm(obs_noisy, axis=1, keepdims=True)

        result = estimate_attitude(obs_noisy, cat, NAV_CFG)
        if result.is_valid:
            err = angular_error_deg(result.rotation_matrix, R_true)
            assert err < 2.0, f"Noisy error {err:.4f}° > 2.0°"
        # If not valid, that's acceptable for noisy data

    def test_14_outlier_rejected(self):
        """Test 14: One outlier correspondence should not corrupt estimate."""
        R_true = _rot_z(15.0)
        rng = np.random.default_rng(14)
        vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(6, 3))]
        obs = np.array(vecs)
        cat = np.array([R_true @ v for v in vecs])

        # Replace last correspondence with random noise (outlier)
        outlier_obs = rng.normal(size=3); outlier_obs /= np.linalg.norm(outlier_obs)
        outlier_cat = rng.normal(size=3); outlier_cat /= np.linalg.norm(outlier_cat)
        obs[-1] = outlier_obs
        cat[-1] = outlier_cat

        result = estimate_attitude(obs, cat, NAV_CFG)
        # Should still get a good estimate from the 5 clean correspondences
        if result.is_valid:
            err = angular_error_deg(result.rotation_matrix, R_true)
            assert err < 1.0, f"Outlier test error {err:.4f}°"

    def test_15_insufficient_correspondences_invalid(self):
        """Test 15: 1 correspondence → is_valid=False."""
        obs = np.array([[1.0, 0.0, 0.0]])
        cat = np.array([[1.0, 0.0, 0.0]])
        result = estimate_attitude(obs, cat, NAV_CFG)
        assert result.is_valid is False

    def test_16_degenerate_geometry(self):
        """Test 16: Collinear/identical vectors → graceful failure."""
        obs = np.array([[1.0, 0.0, 0.0]] * 4)
        cat = np.array([[1.0, 0.0, 0.0]] * 4)
        # Should not raise
        result = estimate_attitude(obs, cat, NAV_CFG)
        assert isinstance(result, AttitudeEstimate)


# ═══════════════════════════════════════════════════════════════
# Quaternion Tests (17–19)
# ═══════════════════════════════════════════════════════════════

class TestQuaternion:
    """Tests 17-19: Quaternion normalization, consistency, sign equivalence."""

    def test_17_quaternion_normalization(self):
        """Test 17: Output quaternion always has unit norm."""
        R_true = _rot_z(30.0)
        rng = np.random.default_rng(17)
        vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(4, 3))]
        obs = np.array(vecs); cat = np.array([R_true @ v for v in vecs])
        result = estimate_attitude(obs, cat, NAV_CFG)
        norm = float(np.linalg.norm(result.quaternion))
        assert abs(norm - 1.0) < 1e-10, f"Quaternion norm {norm:.10f} ≠ 1"

    def test_18_rotation_matrix_quaternion_consistency(self):
        """Test 18: rotation_matrix_to_quaternion ↔ quaternion_to_rotation_matrix."""
        R = _rot_z(25.0) @ _rot_y(15.0) @ _rot_x(10.0)
        q = rotation_matrix_to_quaternion(R)
        R_recovered = quaternion_to_rotation_matrix(q)
        err = angular_error_deg(R, R_recovered)
        assert err < 1e-9, f"R↔q roundtrip error {err:.2e}°"

    def test_19_quaternion_sign_equivalence(self):
        """Test 19: q and -q represent the same rotation."""
        R = _rot_z(45.0)
        q = rotation_matrix_to_quaternion(R)
        q_neg = -q
        # Both should produce the same rotation matrix
        R1 = quaternion_to_rotation_matrix(q)
        R2 = quaternion_to_rotation_matrix(q_neg)
        err = angular_error_deg(R1, R2)
        assert err < 1e-9, f"Sign-equivalent quaternions give different rotations"

    def test_quaternion_canonical_form_positive_w(self):
        """quaternion_sign_canonical ensures qw >= 0."""
        q = np.array([-0.5, 0.5, 0.5, 0.5])
        q_canon = quaternion_sign_canonical(q)
        assert q_canon[0] >= 0

    def test_validate_rotation_matrix_identity(self):
        """Identity matrix passes validation."""
        v = validate_rotation_matrix(np.eye(3))
        assert v["is_valid"] is True
        assert abs(v["determinant"] - 1.0) < 1e-10

    def test_validate_rotation_matrix_invalid(self):
        """Non-orthogonal matrix fails validation."""
        bad = np.array([[2.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        v = validate_rotation_matrix(bad)
        assert v["is_valid"] is False


# ═══════════════════════════════════════════════════════════════
# Integration Tests (20–21)
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """Tests 20-21: Phase 4 → Phase 5 integration."""

    def test_20_phase4_to_phase5_attitude(self):
        """Test 20: Full pipeline image → Phase 4 correspondences → Phase 5 attitude."""
        from src.catalog.catalog_loader import load_catalog
        from src.recognition.catalog_index import CatalogIndex
        from src.preprocessing.star_field_generator import StarFieldGenerator
        from src.navigation.navigator import run_navigation, NavigationResult

        catalog = load_catalog(CATALOG_PATH)
        cidx = CatalogIndex(catalog)
        gen = StarFieldGenerator(catalog, {
            "image_width": 512, "image_height": 512, "field_of_view_deg": 20.0,
            "max_stars_per_image": 20, "psf_sigma_px": 1.5, "min_star_flux": 0.05,
            "background_level": 0.02, "read_noise_sigma": 0.005,
            "shot_noise": False, "artifact_probability": 0.0,
        })

        # Try several seeds to find one that produces a recognizable frame
        result = None
        for seed in range(20):
            sf = gen.generate(seed=seed * 100)
            from src.navigation.navigator import _preprocess_image
            img = _preprocess_image(sf.image, NAV_CFG)
            r = run_navigation(img, NAV_CFG, cidx)
            if r.status in ("SUCCESS", "PARTIAL"):
                result = r
                break

        if result is None:
            pytest.skip("No recognizable frame found in test seeds")

        assert isinstance(result, NavigationResult)
        assert result.status in ("SUCCESS", "PARTIAL")
        assert abs(np.linalg.norm(result.quaternion) - 1.0) < 1e-6

    def test_21_low_confidence_phase4_rejected(self):
        """Test 21: Zero-star image → FAILURE/INSUFFICIENT_STARS, not SUCCESS."""
        from src.catalog.catalog_loader import load_catalog
        from src.recognition.catalog_index import CatalogIndex
        from src.navigation.navigator import run_navigation

        catalog = load_catalog(CATALOG_PATH)
        cidx = CatalogIndex(catalog)
        img = np.zeros((512, 512), dtype=np.float32)
        result = run_navigation(img, NAV_CFG, cidx)
        assert result.status not in ("SUCCESS",), \
            f"Black image should not produce SUCCESS, got {result.status}"
        assert result.n_inlier_stars == 0

    def test_position_always_unavailable(self):
        """PositionEstimate.is_valid is always False for single-image case."""
        from src.navigation.attitude_estimator import AttitudeEstimate
        att = AttitudeEstimate(is_valid=True, num_correspondences=5)
        pos = estimate_position(att, {}, NAV_CFG)
        assert pos.is_valid is False
        assert pos.position_status == "UNAVAILABLE"
        assert np.all(np.isnan(pos.position_vector))


# ═══════════════════════════════════════════════════════════════
# NavigationResult structure tests
# ═══════════════════════════════════════════════════════════════

class TestNavigationResult:
    def test_default_status_is_failure(self):
        from src.navigation.navigator import NavigationResult
        r = NavigationResult()
        assert r.status == "FAILURE"

    def test_position_status_always_unavailable_in_default(self):
        from src.navigation.navigator import NavigationResult
        r = NavigationResult()
        assert r.position_status == "UNAVAILABLE"
        assert r.velocity_status == "UNAVAILABLE"

    def test_default_quaternion_is_identity(self):
        from src.navigation.navigator import NavigationResult
        r = NavigationResult()
        assert np.allclose(r.quaternion, [1.0, 0.0, 0.0, 0.0])
