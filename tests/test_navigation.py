"""
test_navigation.py
==================
Tests for Phase 5 navigation modules.

Updated from Phase 5 stubs — estimate_attitude and estimate_position
are now fully implemented.
"""
from __future__ import annotations
import pytest
import numpy as np
from src.navigation.attitude_estimator import AttitudeEstimate, estimate_attitude
from src.navigation.position_estimator import PositionEstimate, estimate_position

NAV_CFG = {
    "navigation": {
        "min_correspondences": 2,
        "max_residual_threshold_deg": 2.0,
        "outlier_rejection_threshold_deg": 2.0,
        "outlier_rejection_max_iter": 3,
    }
}


@pytest.fixture()
def identity_correspondences():
    d = np.eye(3, dtype=np.float64)
    return d, d.copy()

@pytest.fixture()
def mismatched_correspondences():
    obs = np.array([[1.,0.,0.],[0.,1.,0.]])
    cat = np.array([[1.,0.,0.]])
    return obs, cat

@pytest.fixture()
def valid_attitude_estimate():
    return AttitudeEstimate(quaternion=np.array([1.,0.,0.,0.]),
                            rotation_matrix=np.eye(3), residual_deg=0.0,
                            num_correspondences=3, is_valid=True)


class TestAttitudeEstimateDefaults:
    def test_default_quaternion_is_unit(self):
        est = AttitudeEstimate()
        assert np.allclose(est.quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_default_rotation_matrix_is_identity(self):
        assert np.allclose(AttitudeEstimate().rotation_matrix, np.eye(3))

    def test_default_is_not_valid(self):
        assert AttitudeEstimate().is_valid is False

    def test_has_euler_angles_field(self):
        est = AttitudeEstimate()
        assert hasattr(est, "euler_angles_deg")
        assert len(est.euler_angles_deg) == 3

    def test_has_attitude_confidence_field(self):
        est = AttitudeEstimate()
        assert hasattr(est, "attitude_confidence")
        assert est.attitude_confidence == 0.0


class TestEstimateAttitude:
    def test_identity_no_longer_raises(self, identity_correspondences):
        """Phase 5: estimate_attitude is implemented, not NotImplementedError."""
        obs, cat = identity_correspondences
        result = estimate_attitude(obs, cat, NAV_CFG)
        assert isinstance(result, AttitudeEstimate)

    def test_identity_produces_valid_result(self, identity_correspondences):
        obs, cat = identity_correspondences
        result = estimate_attitude(obs, cat, NAV_CFG)
        assert result.is_valid
        assert result.residual_deg < 0.01

    def test_raises_value_error_on_shape_mismatch(self, mismatched_correspondences):
        obs, cat = mismatched_correspondences
        with pytest.raises(ValueError):
            estimate_attitude(obs, cat, NAV_CFG)

    def test_too_few_correspondences_invalid(self):
        obs = np.array([[1.0, 0.0, 0.0]])
        cat = np.array([[1.0, 0.0, 0.0]])
        result = estimate_attitude(obs, cat, NAV_CFG)
        assert result.is_valid is False


class TestPositionEstimateDefaults:
    def test_default_is_not_valid(self):
        assert PositionEstimate().is_valid is False

    def test_default_position_is_nan(self):
        assert np.all(np.isnan(PositionEstimate().position_vector))

    def test_has_position_status(self):
        assert PositionEstimate().position_status == "UNAVAILABLE"


class TestEstimatePosition:
    def test_no_longer_raises(self, valid_attitude_estimate):
        """Phase 5: estimate_position returns PositionEstimate, doesn't raise."""
        result = estimate_position(valid_attitude_estimate, {}, NAV_CFG)
        assert isinstance(result, PositionEstimate)

    def test_always_invalid_single_image(self, valid_attitude_estimate):
        result = estimate_position(valid_attitude_estimate, {}, NAV_CFG)
        assert result.is_valid is False

    def test_position_unavailable_status(self, valid_attitude_estimate):
        result = estimate_position(valid_attitude_estimate, {}, NAV_CFG)
        assert result.position_status == "UNAVAILABLE"

    def test_notes_explain_why(self, valid_attitude_estimate):
        result = estimate_position(valid_attitude_estimate, {}, NAV_CFG)
        assert len(result.notes) > 20  # non-trivial explanation
