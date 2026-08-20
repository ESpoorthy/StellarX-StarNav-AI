"""
test_navigation.py
==================
Unit tests for src.navigation.attitude_estimator and
src.navigation.position_estimator.

Tests are structured to cover the public API of each module.
Implementation of test bodies is deferred until Phase 5, when the
corresponding source functions are implemented.

Run with:
    pytest tests/test_navigation.py
"""

from __future__ import annotations

import pytest
import numpy as np

from src.navigation.attitude_estimator import AttitudeEstimate
from src.navigation.position_estimator import PositionEstimate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def identity_correspondences() -> tuple[np.ndarray, np.ndarray]:
    """Return perfectly aligned observed and catalog direction pairs.

    Both arrays are identical unit vectors — the expected attitude solution
    is the identity rotation.
    """
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return directions, directions.copy()


@pytest.fixture()
def mismatched_correspondences() -> tuple[np.ndarray, np.ndarray]:
    """Return direction arrays with inconsistent shapes to test validation."""
    observed = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    catalog = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    return observed, catalog


@pytest.fixture()
def navigation_config() -> dict:
    """Return a minimal navigation configuration dict."""
    return {
        "attitude_method": None,
        "position_method": None,
    }


@pytest.fixture()
def valid_attitude_estimate() -> AttitudeEstimate:
    """Return a valid AttitudeEstimate (identity rotation)."""
    return AttitudeEstimate(
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        rotation_matrix=np.eye(3),
        residual_deg=0.0,
        num_correspondences=3,
        is_valid=True,
    )


# ---------------------------------------------------------------------------
# AttitudeEstimate dataclass tests
# ---------------------------------------------------------------------------


class TestAttitudeEstimateDefaults:
    """Tests for AttitudeEstimate default values."""

    def test_default_quaternion_is_unit(self):
        """Default quaternion should be [1, 0, 0, 0] (identity)."""
        est = AttitudeEstimate()
        assert np.allclose(est.quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_default_rotation_matrix_is_identity(self):
        """Default rotation matrix should be the 3×3 identity."""
        est = AttitudeEstimate()
        assert np.allclose(est.rotation_matrix, np.eye(3))

    def test_default_is_not_valid(self):
        """Default AttitudeEstimate should have is_valid=False."""
        est = AttitudeEstimate()
        assert est.is_valid is False


# ---------------------------------------------------------------------------
# estimate_attitude tests
# ---------------------------------------------------------------------------


class TestEstimateAttitude:
    """Tests for attitude_estimator.estimate_attitude."""

    def test_raises_not_implemented(
        self, identity_correspondences, navigation_config
    ):
        """estimate_attitude should raise NotImplementedError until Phase 5."""
        from src.navigation.attitude_estimator import estimate_attitude

        observed, catalog = identity_correspondences
        with pytest.raises(NotImplementedError):
            estimate_attitude(observed, catalog, navigation_config)

    def test_raises_value_error_on_shape_mismatch(
        self, mismatched_correspondences, navigation_config
    ):
        """estimate_attitude should raise ValueError on mismatched shapes."""
        from src.navigation.attitude_estimator import estimate_attitude

        observed, catalog = mismatched_correspondences
        with pytest.raises(ValueError):
            estimate_attitude(observed, catalog, navigation_config)

    # TODO (Phase 5): add tests for:
    #   - identity correspondences → quaternion ≈ [1, 0, 0, 0]
    #   - known rotation → recovered quaternion within tolerance
    #   - residual_deg is non-negative
    #   - is_valid=True when residual is below threshold
    #   - is_valid=False when fewer than minimum correspondences provided


# ---------------------------------------------------------------------------
# PositionEstimate dataclass tests
# ---------------------------------------------------------------------------


class TestPositionEstimateDefaults:
    """Tests for PositionEstimate default values."""

    def test_default_is_not_valid(self):
        """Default PositionEstimate should have is_valid=False."""
        est = PositionEstimate()
        assert est.is_valid is False

    def test_default_position_is_nan(self):
        """Default position_vector should contain NaN values."""
        est = PositionEstimate()
        assert np.all(np.isnan(est.position_vector))


# ---------------------------------------------------------------------------
# estimate_position tests
# ---------------------------------------------------------------------------


class TestEstimatePosition:
    """Tests for position_estimator.estimate_position."""

    def test_raises_not_implemented(
        self, valid_attitude_estimate, navigation_config
    ):
        """estimate_position should raise NotImplementedError until Phase 5."""
        from src.navigation.position_estimator import estimate_position

        with pytest.raises(NotImplementedError):
            estimate_position(valid_attitude_estimate, {}, navigation_config)

    # TODO (Phase 5): add tests for:
    #   - returns PositionEstimate (never raises) when methodology is unsupported
    #   - is_valid=False with explanatory note when estimation is not possible
    #   - valid estimate when sufficient data is available (if implemented)
