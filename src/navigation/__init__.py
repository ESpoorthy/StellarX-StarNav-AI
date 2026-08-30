"""
src.navigation — Phase 5
------------------------
Spacecraft attitude determination from verified star correspondences.

Modules
-------
camera_model      : CameraModel — pinhole projection, pixel↔unit-vector
attitude_estimator: estimate_attitude() — Wahba/SVD, quaternion, Euler angles
position_estimator: estimate_position() — always UNAVAILABLE (single image)
navigator         : run_navigation(), run_full_pipeline(), NavigationResult
"""
