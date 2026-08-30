# Requirements Document

## Introduction

This document specifies the requirements for Phases 4, 5, and 6 of the StellarX-StarNav-AI project, developed for Smart India Hackathon 2026 (SIH 2026). The system performs autonomous spacecraft attitude determination from star-field imagery without dependence on external positioning infrastructure.

Phases 1–3 (already complete) deliver a 50-star Hipparcos catalog, star detection yielding `list[StarCandidate]` with `.x`, `.y`, `.brightness`, `.peak`, `.area` attributes, a 90-dimensional float32 feature vector (45 pairwise pixel distances + 45 brightness ratios for top-10 stars), and a trained scikit-learn classifier (`RandomForest`/`KNN`/`MLP`) that classifies star fields into sky-cell labels with `RecognitionResult(pattern_id, confidence, top_k_predictions, raw_output, latency_ms)`.

Phases 4–6 must build on these foundations to implement: geometric catalog matching with RANSAC outlier rejection (Phase 4), Wahba/SVD-based attitude estimation (Phase 5), and a fully profiled, optimized pipeline with runnable evaluation scripts (Phase 6).

The stub modules to be implemented are:
- `src/catalog/pattern_matcher.py` — `match_pattern()`, `MatchResult`
- `src/navigation/attitude_estimator.py` — `estimate_attitude()`, `AttitudeEstimate`
- `src/navigation/position_estimator.py` — `estimate_position()`, `PositionEstimate`
- `src/utils/visualization.py` — all plot functions

Supporting modules already partially scaffolded (and being completed in these phases):
- `src/recognition/catalog_index.py` — `CatalogIndex`, `build_catalog_index()`
- `src/recognition/pattern_builder.py` — `build_pattern()`, `StarPattern`, `pixels_to_unit_vector()`
- `src/recognition/pattern_matcher.py` — `run_recognition()`, `RecognitionOutput`, `RecognitionStatus`
- `src/navigation/navigator.py` — `run_navigation()`, `NavigationResult`
- `src/navigation/camera_model.py` — `CameraModel`
- `src/optimization/pipeline.py` — `OptimizedPipeline`, `BenchmarkResult`

---

## Glossary

- **Attitude**: The orientation of the spacecraft's camera frame relative to the inertial (J2000 ICRS) reference frame, expressed as a rotation.
- **Boresight**: The optical axis of the camera sensor; the +Z direction in the camera coordinate frame.
- **Camera Frame**: Right-handed coordinate system with +Z along the boresight (into the scene), +X aligned with the image column direction (right), and +Y aligned opposite the image row direction (up).
- **CatalogIndex**: A KD-tree-indexed structure built from the `StarCatalog` that enables O(log N) angular-distance queries and precomputed pairwise angular separations.
- **CatalogStar**: A single entry from the Hipparcos catalog with `star_id`, `ra_deg`, `dec_deg`, `magnitude`, and a `unit_vector()` method returning `[cos(dec)cos(ra), cos(dec)sin(ra), sin(dec)]`.
- **Chord Distance**: The Euclidean distance between two unit vectors on the unit sphere, equal to `2*sin(θ/2)` where θ is the angular separation. Used by the KD-tree for efficient cone queries.
- **Focal Length in Pixels (focal_px)**: `(image_width/2) / tan(fov_deg/2)`. For the default 512×512 image with 20° FoV, `focal_px ≈ 1448.2 px`.
- **FoV**: Field of view of the simulated sensor, 20 degrees full-angle by default (sourced from `config['dataset']['field_of_view_deg']`).
- **Gnomonic Projection**: The tangent-plane map used in `star_field_generator.py` to project catalog stars onto the focal plane. Inverse used in Phase 4 to convert pixel coordinates back to camera-frame unit vectors.
- **IdentifiedStar**: A detected star successfully matched to a catalog entry, carrying observed pixel coordinates, camera-frame unit vector, catalog ID, catalog inertial unit vector, angular residual (degrees), per-star confidence, and brightness.
- **Inertial Frame**: J2000 ICRS. Unit vectors are defined as `[cos(dec)cos(ra), cos(dec)sin(ra), sin(dec)]`.
- **J2000 ICRS**: The standard celestial reference frame used in the Hipparcos catalog, with origin at the solar system barycenter and axes fixed to the mean equinox and equator of 1 January 2000.
- **MatchedPattern**: Summary of the geometric pattern match: pattern type, candidate count, inlier count, total observed stars, geometric residual (degrees), and overall confidence.
- **NavigationResult**: The complete output of the navigation pipeline for a single image, containing attitude quaternion, rotation matrix, Euler angles, confidence, residual, star counts, timing, and a position note.
- **Neural Prior**: The top-k sky-cell predictions and confidences from the Phase 3 sklearn classifier. Used as a soft bias (not a hard constraint) on candidate correspondences during geometric matching.
- **OptimizedPipeline**: A pipeline wrapper that loads and indexes the star catalog exactly once at construction, caches the `CatalogIndex`, and exposes `process_image()` and `benchmark()` methods.
- **Pixel Coordinate**: A 2D location `(col, row)` in the image, where `col` increases rightward and `row` increases downward, 0-indexed.
- **Principal Point**: The pixel coordinates of the optical center, set to `(image_width/2, image_height/2)` for the simulated sensor.
- **Quaternion**: A unit quaternion `[qw, qx, qy, qz]` representing the rotation from the camera frame to the inertial frame. Convention: `v_inertial = R @ v_camera`.
- **RANSAC**: Random Sample Consensus. An iterative algorithm that selects minimal hypothesis subsets, counts geometric inliers, and retains the hypothesis with the most inliers.
- **RecognitionOutput**: The complete output of the pattern recognition step, including `identified_stars`, `matched_pattern`, `status`, timing, star counts, confidence, and neural prior information.
- **RecognitionStatus**: An enumeration: `SUCCESS`, `PARTIAL`, `LOW_CONFIDENCE`, `FAILURE`.
- **RecognitionResult**: Phase 3 output from `run_inference()` in `src/models/inference.py`, carrying `pattern_id`, `confidence`, `top_k_predictions`, `raw_output`, `latency_ms`.
- **Rotation Matrix**: A 3×3 orthogonal matrix with determinant +1 that encodes the same rotation as the quaternion.
- **Sky Cell**: A discrete region of the sky identified by the Phase 3 classifier. Derived from `boresight_to_label(ra_deg, dec_deg, n_sky_cells)` in `sklearn_classifier.py`.
- **StarCandidate**: Phase 2 output from `detect_stars()`, carrying `.x` (column centroid), `.y` (row centroid), `.brightness` (integrated flux), `.peak` (peak pixel), and `.area` (blob pixel count).
- **StarPattern**: The angular descriptor built from detected stars: camera-frame unit vectors (N×3), pixel coordinates (N×2), brightnesses (N,), and the N×N pairwise angular separation matrix (degrees).
- **TRIAD**: A closed-form rotation estimator from exactly two vector correspondences. Used as the RANSAC hypothesis generator; result is SVD-corrected to ensure `det(R) = +1`.
- **Unit Vector**: A 3D vector of magnitude 1.0. All directions (catalog and observed) are normalized to unit vectors before any angular computation.
- **Vote Matrix**: An (N_obs × N_catalog) integer array where `votes[i][k]` counts the number of angular-distance pair matches supporting the hypothesis that observed star `i` corresponds to catalog star `k`.
- **Wahba's Problem**: The weighted least-squares problem: find rotation R minimizing `sum_i w_i * ||cat_i - R @ obs_i||^2`. Solved by the SVD method (Markley 1988).
- **ZYX Euler Angles**: `[yaw, pitch, roll]` in degrees. Convention: `R = R_z(yaw) @ R_y(pitch) @ R_x(roll)`. Computed as: `pitch = arcsin(-R[2,0])`, `yaw = atan2(R[1,0], R[0,0])`, `roll = atan2(R[2,1], R[2,2])`. Gimbal lock at `|pitch| = 90°` is handled by setting `roll = 0`.

---

## Requirements

---

### Requirement 1: Catalog KD-Tree Index

**User Story:** As a navigation engineer, I want an efficient spatial index over the star catalog, so that angular-distance queries execute in O(log N) time rather than O(N) linear scans.

#### Acceptance Criteria

1. THE `CatalogIndex` SHALL build a `scipy.spatial.KDTree` on the array of catalog star unit vectors (shape N×3) at construction time.
2. THE `CatalogIndex` SHALL convert each `CatalogStar.unit_vector()` result from `catalog_loader.py` into a row of the unit-vector array, preserving the order and indexing of the source `StarCatalog`.
3. WHEN `query_cone(ra_deg, dec_deg, radius_deg)` is called, THE `CatalogIndex` SHALL use chord-distance threshold `2 * sin(radians(radius_deg) / 2)` for the KD-tree ball query to retrieve all stars within `radius_deg` of the query direction.
4. WHEN `find_pairs_by_angle(angle_deg, tolerance_deg)` is called, THE `CatalogIndex` SHALL return all precomputed catalog star pairs whose angular separation falls within `[angle_deg - tolerance_deg, angle_deg + tolerance_deg]`, sorted by ascending absolute deviation from `angle_deg`.
5. THE `CatalogIndex` SHALL precompute all pairwise angular separations for the 50-star catalog (1225 pairs) at construction time using `arccos(dot(u_i, u_j))` with dot products clamped to `[-1.0, 1.0]`.
6. THE `CatalogIndex` SHALL store precomputed pair angles in a dictionary keyed by `(min(i,j), max(i,j))` for O(1) individual pair lookups via `query_angular_distance(i, j)`.
7. IF the input `StarCatalog` is empty, THEN THE `CatalogIndex` SHALL return an empty list from `query_cone()`, an empty list from `find_pairs_by_angle()`, and `NaN` from `query_angular_distance()` without raising an exception.
8. THE `build_catalog_index(catalog_path, config, mag_limit)` factory function SHALL call `load_catalog()` followed by `CatalogIndex()` construction and return the populated index in a single call.

---

### Requirement 2: Pixel-to-Unit-Vector Conversion

**User Story:** As a navigation engineer, I want pixel centroids from star detection to be converted to calibrated camera-frame unit vectors, so that geometric matching operates in 3D angular space rather than 2D pixel space.

#### Acceptance Criteria

1. THE `pixels_to_unit_vector(col, row, image_width, image_height, fov_deg)` function SHALL compute focal length as `focal_px = (image_width / 2) / tan(radians(fov_deg / 2))`.
2. THE `pixels_to_unit_vector` function SHALL compute the camera-frame direction as `x_cam = (col - cx) / focal_px`, `y_cam = -(row - cy) / focal_px`, `z_cam = 1.0`, where `cx = image_width / 2` and `cy = image_height / 2`.
3. THE `pixels_to_unit_vector` function SHALL return a unit vector by dividing `[x_cam, y_cam, z_cam]` by its L2 norm, with the fallback `[0, 0, 1]` when the norm is less than `1e-12`.
4. THE `CameraModel.pixel_to_unit_vector(col, row)` method SHALL implement the same formula as `pixels_to_unit_vector`, using `self.cx`, `self.cy`, and `self.focal_px` from the `CameraModel` instance.
5. THE `CameraModel.unit_vector_to_pixel(unit_vec)` method SHALL implement the inverse projection: `col = (x/z) * focal_px + cx`, `row = -(y/z) * focal_px + cy`, and SHALL return `(nan, nan)` when `z <= 0`.
6. WHEN the boresight pixel `(cx, cy)` is provided, THE `pixels_to_unit_vector` function SHALL return the vector `[0, 0, 1]` (pointing along +Z boresight) to within floating-point precision.
7. THE `CameraModel.from_config(config)` class method SHALL read `image_width`, `image_height`, and `field_of_view_deg` from `config['dataset']` and construct the `CameraModel` with derived `cx`, `cy`, and `focal_px`.

---

### Requirement 3: Star Pattern Construction

**User Story:** As a navigation engineer, I want a rotation- and translation-invariant angular descriptor built from detected stars, so that the pattern can be matched against the catalog regardless of spacecraft orientation.

#### Acceptance Criteria

1. THE `build_pattern(stars, config)` function SHALL sort the input `StarCandidate` list by descending `brightness` and select the top `config['features']['max_stars']` stars (default 10).
2. THE `build_pattern` function SHALL call `pixels_to_unit_vector` for each selected star to produce a `(N, 3)` unit-vector array in the camera frame.
3. THE `build_pattern` function SHALL compute the symmetric N×N pairwise angular separation matrix using `arccos(dot(u_i, u_j))` with all dot products clamped to `[-1.0, 1.0]`.
4. THE `StarPattern` dataclass SHALL carry: `unit_vectors` (N×3 float64), `pixel_coords` (N×2 float64 with `(col, row)` ordering), `brightnesses` (N float64), `pairwise_angles_deg` (N×N float64), `n_stars` (int), `focal_px` (float), `image_width` (int), `image_height` (int), and `fov_deg` (float).
5. WHEN `build_pattern` is called with zero stars, THE `StarPattern` SHALL be returned with `n_stars=0` and all array fields having zero rows, without raising an exception.
6. WHEN `build_pattern` is called with exactly one star, THE `StarPattern` SHALL be returned with `n_stars=1` and a `pairwise_angles_deg` matrix of shape `(1, 1)` containing a single zero entry.
7. THE pairwise angular separation matrix SHALL be symmetric: `pairwise_angles_deg[i, j] == pairwise_angles_deg[j, i]` for all valid index pairs.
8. THE diagonal entries of `pairwise_angles_deg` SHALL be `0.0` for all valid indices.

---

### Requirement 4: Angular Pair Voting

**User Story:** As a navigation engineer, I want a voting mechanism that counts how many angular-distance pair matches support each hypothetical correspondence, so that genuine star-catalog matches accumulate higher vote scores than false coincidences.

#### Acceptance Criteria

1. THE `run_recognition` function SHALL maintain a vote matrix `votes` of shape `(n_obs, n_catalog)` initialised to all zeros, where `votes[i][k]` accumulates the number of angular-distance pair matches supporting the hypothesis that observed star `i` corresponds to catalog star `k`.
2. WHEN processing observed pair `(i, j)` with angular separation `obs_angle_deg`, THE `run_recognition` function SHALL query `catalog_index.find_pairs_by_angle(obs_angle_deg, angle_tolerance_deg)` and increment `votes[i][cat_k]`, `votes[j][cat_l]`, `votes[i][cat_l]`, and `votes[j][cat_k]` for every returned catalog pair `(cat_k, cat_l)`.
3. THE `run_recognition` function SHALL skip observed pairs whose angular separation is less than `0.01` degrees (degenerate coincident detections).
4. THE angle tolerance for pair matching SHALL be read from `config['recognition']['angle_tolerance_deg']` (default `0.5` degrees).
5. AFTER vote accumulation, THE `run_recognition` function SHALL perform greedy correspondence assignment: for each observed star in descending order of its maximum vote count, assign the catalog star with the highest vote count that has not already been assigned.
6. THE greedy correspondence assignment SHALL produce at most one catalog star per observed star and at most one observed star per catalog star (no double-use of catalog stars).

---

### Requirement 5: RANSAC Outlier Rejection

**User Story:** As a navigation engineer, I want RANSAC-based outlier rejection applied to the initial vote correspondences, so that one or more false matches do not corrupt the attitude estimate.

#### Acceptance Criteria

1. THE `run_recognition` function SHALL perform RANSAC over `config['recognition']['ransac_iterations']` (default 50) iterations using randomly sampled pairs of 2 correspondences as minimal hypotheses.
2. WHEN a 2-correspondence RANSAC hypothesis is drawn, THE `run_recognition` function SHALL compute the rotation using the TRIAD algorithm: build orthonormal triads from the two observed and two catalog vector pairs, then compute `R = M_ref @ M_obs.T`, SVD-corrected to enforce `det(R) = +1`.
3. FOR each RANSAC hypothesis rotation, THE `run_recognition` function SHALL classify a correspondence as an inlier if the angular residual `arccos(dot(normalize(R @ obs_vec), cat_vec))` is at most `config['recognition']['max_residual_deg']` (default `1.0` degree).
4. THE `run_recognition` function SHALL retain the RANSAC hypothesis with the maximum inlier count, and SHALL use it as the initial rotation for the subsequent SVD refinement step.
5. IF two TRIAD input vectors in a hypothesis subset are nearly parallel (cross-product norm less than `1e-10`), THEN THE `run_recognition` function SHALL skip that hypothesis and sample a new one without incrementing the iteration counter.
6. AFTER RANSAC, THE `run_recognition` function SHALL refit the rotation using all inlier correspondences via the Wahba/SVD method, replacing the TRIAD hypothesis rotation with the refined result.

---

### Requirement 6: Wahba/SVD Attitude Refinement in Pattern Recognition

**User Story:** As a navigation engineer, I want the best rotation to be refined using all inlier correspondences simultaneously, so that the final rotation minimizes the sum of squared angular residuals rather than fitting only two stars.

#### Acceptance Criteria

1. THE `_wahba_svd(obs_vecs, cat_vecs, weights)` function SHALL compute the attitude profile matrix `B = sum_i (weights[i] * outer(cat_vecs[i], obs_vecs[i]))` summed over all inlier correspondences.
2. THE `_wahba_svd` function SHALL decompose B using `numpy.linalg.svd` and compute `R = U @ diag(1, 1, det(U @ V^T)) @ V^T` to ensure the result is a proper rotation matrix with determinant exactly `+1`.
3. WHEN `numpy.linalg.LinAlgError` is raised during SVD, THE `_wahba_svd` function SHALL return `None` without propagating the exception to the caller.
4. THE `run_recognition` function SHALL use equal weights for the Wahba SVD refinement step (all inlier weights set to `1.0`) unless per-star confidence scores from `IdentifiedStar.confidence` are available.
5. WHEN fewer than 2 inlier correspondences remain after RANSAC, THE `run_recognition` function SHALL skip the SVD refinement step and proceed directly to status determination using the RANSAC inlier count.

---

### Requirement 7: Neural Prior Integration

**User Story:** As a navigation engineer, I want the Phase 3 classifier output to contribute a soft bonus to the recognition confidence score, so that high neural confidence in the correct sky cell improves the overall confidence without making the system dependent on the neural classifier.

#### Acceptance Criteria

1. THE `run_recognition(pattern, catalog_index, config, neural_result)` function SHALL accept an optional `neural_result` parameter of type `RecognitionResult` from `src/models/inference.py`, defaulting to `None`.
2. WHEN `neural_result` is not `None`, THE `run_recognition` function SHALL extract `neural_pattern_id = neural_result.pattern_id` and `neural_confidence = neural_result.confidence`.
3. THE confidence score formula SHALL be: `confidence = 0.5 * inlier_fraction + 0.4 * residual_quality + 0.1 * neural_bonus`, where `inlier_fraction = n_inliers / max(n_matched, 1)`, `residual_quality = max(0.0, 1.0 - mean_residual_deg / max_residual_deg)`, and `neural_bonus = neural_confidence` when `neural_confidence > 0` else `0.0`.
4. WHEN `neural_result` is `None`, THE `run_recognition` function SHALL set `neural_confidence = 0.0` and `neural_pattern_id = None` so the neural bonus term is `0.0` and the confidence formula reduces to the geometric terms only.
5. THE neural prior SHALL NOT be used as a hard constraint that filters candidate correspondences; it SHALL influence only the scalar confidence score and the `neural_pattern_id`/`neural_confidence` fields of `RecognitionOutput`.
6. THE `RecognitionOutput` dataclass SHALL include `neural_pattern_id: str | None` and `neural_confidence: float` fields carrying the neural prior provenance information.

---

### Requirement 8: Recognition Confidence Scoring and Status

**User Story:** As a spacecraft systems engineer, I want every recognition attempt to yield a structured status code and scalar confidence score, so that downstream navigation logic can decide whether to use, flag, or discard the attitude estimate.

#### Acceptance Criteria

1. THE `run_recognition` function SHALL compute the final confidence score using the formula `0.5 * inlier_fraction + 0.4 * residual_quality + 0.1 * neural_bonus` with all component values clamped to `[0.0, 1.0]` before scaling.
2. THE `run_recognition` function SHALL assign `RecognitionStatus.SUCCESS` when `n_inliers >= config['recognition']['min_inliers']` AND `mean_residual_deg <= config['recognition']['max_residual_deg']` AND `confidence >= config['recognition']['confidence_success']`.
3. THE `run_recognition` function SHALL assign `RecognitionStatus.PARTIAL` when `n_inliers >= 2` AND `confidence >= config['recognition']['confidence_partial']` AND the `SUCCESS` conditions are not all met.
4. THE `run_recognition` function SHALL assign `RecognitionStatus.LOW_CONFIDENCE` when `n_inliers >= 1` AND neither `SUCCESS` nor `PARTIAL` conditions are met.
5. THE `run_recognition` function SHALL assign `RecognitionStatus.FAILURE` when `n_inliers == 0` or when fewer than 2 candidate correspondences were found after vote accumulation.
6. THE `RecognitionOutput` dataclass SHALL carry all fields required to reproduce the status determination: `n_observed`, `n_matched`, `n_inliers`, `confidence`, `mean_residual_deg`, `neural_pattern_id`, `neural_confidence`, `processing_time_ms`, `status`, `matched_pattern`, and `identified_stars`.
7. THE `RecognitionOutput.is_successful()` method SHALL return `True` if and only if `status == RecognitionStatus.SUCCESS`.

---

### Requirement 9: Per-Star Identification Output

**User Story:** As a navigation engineer, I want each matched star to carry its own residual and confidence score, so that downstream attitude estimation can apply per-star weights and SIH 2026 evaluators can inspect individual star quality.

#### Acceptance Criteria

1. THE `run_recognition` function SHALL populate the `identified_stars` list with one `IdentifiedStar` entry per RANSAC inlier correspondence.
2. EACH `IdentifiedStar` SHALL carry: `observed_x` (float, pixel column), `observed_y` (float, pixel row), `observed_unit_vec` (np.ndarray shape (3,)), `catalog_id` (str in `"HIP_<N>"` format), `catalog_ra_deg` (float, degrees), `catalog_dec_deg` (float, degrees), `catalog_unit_vec` (np.ndarray shape (3,)), `angular_residual_deg` (float, degrees), `confidence` (float in `[0,1]`), and `brightness` (float).
3. THE per-star angular residual SHALL be computed as `arccos(clamp(dot(normalize(R_final @ obs_unit_vec), cat_unit_vec), -1.0, 1.0))` in degrees, where `R_final` is the SVD-refined rotation.
4. THE per-star confidence SHALL be computed as `max(0.0, 1.0 - angular_residual_deg / max_residual_deg)`, linearly decaying from 1.0 at zero residual to 0.0 at `max_residual_deg`.
5. WHEN the recognition status is `FAILURE` or `LOW_CONFIDENCE` with zero inliers, THE `identified_stars` list SHALL be empty.

---

### Requirement 10: Graceful Handling of Degenerate Inputs in Recognition

**User Story:** As a navigation engineer, I want the recognition pipeline to return a valid `RecognitionOutput` for all edge-case inputs, so that the navigation pipeline never crashes regardless of image quality.

#### Acceptance Criteria

1. WHEN `pattern.n_stars == 0`, THE `run_recognition` function SHALL return `RecognitionOutput(status=RecognitionStatus.FAILURE, n_observed=0)` without performing any matching computation.
2. WHEN `pattern.n_stars == 1`, THE `run_recognition` function SHALL return `RecognitionOutput(status=RecognitionStatus.FAILURE, n_observed=1)` because at least 2 stars are required to form an angular pair.
3. WHEN no catalog pair matches any observed angular distance within `angle_tolerance_deg`, THE `run_recognition` function SHALL return `RecognitionOutput(status=RecognitionStatus.FAILURE, n_matched=0)` with an empty `identified_stars` list.
4. WHEN all candidate correspondences are RANSAC outliers (zero inliers), THE `run_recognition` function SHALL return `RecognitionOutput(status=RecognitionStatus.FAILURE, n_inliers=0)`.
5. WHEN the `CatalogIndex` contains fewer than 2 stars, THE `run_recognition` function SHALL return `RecognitionOutput(status=RecognitionStatus.FAILURE)` without raising an exception.
6. IF an unexpected exception occurs during RANSAC iteration, THEN THE `run_recognition` function SHALL catch the exception, log the error description, and return `RecognitionOutput(status=RecognitionStatus.FAILURE)`.

---

### Requirement 11: Wahba/SVD Attitude Estimation

**User Story:** As a spacecraft systems engineer, I want spacecraft attitude estimated from star direction correspondences using a mathematically rigorous least-squares method, so that the orientation output is provably optimal under the Gaussian noise model.

#### Acceptance Criteria

1. THE `estimate_attitude(observed_directions, catalog_directions, config, weights)` function SHALL solve Wahba's problem by computing the attitude profile matrix `B = sum_i (weights[i] * outer(catalog_directions[i], observed_directions[i]))`.
2. THE `estimate_attitude` function SHALL decompose B via `numpy.linalg.svd` and compute `R = U @ diag(1, 1, det(U @ V^T)) @ V^T` to guarantee `det(R) = +1` (proper rotation, no improper reflection).
3. THE `estimate_attitude` function SHALL raise `ValueError` when `observed_directions.shape != catalog_directions.shape`.
4. WHEN `weights` is `None`, THE `estimate_attitude` function SHALL use uniform weights of `1.0` for all correspondences.
5. THE `estimate_attitude` function SHALL convert the rotation matrix to a unit quaternion `[qw, qx, qy, qz]` using the Shepperd method, normalizing the result to unit magnitude.
6. THE `estimate_attitude` function SHALL compute ZYX Euler angles `[yaw, pitch, roll]` in degrees using: `pitch = degrees(arcsin(-R[2,0]))`, `yaw = degrees(atan2(R[1,0], R[0,0]))`, `roll = degrees(atan2(R[2,1], R[2,2]))`.
7. WHEN `|cos(pitch)| < 1e-6` (gimbal lock at `|pitch| ≈ 90°`), THE `estimate_attitude` function SHALL set `yaw = degrees(atan2(-R[0,1], R[1,1]))` and `roll = 0.0`.
8. THE `estimate_attitude` function SHALL compute `mean_residual_deg` as the arithmetic mean of per-correspondence angular errors `arccos(clamp(dot(normalize(R @ obs_i), cat_i), -1.0, 1.0))` in degrees.
9. THE `estimate_attitude` function SHALL compute `attitude_confidence = clamp(1.0 - residual_deg / max_residual_threshold_deg, 0.0, 1.0)`, where `max_residual_threshold_deg` is read from `config['navigation']['max_residual_threshold_deg']` (default `2.0` degrees).
10. THE `estimate_attitude` function SHALL set `is_valid = True` if and only if `num_correspondences >= min_correspondences` AND `residual_deg < max_residual_threshold_deg`, where `min_correspondences` is read from `config['navigation']['min_correspondences']` (default `2`).
11. WHEN `numpy.linalg.LinAlgError` is raised during SVD, THE `estimate_attitude` function SHALL return `AttitudeEstimate(is_valid=False, residual_deg=nan, attitude_confidence=0.0)` with the identity quaternion and identity rotation matrix.

---

### Requirement 12: AttitudeEstimate Output Structure

**User Story:** As a spacecraft systems engineer, I want the attitude estimate to be returned in a single structured dataclass that is explicit about units and validity, so that all downstream consumers have a consistent, unambiguous interface.

#### Acceptance Criteria

1. THE `AttitudeEstimate` dataclass SHALL contain the following fields with the specified types and semantics:
   - `quaternion`: `np.ndarray` shape `(4,)`, unit quaternion `[qw, qx, qy, qz]`, rotation from camera frame to inertial frame
   - `rotation_matrix`: `np.ndarray` shape `(3, 3)`, equivalent rotation matrix (camera to inertial, `det = +1`)
   - `euler_angles_deg`: `np.ndarray` shape `(3,)`, `[yaw, pitch, roll]` in degrees (ZYX convention)
   - `residual_deg`: `float`, mean angular residual of the fit in degrees
   - `num_correspondences`: `int`, number of star correspondences used
   - `is_valid`: `bool`, `True` when quality thresholds are met
   - `attitude_confidence`: `float` in `[0.0, 1.0]`
2. THE system SHALL NEVER return a silently invalid attitude: WHEN `is_valid = False`, THE `AttitudeEstimate.is_valid` field SHALL be explicitly `False` and the caller SHALL NOT be expected to infer invalidity from other fields.
3. THE default `AttitudeEstimate` (used for failure cases) SHALL have `quaternion = [1, 0, 0, 0]`, `rotation_matrix = I_3`, `euler_angles_deg = [0, 0, 0]`, `residual_deg = nan`, `num_correspondences = 0`, `is_valid = False`, `attitude_confidence = 0.0`.
4. WHEN `num_correspondences < config['navigation']['min_correspondences']`, THE `estimate_attitude` function SHALL return the default `AttitudeEstimate` with `is_valid = False` immediately, without performing SVD.

---

### Requirement 13: Attitude Error Metrics

**User Story:** As a navigation engineer, I want standard angular error statistics computed against ground truth, so that SIH 2026 evaluators can compare system performance against quantitative benchmarks.

#### Acceptance Criteria

1. THE `angular_error_deg(R1, R2)` function SHALL compute the geodesic rotation distance on SO(3) as `degrees(arccos(clamp((trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)))`, returning a value in `[0°, 180°]`.
2. WHEN evaluated on a synthetic star field with known ground-truth rotation, THE `estimate_attitude` function SHALL produce a rotation matrix whose `angular_error_deg` from ground truth is less than `0.1` degrees when at least 3 noise-free correspondences are provided.
3. THE benchmark evaluation script SHALL compute the following statistics over a test set of recognition results: mean residual (degrees), median residual (degrees), RMSE (degrees), and P95 residual (degrees).
4. ALL angular error statistics SHALL be computed only from images where `AttitudeEstimate.is_valid == True`; invalid estimates SHALL be counted separately and reported as a failure rate.
5. THE `estimate_attitude_weighted(observed_directions, catalog_directions, weights, config)` convenience wrapper SHALL call `estimate_attitude` with the provided `weights` array and return the same `AttitudeEstimate` structure.

---

### Requirement 14: Confidence-Weighted Attitude Estimation

**User Story:** As a navigation engineer, I want per-star confidence scores from the recognition step to be used as weights in attitude estimation, so that high-quality correspondences contribute more to the rotation solution than uncertain ones.

#### Acceptance Criteria

1. THE `run_navigation` pipeline SHALL extract `IdentifiedStar.confidence` from each inlier star in `RecognitionOutput.identified_stars` and pass these as the `weights` array to `estimate_attitude`.
2. THE `_estimate_attitude_from_recognition(rec_output, config)` helper SHALL assemble `obs_vecs = array([s.observed_unit_vec for s in identified_stars])`, `cat_vecs = array([s.catalog_unit_vec for s in identified_stars])`, and `weights = array([max(s.confidence, 1e-6) for s in identified_stars])`.
3. WHEN all inlier star confidences are equal, THE confidence-weighted attitude estimate SHALL produce the same rotation as uniform-weight Wahba/SVD to within floating-point precision.
4. ALL weights passed to `estimate_attitude` SHALL be non-negative; weights less than `1e-6` SHALL be clamped to `1e-6` to prevent numerical issues from near-zero weights.

---

### Requirement 15: Position Estimation Architecture

**User Story:** As a spacecraft systems engineer, I want the position estimator to correctly document the physical impossibility of single-image absolute position determination, so that the system architecture is honest and the design supports future multi-image or orbital extension.

#### Acceptance Criteria

1. THE `estimate_position(attitude_estimate, catalog_match_metadata, config)` function SHALL return a `PositionEstimate` with `is_valid = False` for single-image inputs, because star directions are angular measurements that cannot determine absolute position in 3D space.
2. THE `PositionEstimate` dataclass SHALL carry: `position_vector` (`np.ndarray` shape `(3,)`, all `NaN` for single-image case), `uncertainty` (`np.ndarray` shape `(3,3)`, all `NaN`), `is_valid` (`bool`), `method` (`str`), and `notes` (`str`).
3. THE `notes` field of the returned `PositionEstimate` SHALL contain a human-readable explanation stating that position estimation requires multi-image triangulation, orbital mechanics, or additional sensors (IMU, GPS, planetary limb).
4. THE `NavigationResult` dataclass SHALL include a `position_note: str` field carrying the same explanation, so that the navigation output is self-documenting even when no `PositionEstimate` object is returned directly.
5. WHERE multi-image data or an orbital propagator is available, THE `estimate_position` function architecture SHALL be designed to accept an extended `catalog_match_metadata` dict containing additional ephemeris or multi-image correspondence data without requiring changes to the function signature.

---

### Requirement 16: Navigation Pipeline Integration

**User Story:** As a navigation engineer, I want the complete navigation pipeline orchestrated in a single function call, so that all processing steps from image input to attitude output are consistently applied in the correct order.

#### Acceptance Criteria

1. THE `run_navigation(image, config, catalog_index, neural_model)` function SHALL execute the following stages in sequence: (1) star detection, (2) feature extraction and neural inference (if `neural_model` is not `None` and at least 2 stars detected), (3) pattern building, (4) pattern recognition, (5) attitude estimation.
2. THE `run_navigation` function SHALL time each stage using `time.perf_counter()` and record results in the corresponding `_time_ms` fields of `NavigationResult`: `detection_time_ms`, `feature_extraction_time_ms`, `recognition_time_ms`, `attitude_time_ms`, and `total_time_ms`.
3. THE `NavigationResult` dataclass SHALL carry all attitude fields from `AttitudeEstimate` (`quaternion`, `rotation_matrix`, `euler_angles_deg`, `attitude_confidence`, `attitude_residual_deg`), all star count fields (`n_observed_stars`, `n_matched_stars`, `n_inlier_stars`), `identified_stars`, `status`, `position_note`, `timestamp`, `total_time_ms`, and `error_message`.
4. THE `status` field of `NavigationResult` SHALL be a string matching the `RecognitionStatus.value` of the recognition output: `"SUCCESS"`, `"PARTIAL"`, `"LOW_CONFIDENCE"`, or `"FAILURE"`.
5. IF any exception is raised during pipeline execution, THEN THE `run_navigation` function SHALL catch the exception and return `NavigationResult(status="ERROR", error_message=str(exception), total_time_ms=<elapsed_ms>)` without propagating the exception.
6. THE `run_full_pipeline(image_path_or_array, config, catalog_index, neural_model)` function SHALL additionally apply image preprocessing (background subtraction, noise reduction, normalization from `config['preprocessing']`) before calling `run_navigation`, and SHALL record `preprocessing_time_ms` in the result.

---

### Requirement 17: Component-Level Pipeline Timing

**User Story:** As a performance engineer, I want every pipeline stage timed independently using `time.perf_counter()`, so that bottlenecks can be identified and optimization efforts directed at the highest-impact components.

#### Acceptance Criteria

1. THE `OptimizedPipeline.process_image(image)` method SHALL record separate timing measurements in milliseconds for: star detection, feature extraction, pattern recognition (including pattern building), attitude estimation, and total end-to-end latency.
2. THE `run_navigation` function SHALL measure stage latencies using `time.perf_counter()` at the start and end of each stage, computing elapsed time as `(t_end - t_start) * 1000.0` milliseconds.
3. THE timing dict returned by `OptimizedPipeline.process_image()` SHALL have keys: `detection_ms`, `feature_ms`, `recognition_ms`, `attitude_ms`, and `total_ms`, all with `float` values.
4. ALL timing values SHALL be sourced from actual `time.perf_counter()` measurements; no timing values SHALL be hardcoded, estimated, or fabricated.
5. THE `BenchmarkResult` dataclass SHALL include a `component_times_ms: dict` field containing the mean per-stage latency in milliseconds over all benchmark images, with the same key names as the timing dict.

---

### Requirement 18: Memory Profiling

**User Story:** As a performance engineer, I want peak memory usage measured using `tracemalloc`, so that the pipeline's memory footprint can be reported accurately for resource-constrained deployment scenarios.

#### Acceptance Criteria

1. THE `OptimizedPipeline.benchmark(images, n_warmup)` method SHALL measure memory usage using `tracemalloc.start()` before and `tracemalloc.stop()` after the benchmark run, reporting the incremental peak allocation in megabytes.
2. THE `BenchmarkResult.peak_memory_mb` field SHALL contain the peak memory allocated during the benchmark run, computed from `tracemalloc` snapshot comparison as `sum(stat.size_diff for stat in stats if stat.size_diff > 0) / (1024 * 1024)`.
3. IF `tracemalloc` measurement fails for any reason, THEN THE `OptimizedPipeline.benchmark` method SHALL set `peak_memory_mb = 0.0` and continue without propagating the exception.
4. THE memory measurement SHALL report current allocation in megabytes in addition to peak allocation, and both values SHALL be included in any printed benchmark report.
5. ALL memory values in outputs and reports SHALL be labeled with the unit `MB` (megabytes) to avoid ambiguity with kilobytes or bytes.

---

### Requirement 19: Latency Statistics

**User Story:** As a performance engineer, I want mean, median, P95, P99 latency, and FPS statistics computed over the benchmark run, so that tail-latency behavior (not just average performance) is captured for real-time suitability assessment.

#### Acceptance Criteria

1. THE `OptimizedPipeline.benchmark(images, n_warmup)` method SHALL run `n_warmup` warmup images before the timed benchmark, excluding warmup results from all statistics.
2. THE `BenchmarkResult` dataclass SHALL carry `mean_latency_ms`, `median_latency_ms`, `p95_latency_ms`, `p99_latency_ms` (all float, milliseconds), and `fps` (float, frames per second).
3. THE `fps` SHALL be computed as `1000.0 / mean_latency_ms`; WHEN `mean_latency_ms` is zero, THE `fps` SHALL be reported as `0.0`.
4. THE P95 and P99 latencies SHALL be computed using `numpy.percentile(latencies_array, 95)` and `numpy.percentile(latencies_array, 99)` respectively.
5. ALL latency statistics SHALL be computed from actual measured latencies returned by `process_image()`; no latency values SHALL be synthetic, interpolated, or fabricated.

---

### Requirement 20: Catalog Matching Optimization and Baseline Comparison

**User Story:** As a performance engineer, I want a measured comparison between the KD-tree-indexed pipeline and a naive baseline that rebuilds the catalog index per image, so that the optimization benefit is empirically demonstrated rather than claimed.

#### Acceptance Criteria

1. THE `OptimizedPipeline` SHALL load and index the star catalog exactly once at construction time and reuse the `CatalogIndex` for all subsequent `process_image()` calls, eliminating the O(N log N) KD-tree build cost per image.
2. THE `compare_baseline_vs_optimized(config, catalog_path, images)` function SHALL measure the "baseline" by rebuilding the `CatalogIndex` from scratch for every image in the input list, using the same recognition and attitude logic as the optimized version.
3. THE `compare_baseline_vs_optimized` function SHALL return a dict with keys `'optimized'`, `'baseline'`, `'speedup_ratio'`, and `'accuracy_comparison'`, where `speedup_ratio = baseline_mean_latency_ms / optimized_mean_latency_ms`.
4. ALL latency and accuracy values in the comparison output SHALL be computed from actual algorithm execution on the provided images; no values SHALL be fabricated or hardcoded.
5. THE `CatalogIndex.__init__` method SHALL use vectorized `numpy` operations for the pairwise angle computation where feasible, replacing per-pair Python loops with matrix dot products for the N×N dot-product matrix computation.

---

### Requirement 21: Recognition and Attitude Accuracy Evaluation

**User Story:** As a navigation engineer, I want robustness metrics computed against synthetic test images with known ground truth, so that the system's accuracy under noise, missing stars, and false detections can be reported to SIH 2026 evaluators.

#### Acceptance Criteria

1. THE evaluation script `benchmark.py` SHALL generate synthetic test images using `StarFieldGenerator` with at least the following conditions: clean images (noise-free), noisy images (with standard read noise from config), images with 1 false detection inserted, and images with a random star randomly removed.
2. FOR each test condition, THE evaluation script SHALL report: recognition accuracy (fraction with `SUCCESS` or `PARTIAL` status), mean angular attitude residual (degrees), P95 angular attitude residual (degrees), and failure rate.
3. THE benchmark evaluation SHALL report the number of images where `angular_error_deg(R_estimated, R_ground_truth) < 0.1 degrees` as the "high-precision" success count.
4. ALL accuracy numbers in `benchmark.py` output SHALL be computed from running the actual `run_navigation` function on the generated test images; no accuracy numbers SHALL be hardcoded or fabricated.
5. THE evaluation SHALL include a test where one intentional false detection (a pixel blob not corresponding to any catalog star) is inserted into the detected star list, and the RANSAC step SHALL still produce a valid attitude estimate from the remaining true detections.

---

### Requirement 22: Runnable Pipeline and Benchmark Scripts

**User Story:** As a SIH 2026 evaluator, I want two runnable command-line scripts — `run_pipeline.py` and `benchmark.py` — so that I can reproduce all demonstration results without modifying source code.

#### Acceptance Criteria

1. THE `run_pipeline.py` script SHALL accept a catalog path and generate one synthetic star field image, run the complete `run_full_pipeline()` function, and print a structured result including: status, number of observed stars, number of inlier stars, attitude quaternion `[qw, qx, qy, qz]`, Euler angles `[yaw, pitch, roll]` in degrees, mean residual in degrees, `is_valid` flag, and total latency in milliseconds.
2. THE `benchmark.py` script SHALL generate a configurable number of synthetic test images (default 20), run `OptimizedPipeline.benchmark()`, and print a table showing: N images, mean latency (ms), median latency (ms), P95 latency (ms), FPS, peak memory (MB), recognition accuracy, and mean attitude residual (degrees).
3. THE `benchmark.py` script SHALL also run `compare_baseline_vs_optimized()` and print the speedup ratio between the baseline and optimized pipeline.
4. BOTH scripts SHALL load `config.yaml` using `yaml.safe_load` and SHALL NOT hardcode any path or parameter that is defined in `config.yaml`.
5. BOTH scripts SHALL exit with return code `0` on success and a non-zero return code when a fatal error occurs (catalog not found, model not found, etc.).
6. THE scripts SHALL be located at the repository root level (not inside `src/`) for direct execution with `python run_pipeline.py` and `python benchmark.py`.

---

### Requirement 23: Visualization Functions

**User Story:** As a SIH 2026 evaluator, I want matplotlib-based visualization functions for star fields, detection overlays, confidence distributions, and attitude residuals, so that pipeline results can be inspected visually in notebooks and the Streamlit app.

#### Acceptance Criteria

1. THE `plot_star_field(image, title, cmap)` function in `src/utils/visualization.py` SHALL render the 2D float32 image array using `matplotlib.pyplot.imshow` with the specified colormap and title, and SHALL return a `matplotlib.figure.Figure` object.
2. THE `plot_detections(image, star_positions, title)` function SHALL overlay circular markers at each `(x, y)` position in `star_positions` (shape N×2) on the star-field image, using a contrasting color (default red), and SHALL return a `matplotlib.figure.Figure` object.
3. THE `plot_confidence_distribution(confidences, title)` function SHALL render a histogram of the confidence values array using 20 equal-width bins in `[0.0, 1.0]`, label the x-axis "Confidence" and y-axis "Count", and SHALL return a `matplotlib.figure.Figure` object.
4. THE `plot_attitude_residuals(residuals_deg, title)` function SHALL render a histogram of the angular residual values array, label the x-axis "Residual (degrees)" and y-axis "Count", mark the mean and P95 residual with vertical lines, and SHALL return a `matplotlib.figure.Figure` object.
5. ALL visualization functions SHALL accept only `numpy.ndarray` inputs for array parameters; they SHALL NOT accept Python lists directly (the caller is responsible for converting to ndarray).
6. ALL visualization functions SHALL NOT call `plt.show()` internally; callers (notebooks, Streamlit) SHALL be responsible for display, ensuring that visualization functions are usable in non-interactive environments.
7. WHEN the input array to any visualization function has zero elements, THE function SHALL return a `matplotlib.figure.Figure` with an empty plot and the specified title rather than raising an exception.

---

### Requirement 24: Test Coverage for SIH 2026 Quality Bar

**User Story:** As a SIH 2026 evaluator, I want specific automated tests to cover the key quality criteria, so that the project can be objectively verified against the hackathon evaluation rubric.

#### Acceptance Criteria

1. THE test suite SHALL include a test that constructs a known rotation matrix `R_true`, generates synthetic correspondences `cat_i = R_true @ obs_i` from at least 3 noise-free unit vectors, calls `estimate_attitude`, and asserts that `angular_error_deg(R_estimated, R_true) < 0.1`.
2. THE test suite SHALL include a test that runs `run_recognition` on a pattern with one deliberately incorrect correspondence (a false detection), verifies that the false detection is classified as a RANSAC outlier, and asserts that `RecognitionOutput.status` is `SUCCESS` or `PARTIAL` based on the remaining true correspondences.
3. THE test suite SHALL include a test that calls `run_navigation` with a zero-star image (all-black float32 array) and asserts that the returned `NavigationResult.status` is `"FAILURE"` and no exception is raised.
4. THE test suite SHALL include a test that calls `estimate_attitude` with fewer than `min_correspondences` stars and asserts that `AttitudeEstimate.is_valid == False`.
5. THE test suite SHALL include a test that verifies `RecognitionOutput.is_valid` is never implicitly `True` when `RecognitionOutput.status == RecognitionStatus.FAILURE`.
6. THE test suite SHALL include a test that runs `OptimizedPipeline.benchmark()` on at least 5 synthetic images and asserts that all `BenchmarkResult` latency fields are positive finite floats and `fps > 0`.
7. ALL angular error values in test assertions SHALL be expressed in degrees with the unit explicitly stated in the assertion message or comment.

---

### Requirement 25: Units and Labeling

**User Story:** As a spacecraft systems engineer and SIH 2026 evaluator, I want all physical quantities to be explicitly labeled with their units in code, dataclass docstrings, and printed outputs, so that there is no ambiguity in any intermediate or final value.

#### Acceptance Criteria

1. ALL angle-valued fields and variables in `AttitudeEstimate`, `IdentifiedStar`, `MatchedPattern`, `RecognitionOutput`, `NavigationResult`, and `BenchmarkResult` SHALL carry the unit in their field name or docstring: fields ending in `_deg` are in degrees, fields ending in `_rad` are in radians.
2. ALL time-valued fields SHALL carry the unit in their field name: fields ending in `_ms` are in milliseconds, fields ending in `_sec` are in seconds.
3. ALL memory-valued fields SHALL carry the unit in their field name: fields ending in `_mb` are in megabytes.
4. THE `run_pipeline.py` and `benchmark.py` scripts SHALL label all printed numeric outputs with their unit: e.g., `"Residual: 0.023 deg"`, `"Latency: 45.2 ms"`, `"Memory: 12.4 MB"`.
5. THE `AttitudeEstimate` docstring SHALL state the rotation convention: "R maps camera-frame vectors to inertial-frame vectors: v_inertial = R @ v_camera".
6. THE `NavigationResult` docstring SHALL state the quaternion convention: "[qw, qx, qy, qz], rotation from camera frame to inertial frame".
