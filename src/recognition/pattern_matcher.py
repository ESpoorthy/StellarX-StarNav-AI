"""
pattern_matcher.py — Phase 4
=============================
Hybrid neural+geometric star identification with RANSAC outlier rejection.

Algorithm:
1. Vote matrix: for each observed pair (i,j), find catalog pairs with
   matching angular separation (within angle_tolerance_deg). Vote for
   (obs_i → cat_k) and (obs_j → cat_l) for all matching catalog pairs.
2. Greedy correspondence assignment: for each observed star (sorted by vote
   strength), assign the catalog star with most votes (no double-use).
3. RANSAC: iterate over minimal 2-star subsets from correspondences.
   For each, compute rotation via TRIAD. Count inliers (residual < threshold).
   Keep best rotation and inlier set.
4. Refine: refit rotation using all inliers via Wahba/SVD.
5. Score: confidence = 0.5*(inlier_frac) + 0.4*(residual_quality) + 0.1*(neural_bonus)
6. Status:
   SUCCESS: n_inliers >= min_inliers AND residual <= max_residual AND conf >= conf_success
   PARTIAL: n_inliers >= 2 AND conf >= conf_partial
   LOW_CONFIDENCE: n_inliers >= 1
   FAILURE: otherwise
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.recognition.catalog_index import CatalogIndex, IndexedCatalogStar
from src.recognition.pattern_builder import StarPattern


# ---------------------------------------------------------------------------
# Enums and Data structures
# ---------------------------------------------------------------------------


class RecognitionStatus(Enum):
    """Status of a star pattern recognition attempt."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FAILURE = "FAILURE"


@dataclass
class IdentifiedStar:
    """A detected star matched to a catalog entry.

    Attributes
    ----------
    observed_x, observed_y : float
        Pixel coordinates of the observed star.
    observed_unit_vec : np.ndarray
        Camera-frame unit vector.
    catalog_id : str
        Hipparcos catalog identifier.
    catalog_ra_deg, catalog_dec_deg : float
        Catalog position in degrees.
    catalog_unit_vec : np.ndarray
        Inertial-frame unit vector from catalog.
    angular_residual_deg : float
        Angular distance between observed and predicted directions.
    confidence : float
        Per-star confidence score.
    brightness : float
        Observed brightness value.
    """

    observed_x: float = 0.0
    observed_y: float = 0.0
    observed_unit_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    catalog_id: str = ""
    catalog_ra_deg: float = 0.0
    catalog_dec_deg: float = 0.0
    catalog_unit_vec: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    angular_residual_deg: float = 0.0
    confidence: float = 0.0
    brightness: float = 0.0


@dataclass
class MatchedPattern:
    """Summary of the geometric pattern match.

    Attributes
    ----------
    pattern_type : str
        Description of match type (e.g. "geometric_ransac").
    candidate_count : int
        Number of candidates considered.
    inlier_count : int
        Number of inlier correspondences after RANSAC.
    total_stars : int
        Total observed stars.
    geometric_residual_deg : float
        Mean angular residual over inliers.
    confidence : float
        Overall match confidence in [0, 1].
    """

    pattern_type: str = "geometric_ransac"
    candidate_count: int = 0
    inlier_count: int = 0
    total_stars: int = 0
    geometric_residual_deg: float = float("nan")
    confidence: float = 0.0


@dataclass
class RecognitionOutput:
    """Complete output from the recognition pipeline.

    Attributes
    ----------
    identified_stars : list[IdentifiedStar]
        Stars successfully matched to catalog entries.
    matched_pattern : MatchedPattern or None
        Pattern match summary (None on FAILURE).
    status : RecognitionStatus
        Overall recognition status.
    processing_time_ms : float
        Total processing time in milliseconds.
    n_observed : int
        Number of observed (detected) stars.
    n_matched : int
        Number of stars with a tentative catalog match.
    n_inliers : int
        Number of RANSAC inlier correspondences.
    confidence : float
        Overall confidence score.
    mean_residual_deg : float
        Mean angular residual over inlier stars.
    neural_pattern_id : str or None
        Pattern ID from neural prior (if provided).
    neural_confidence : float
        Confidence of the neural prior.
    """

    identified_stars: list[IdentifiedStar] = field(default_factory=list)
    matched_pattern: Optional[MatchedPattern] = None
    status: RecognitionStatus = RecognitionStatus.FAILURE
    processing_time_ms: float = 0.0
    n_observed: int = 0
    n_matched: int = 0
    n_inliers: int = 0
    confidence: float = 0.0
    mean_residual_deg: float = float("nan")
    neural_pattern_id: Optional[str] = None
    neural_confidence: float = 0.0

    def is_successful(self) -> bool:
        """Return True if status is SUCCESS."""
        return self.status == RecognitionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_recognition(
    pattern: StarPattern,
    catalog_index: CatalogIndex,
    config: dict,
    neural_result=None,
) -> RecognitionOutput:
    """Run the complete star pattern recognition pipeline.

    Parameters
    ----------
    pattern : StarPattern
        Angular pattern built from detected stars.
    catalog_index : CatalogIndex
        Indexed star catalog for matching.
    config : dict
        Project configuration dict. Reads from config['recognition'].
    neural_result : RecognitionResult, optional
        Neural network output for prior bias (optional).

    Returns
    -------
    RecognitionOutput
        Complete recognition result including identified stars and status.
    """
    t_start = time.perf_counter()

    # Extract recognition config
    rec_cfg = config.get("recognition", {})
    angle_tol = float(rec_cfg.get("angle_tolerance_deg", 0.5))
    min_inliers = int(rec_cfg.get("min_inliers", 3))
    conf_success = float(rec_cfg.get("confidence_success", 0.6))
    conf_partial = float(rec_cfg.get("confidence_partial", 0.3))
    max_residual = float(rec_cfg.get("max_residual_deg", 1.0))
    ransac_iters = int(rec_cfg.get("ransac_iterations", 50))

    # Neural prior extraction
    neural_pattern_id = None
    neural_confidence = 0.0
    if neural_result is not None:
        neural_pattern_id = getattr(neural_result, "pattern_id", None)
        neural_confidence = float(getattr(neural_result, "confidence", 0.0))

    # Handle degenerate cases
    if pattern.n_stars < 2:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return RecognitionOutput(
            status=RecognitionStatus.FAILURE,
            n_observed=pattern.n_stars,
            processing_time_ms=elapsed_ms,
            neural_pattern_id=neural_pattern_id,
            neural_confidence=neural_confidence,
        )

    # Step 1: Vote accumulation
    # votes[obs_i][cat_j] = number of pair-matches supporting obs_i → cat_j
    n_obs = pattern.n_stars
    n_cat = len(catalog_index)
    votes = np.zeros((n_obs, n_cat), dtype=np.int32)

    for obs_i in range(n_obs):
        for obs_j in range(obs_i + 1, n_obs):
            obs_angle = pattern.pairwise_angles_deg[obs_i, obs_j]
            if obs_angle < 0.01:
                continue
            matching_pairs = catalog_index.find_pairs_by_angle(obs_angle, angle_tol)
            for cat_k, cat_l, _ in matching_pairs:
                votes[obs_i, cat_k] += 1
                votes[obs_j, cat_l] += 1
                votes[obs_i, cat_l] += 1
                votes[obs_j, cat_k] += 1

    # Step 2: Greedy correspondence assignment
    # For each observed star, pick the highest-voted catalog star (no reuse)
    correspondences: list[tuple[int, int]] = []  # (obs_idx, cat_idx)
    used_cat: set[int] = set()

    # Sort observed stars by max vote count (most certain first)
    obs_order = sorted(range(n_obs), key=lambda i: votes[i].max(), reverse=True)

    for obs_i in obs_order:
        if votes[obs_i].max() == 0:
            continue
        # Sort catalog candidates by vote count
        sorted_cats = np.argsort(votes[obs_i])[::-1]
        for cat_k in sorted_cats:
            if cat_k not in used_cat:
                correspondences.append((obs_i, cat_k))
                used_cat.add(cat_k)
                break

    n_matched = len(correspondences)

    if n_matched < 2:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return RecognitionOutput(
            status=RecognitionStatus.FAILURE if n_matched == 0 else RecognitionStatus.LOW_CONFIDENCE,
            n_observed=n_obs,
            n_matched=n_matched,
            n_inliers=min(n_matched, 1),
            processing_time_ms=elapsed_ms,
            neural_pattern_id=neural_pattern_id,
            neural_confidence=neural_confidence,
        )

    # Gather observed and catalog unit vectors for correspondences
    obs_vecs = np.array([pattern.unit_vectors[i] for i, _ in correspondences])
    cat_vecs = np.array([catalog_index.get_by_catalog_index(j).unit_vec for _, j in correspondences])

    # Step 3: RANSAC to find best rotation
    best_R = None
    best_inliers: list[int] = []
    n_corr = len(correspondences)

    for _ in range(ransac_iters):
        # Pick 2 random correspondences as hypothesis
        if n_corr < 2:
            break
        idx = np.random.choice(n_corr, size=2, replace=False)
        i0, i1 = int(idx[0]), int(idx[1])

        R_hyp = _triad_rotation(
            obs_vecs[i0], obs_vecs[i1],
            cat_vecs[i0], cat_vecs[i1],
        )
        if R_hyp is None:
            continue

        # Count inliers
        inliers = []
        for k in range(n_corr):
            pred = R_hyp @ obs_vecs[k]
            pred = pred / max(np.linalg.norm(pred), 1e-12)
            dot = float(np.dot(pred, cat_vecs[k]))
            dot = max(-1.0, min(1.0, dot))
            residual = math.degrees(math.acos(dot))
            if residual <= max_residual:
                inliers.append(k)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R_hyp

    n_inliers = len(best_inliers)

    # Step 4: Refine rotation using all inliers
    if n_inliers >= 2 and best_R is not None:
        inlier_obs = obs_vecs[best_inliers]
        inlier_cat = cat_vecs[best_inliers]
        weights = np.ones(n_inliers)
        refined_R = _wahba_svd(inlier_obs, inlier_cat, weights)
        if refined_R is not None:
            best_R = refined_R

    # Compute residuals for inlier correspondences
    residuals: list[float] = []
    identified_stars: list[IdentifiedStar] = []

    if best_R is not None and n_inliers > 0:
        for k in best_inliers:
            obs_idx, cat_idx = correspondences[k]
            pred = best_R @ obs_vecs[k]
            pred = pred / max(np.linalg.norm(pred), 1e-12)
            dot = float(np.dot(pred, cat_vecs[k]))
            dot = max(-1.0, min(1.0, dot))
            residual = math.degrees(math.acos(dot))
            residuals.append(residual)

            cat_star_indexed = catalog_index.get_by_catalog_index(cat_idx)
            cat_star = cat_star_indexed.star

            identified_stars.append(IdentifiedStar(
                observed_x=float(pattern.pixel_coords[obs_idx, 0]),
                observed_y=float(pattern.pixel_coords[obs_idx, 1]),
                observed_unit_vec=pattern.unit_vectors[obs_idx].copy(),
                catalog_id=cat_star.star_id,
                catalog_ra_deg=cat_star.ra_deg,
                catalog_dec_deg=cat_star.dec_deg,
                catalog_unit_vec=cat_star_indexed.unit_vec.copy(),
                angular_residual_deg=residual,
                confidence=max(0.0, 1.0 - residual / max_residual),
                brightness=float(pattern.brightnesses[obs_idx]),
            ))

    mean_residual = float(np.mean(residuals)) if residuals else float("nan")

    # Step 5: Compute confidence score
    inlier_frac = n_inliers / max(n_matched, 1)

    if residuals and not math.isnan(mean_residual):
        residual_quality = max(0.0, 1.0 - mean_residual / max_residual)
    else:
        residual_quality = 0.0

    neural_bonus = neural_confidence if neural_confidence > 0 else 0.0

    confidence = (
        0.5 * inlier_frac
        + 0.4 * residual_quality
        + 0.1 * neural_bonus
    )

    # Step 6: Determine status
    residual_ok = (not math.isnan(mean_residual)) and (mean_residual <= max_residual)

    if (n_inliers >= min_inliers and residual_ok and confidence >= conf_success):
        status = RecognitionStatus.SUCCESS
    elif n_inliers >= 2 and confidence >= conf_partial:
        status = RecognitionStatus.PARTIAL
    elif n_inliers >= 1:
        status = RecognitionStatus.LOW_CONFIDENCE
    else:
        status = RecognitionStatus.FAILURE

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    matched_pattern = MatchedPattern(
        pattern_type="geometric_ransac",
        candidate_count=n_matched,
        inlier_count=n_inliers,
        total_stars=n_obs,
        geometric_residual_deg=mean_residual,
        confidence=confidence,
    )

    return RecognitionOutput(
        identified_stars=identified_stars,
        matched_pattern=matched_pattern,
        status=status,
        processing_time_ms=elapsed_ms,
        n_observed=n_obs,
        n_matched=n_matched,
        n_inliers=n_inliers,
        confidence=confidence,
        mean_residual_deg=mean_residual,
        neural_pattern_id=neural_pattern_id,
        neural_confidence=neural_confidence,
    )


# ---------------------------------------------------------------------------
# Internal algorithms
# ---------------------------------------------------------------------------


def _triad_rotation(
    obs1: np.ndarray,
    obs2: np.ndarray,
    ref1: np.ndarray,
    ref2: np.ndarray,
) -> Optional[np.ndarray]:
    """TRIAD algorithm: compute rotation R such that R@obs ≈ ref.

    Builds orthonormal triads from two vector pairs, then:
        R = M_ref @ M_obs.T

    SVD-corrects to ensure det(R) = +1 (proper rotation).

    Parameters
    ----------
    obs1, obs2 : np.ndarray
        Shape (3,) observed unit vectors (camera frame).
    ref1, ref2 : np.ndarray
        Shape (3,) reference unit vectors (inertial frame).

    Returns
    -------
    np.ndarray or None
        3x3 rotation matrix, or None if vectors are nearly parallel.
    """
    # Build obs triad
    o1 = _safe_normalize(obs1)
    o2 = _safe_normalize(obs2)
    if o1 is None or o2 is None:
        return None

    o12 = _safe_normalize(np.cross(o1, o2))
    if o12 is None:
        return None
    o3 = np.cross(o1, o12)

    M_obs = np.column_stack([o1, o12, o3])  # columns are the triad

    # Build ref triad
    r1 = _safe_normalize(ref1)
    r2 = _safe_normalize(ref2)
    if r1 is None or r2 is None:
        return None

    r12 = _safe_normalize(np.cross(r1, r2))
    if r12 is None:
        return None
    r3 = np.cross(r1, r12)

    M_ref = np.column_stack([r1, r12, r3])

    # R = M_ref @ M_obs.T
    R = M_ref @ M_obs.T

    # SVD-correct to ensure proper rotation (det = +1)
    U, S, Vt = np.linalg.svd(R)
    det = np.linalg.det(U @ Vt)
    R = U @ np.diag([1.0, 1.0, det]) @ Vt

    return R


def _wahba_svd(
    obs_vecs: np.ndarray,
    cat_vecs: np.ndarray,
    weights: np.ndarray,
) -> Optional[np.ndarray]:
    """Solve Wahba's problem via SVD to find optimal rotation.

    Minimises: sum_i w_i * ||ref_i - R @ obs_i||^2

    Algorithm:
        B = sum_i (w_i * ref_i @ obs_i^T)
        B = U S V^T
        R = U @ diag(1, 1, det(U @ V^T)) @ V^T

    Parameters
    ----------
    obs_vecs : np.ndarray
        Shape (N, 3) observed unit vectors (camera frame).
    cat_vecs : np.ndarray
        Shape (N, 3) catalog unit vectors (inertial frame).
    weights : np.ndarray
        Shape (N,) non-negative weights.

    Returns
    -------
    np.ndarray or None
        3x3 rotation matrix, or None if computation fails.
    """
    if len(obs_vecs) == 0 or len(cat_vecs) == 0:
        return None

    try:
        # Build attitude profile matrix B
        B = np.zeros((3, 3), dtype=np.float64)
        for i in range(len(obs_vecs)):
            B += weights[i] * np.outer(cat_vecs[i], obs_vecs[i])

        U, S, Vt = np.linalg.svd(B)
        det = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0, 1.0, det]) @ Vt
        return R
    except np.linalg.LinAlgError:
        return None


def _safe_normalize(
    v: np.ndarray,
    tol: float = 1e-10,
) -> Optional[np.ndarray]:
    """Normalize a vector, returning None if the norm is too small.

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    tol : float
        Minimum norm threshold below which None is returned.

    Returns
    -------
    np.ndarray or None
        Unit vector, or None if ||v|| < tol.
    """
    norm = float(np.linalg.norm(v))
    if norm < tol:
        return None
    return v / norm
