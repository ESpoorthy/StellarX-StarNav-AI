"""
ground_truth_validator.py
=========================
Synthetic ground-truth attitude validation.

Uses the existing star-field generator to produce images at known orientations,
runs the full navigation pipeline on each, and computes true angular attitude error
using the geodesic rotation-distance metric (not Euler-angle differences).

Scientific basis
----------------
True attitude error is computed as the geodesic distance on SO(3):

    θ = arccos( (trace(R_true^T @ R_est) - 1) / 2 )

This is the only correct metric for attitude error. Euler-angle differences
are NOT used because they depend on decomposition convention and suffer from
gimbal lock.

Reference: Markley & Crassidis (2014), §2.7 "Attitude Error Representation".
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class GroundTruthRun:
    """Result of one ground-truth validation run."""
    seed: int = 0
    boresight_ra_deg: float = 0.0
    boresight_dec_deg: float = 0.0
    roll_deg: float = 0.0
    n_gt_stars: int = 0
    n_detected: int = 0
    n_matched: int = 0
    n_inliers: int = 0

    # Estimated attitude
    estimated_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1., 0., 0., 0.]))
    estimated_euler_deg: np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    # True attitude (from generator)
    true_rotation_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3))

    # Error metrics
    attitude_error_deg: float = float("nan")   # geodesic SO(3) distance
    mean_residual_deg: float = float("nan")
    integrity_status: str = "REJECTED"
    pipeline_status: str = "FAILURE"
    latency_ms: float = 0.0


@dataclass
class ValidationReport:
    """Summary across multiple ground-truth runs."""
    n_runs: int = 0
    n_success: int = 0
    n_partial: int = 0
    n_failure: int = 0
    success_rate: float = 0.0

    # Attitude error statistics (over successful runs only)
    mean_error_deg: float = float("nan")
    median_error_deg: float = float("nan")
    p95_error_deg: float = float("nan")
    max_error_deg: float = float("nan")

    # Latency statistics
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0

    runs: list[GroundTruthRun] = field(default_factory=list)
    noise_level: float = 0.0
    note: str = ""


def run_ground_truth_validation(
    config: dict,
    n_runs: int = 10,
    seed_start: int = 1000,
    noise_level: float = 0.005,
    verbose: bool = True,
) -> ValidationReport:
    """Run ground-truth attitude validation.

    Generates synthetic star fields at known orientations, runs the full
    navigation pipeline, and computes true geodesic attitude error.

    Parameters
    ----------
    config : dict
        Full project configuration dict.
    n_runs : int
        Number of validation runs.
    seed_start : int
        Starting random seed (seeds = seed_start, seed_start+1, …).
    noise_level : float
        Gaussian read noise sigma applied to synthetic images.
    verbose : bool
        Print progress.

    Returns
    -------
    ValidationReport
    """
    try:
        from src.catalog.catalog_loader import load_catalog
        from src.recognition.catalog_index import CatalogIndex
        from src.preprocessing.star_field_generator import StarFieldGenerator
        from src.navigation.navigator import run_navigation, _preprocess_image
        from src.navigation.attitude_estimator import (
            rotation_matrix_to_quaternion, angular_error_deg
        )
        from src.navigation.pipeline_result import build_pipeline_result
    except ImportError as exc:
        return ValidationReport(note=f"Import error: {exc}")

    cat_path = Path(config["dataset"]["catalog_file"])
    catalog  = load_catalog(str(cat_path), config=config)
    cidx     = CatalogIndex(catalog)

    gen_cfg = {**config["dataset"], "read_noise_sigma": noise_level,
               "artifact_probability": 0.0}
    gen = StarFieldGenerator(catalog, gen_cfg)

    runs: list[GroundTruthRun] = []

    for i in range(n_runs):
        seed = seed_start + i
        sf = gen.generate(seed=seed)

        # True rotation: from generator boresight (RA, Dec) + roll
        # Build the true rotation matrix from the gnomonic projection geometry
        true_R = _boresight_to_rotation(
            sf.boresight_ra_deg, sf.boresight_dec_deg, sf.roll_deg
        )

        t0 = time.perf_counter()
        preprocessed = _preprocess_image(sf.image, config)
        nav = run_navigation(preprocessed, config, cidx)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pr = build_pipeline_result(nav, config, is_demo=False)

        # Geodesic attitude error
        err_deg = float("nan")
        if nav.attitude_status in ("DETERMINED", "PARTIAL"):
            R_est = nav.rotation_matrix
            if np.isfinite(R_est).all():
                err_deg = angular_error_deg(true_R, R_est)

        run = GroundTruthRun(
            seed=seed,
            boresight_ra_deg=sf.boresight_ra_deg,
            boresight_dec_deg=sf.boresight_dec_deg,
            roll_deg=sf.roll_deg,
            n_gt_stars=len(sf.stars),
            n_detected=nav.n_observed_stars,
            n_matched=nav.n_matched_stars,
            n_inliers=nav.n_inlier_stars,
            estimated_quaternion=nav.quaternion.copy(),
            estimated_euler_deg=nav.euler_angles_deg.copy(),
            true_rotation_matrix=true_R,
            attitude_error_deg=err_deg,
            mean_residual_deg=nav.attitude_residual_deg,
            integrity_status=pr.integrity.status,
            pipeline_status=nav.status,
            latency_ms=latency_ms,
        )
        runs.append(run)

        if verbose:
            err_str = f"{err_deg:.3f}°" if math.isfinite(err_deg) else "—"
            print(f"  Run {i+1:2d}/{n_runs}: seed={seed}  "
                  f"gt={run.n_gt_stars}  det={run.n_detected}  "
                  f"inl={run.n_inliers}  err={err_str}  "
                  f"status={nav.status}  {latency_ms:.0f}ms")

    # Aggregate statistics
    n_success = sum(1 for r in runs if r.pipeline_status == "SUCCESS")
    n_partial  = sum(1 for r in runs if r.pipeline_status == "PARTIAL")
    n_failure  = len(runs) - n_success - n_partial

    valid_errors = [r.attitude_error_deg for r in runs
                    if math.isfinite(r.attitude_error_deg)]
    latencies    = [r.latency_ms for r in runs]

    report = ValidationReport(
        n_runs=len(runs),
        n_success=n_success,
        n_partial=n_partial,
        n_failure=n_failure,
        success_rate=(n_success + 0.5 * n_partial) / max(len(runs), 1),
        mean_error_deg=float(np.mean(valid_errors)) if valid_errors else float("nan"),
        median_error_deg=float(np.median(valid_errors)) if valid_errors else float("nan"),
        p95_error_deg=float(np.percentile(valid_errors, 95)) if valid_errors else float("nan"),
        max_error_deg=float(np.max(valid_errors)) if valid_errors else float("nan"),
        mean_latency_ms=float(np.mean(latencies)),
        median_latency_ms=float(np.median(latencies)),
        runs=runs,
        noise_level=noise_level,
        note="",
    )
    return report


def _boresight_to_rotation(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """Build the camera→inertial rotation matrix from boresight + roll.

    The boresight direction defines the camera +Z axis in inertial frame.
    Roll rotates around that axis.

    Convention matches star_field_generator.py gnomonic projection.
    """
    ra  = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    roll = math.radians(roll_deg)

    # Boresight unit vector (camera +Z in inertial frame)
    bz = np.array([
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    ])

    # Camera +X: east direction (∂bz/∂ra normalised)
    bx = np.array([-math.sin(ra), math.cos(ra), 0.0])
    # If bx is nearly zero (at poles), use a fixed reference
    if np.linalg.norm(bx) < 1e-9:
        bx = np.array([1.0, 0.0, 0.0])

    # Camera +Y = bz × bx
    by = np.cross(bz, bx)
    by = by / max(np.linalg.norm(by), 1e-12)
    bx = np.cross(by, bz)
    bx = bx / max(np.linalg.norm(bx), 1e-12)

    # Apply roll rotation around bz
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    bx_rolled = cos_r * bx - sin_r * by
    by_rolled = sin_r * bx + cos_r * by

    # R columns: camera X, Y, Z in inertial frame
    R = np.column_stack([bx_rolled, by_rolled, bz])
    return R
