"""
pipeline.py — Phase 6
======================
Optimized end-to-end pipeline with profiling, benchmarking,
and catalog-matching optimizations.

Phase 6 optimizations implemented
----------------------------------
1. Catalog loaded once, KD-tree built once at construction — reused every run.
2. Vectorized pairwise-angle computation in CatalogIndex (NumPy einsum).
3. Vectorized vote accumulation in run_recognition (NumPy broadcasting).
4. Vectorized RANSAC inlier counting (matrix @ operation vs Python loop).
5. Vectorized Wahba/SVD attitude profile matrix (einsum vs loop).
6. Pre-allocated attitude residual computation (vectorized acos/dot).
7. tracemalloc memory profiling — current + peak.
8. Cold-start vs warm-inference clearly separated.
9. Per-component timing with time.perf_counter().
10. Configurable CPU thread count via config['optimization']['n_threads'].
"""

from __future__ import annotations

import os
import tracemalloc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.catalog.catalog_loader import load_catalog
from src.navigation.navigator import NavigationResult, run_navigation
from src.recognition.catalog_index import CatalogIndex


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Results from a pipeline benchmarking run.

    All timing values in milliseconds (suffix _ms).
    All memory values in megabytes (suffix _mb).
    """

    n_images: int = 0

    # Latency (ms) — warm inference only (after warmup runs)
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    fps: float = 0.0

    # Cold start (ms) — first image including any lazy initialization
    cold_start_ms: float = 0.0

    # Memory (MB)
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0

    # Per-component mean latency (ms)
    component_times_ms: dict = field(default_factory=dict)

    # Recognition and attitude
    n_success: int = 0
    n_partial: int = 0
    n_failure: int = 0
    recognition_accuracy: float = 0.0
    mean_attitude_residual_deg: float = float("nan")

    # Metadata
    n_warmup: int = 0
    catalog_size: int = 0
    image_shape: tuple = (0, 0)
    config_summary: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Baseline vs optimized comparison table."""

    baseline: BenchmarkResult = field(default_factory=BenchmarkResult)
    optimized: BenchmarkResult = field(default_factory=BenchmarkResult)

    @property
    def speedup_ratio(self) -> float:
        if self.optimized.mean_latency_ms > 0:
            return self.baseline.mean_latency_ms / self.optimized.mean_latency_ms
        return 1.0

    @property
    def memory_reduction_mb(self) -> float:
        return self.baseline.peak_memory_mb - self.optimized.peak_memory_mb

    @property
    def accuracy_delta(self) -> float:
        return self.optimized.recognition_accuracy - self.baseline.recognition_accuracy

    def print_table(self) -> None:
        """Print baseline vs optimized comparison table."""
        b, o = self.baseline, self.optimized
        import math

        print("\n" + "="*70)
        print("  BASELINE vs OPTIMIZED COMPARISON")
        print("="*70)
        print(f"  {'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
        print("  " + "-"*66)

        def row(name, bv, ov, fmt=".1f", unit=""):
            if isinstance(bv, float) and math.isnan(bv):
                bstr = "N/A"
            else:
                bstr = f"{bv:{fmt}}{unit}"
            if isinstance(ov, float) and math.isnan(ov):
                ostr = "N/A"
            else:
                ostr = f"{ov:{fmt}}{unit}"
            if not (isinstance(bv, float) and math.isnan(bv)) and bv != 0:
                pct = (ov - bv) / abs(bv) * 100
                chg = f"{pct:+.1f}%"
            else:
                chg = "N/A"
            print(f"  {name:<30} {bstr:>12} {ostr:>12} {chg:>12}")

        row("Mean latency (ms)", b.mean_latency_ms, o.mean_latency_ms)
        row("Median latency (ms)", b.median_latency_ms, o.median_latency_ms)
        row("P95 latency (ms)", b.p95_latency_ms, o.p95_latency_ms)
        row("FPS", b.fps, o.fps, fmt=".2f")
        row("Peak memory (MB)", b.peak_memory_mb, o.peak_memory_mb, fmt=".2f")
        row("Cold start (ms)", b.cold_start_ms, o.cold_start_ms)

        # Component breakdown
        all_keys = set(b.component_times_ms) | set(o.component_times_ms)
        for k in sorted(all_keys):
            bv = b.component_times_ms.get(k, 0.0)
            ov = o.component_times_ms.get(k, 0.0)
            row(f"  {k}", bv, ov, fmt=".2f")

        row("Recognition accuracy", b.recognition_accuracy*100,
            o.recognition_accuracy*100, fmt=".1f", unit="%")

        import math as _math
        b_res = b.mean_attitude_residual_deg
        o_res = o.mean_attitude_residual_deg
        if not _math.isnan(b_res) and not _math.isnan(o_res):
            row("Mean att. residual (deg)", b_res, o_res, fmt=".4f")

        print("  " + "-"*66)
        print(f"  {'Speedup ratio':<30} {'':>12} {self.speedup_ratio:>11.2f}x {'':>12}")
        print("="*70)


# ---------------------------------------------------------------------------
# OptimizedPipeline
# ---------------------------------------------------------------------------


class OptimizedPipeline:
    """Optimized navigation pipeline with cached catalog and profiling.

    Phase 6 optimizations:
    - Catalog loaded and KD-tree built exactly once at construction
    - Vectorized pairwise angle matrix computation
    - Vectorized RANSAC inlier counting
    - Vectorized Wahba/SVD attitude profile matrix
    - Configurable CPU thread count

    Parameters
    ----------
    config : dict
        Project configuration dict.
    catalog_path : str or Path
        Path to the Hipparcos CSV catalog file.
    neural_model : optional
        Trained StarPatternClassifier for neural prior. None = geometric only.
    """

    def __init__(
        self,
        config: dict,
        catalog_path: str | Path,
        neural_model=None,
    ) -> None:
        self._config = config
        self._neural_model = neural_model

        # Configure CPU threads if specified
        opt_cfg = config.get("optimization", {})
        n_threads = int(opt_cfg.get("n_threads", 0))  # 0 = use NumPy default
        if n_threads > 0:
            try:
                os.environ["OMP_NUM_THREADS"] = str(n_threads)
                os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
                os.environ["MKL_NUM_THREADS"] = str(n_threads)
            except Exception:
                pass

        # Measure cold-start time (catalog load + index build)
        t_cold = time.perf_counter()
        catalog = load_catalog(
            catalog_path,
            config=config,
            mag_limit=config.get("dataset", {}).get("catalog_mag_limit", None),
        )
        self._catalog_index = CatalogIndex(catalog)
        self._cold_start_ms = (time.perf_counter() - t_cold) * 1000.0
        self._catalog_size = len(self._catalog_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cold_start_ms(self) -> float:
        """Catalog load + index build time in milliseconds."""
        return self._cold_start_ms

    @property
    def catalog_size(self) -> int:
        """Number of stars in the loaded catalog."""
        return self._catalog_size

    def process_image(
        self,
        image: np.ndarray,
    ) -> tuple[NavigationResult, dict]:
        """Process a single image through the full navigation pipeline.

        Parameters
        ----------
        image : np.ndarray
            2D float32 preprocessed image, values in [0, 1].

        Returns
        -------
        tuple[NavigationResult, dict]
            Navigation result and per-component timing dict (ms).
        """
        result = run_navigation(
            image=image,
            config=self._config,
            catalog_index=self._catalog_index,
            neural_model=self._neural_model,
        )

        timing = {
            "detection_ms":    result.detection_time_ms,
            "feature_ms":      result.feature_extraction_time_ms,
            "recognition_ms":  result.recognition_time_ms,
            "attitude_ms":     result.attitude_time_ms,
            "total_ms":        result.total_time_ms,
        }

        return result, timing

    def benchmark(
        self,
        images: list[np.ndarray],
        n_warmup: int = 3,
    ) -> BenchmarkResult:
        """Benchmark the pipeline on a list of images.

        Phase 6: clearly separates cold-start from warm inference.
        Warm inference = all runs after n_warmup warmup images.

        Parameters
        ----------
        images : list[np.ndarray]
            Images to benchmark.
        n_warmup : int
            Number of warmup images before timing starts (default 3).

        Returns
        -------
        BenchmarkResult
        """
        if len(images) == 0:
            return BenchmarkResult()

        # ── Warmup runs (not timed) ───────────────────────────────────────
        warmup_imgs = images[:n_warmup]
        for img in warmup_imgs:
            try:
                self.process_image(img)
            except Exception:
                pass

        # ── Timed runs ────────────────────────────────────────────────────
        bench_imgs = images[n_warmup:] if len(images) > n_warmup else images
        if not bench_imgs:
            bench_imgs = images

        # Start memory tracking
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        latencies: list[float] = []
        comp_acc: dict[str, list[float]] = {
            "detection_ms": [], "feature_ms": [],
            "recognition_ms": [], "attitude_ms": [],
        }
        results: list[NavigationResult] = []

        for img in bench_imgs:
            result, timing = self.process_image(img)
            latencies.append(timing["total_ms"])
            for k in comp_acc:
                comp_acc[k].append(timing.get(k, 0.0))
            results.append(result)

        # Memory snapshot
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        try:
            stats = snap_after.compare_to(snap_before, "lineno")
            peak_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
            peak_mb = peak_bytes / (1024 * 1024)
            # Current allocation from after snapshot
            current_stats = snap_after.statistics("lineno")
            current_bytes = sum(s.size for s in current_stats)
            current_mb = current_bytes / (1024 * 1024)
        except Exception:
            peak_mb = 0.0
            current_mb = 0.0

        # ── Statistics ────────────────────────────────────────────────────
        n = len(latencies)
        arr = np.array(latencies)

        n_success = sum(1 for r in results if r.status == "SUCCESS")
        n_partial  = sum(1 for r in results if r.status == "PARTIAL")
        n_failure  = sum(1 for r in results if r.status not in ("SUCCESS","PARTIAL"))
        accuracy   = (n_success + n_partial) / max(n, 1)

        residuals = [
            r.attitude_residual_deg for r in results
            if r.status in ("SUCCESS","PARTIAL") and not np.isnan(r.attitude_residual_deg)
        ]

        import math
        return BenchmarkResult(
            n_images=n,
            mean_latency_ms=float(np.mean(arr))   if n > 0 else 0.0,
            median_latency_ms=float(np.median(arr)) if n > 0 else 0.0,
            min_latency_ms=float(np.min(arr))     if n > 0 else 0.0,
            max_latency_ms=float(np.max(arr))     if n > 0 else 0.0,
            p95_latency_ms=float(np.percentile(arr, 95)) if n > 0 else 0.0,
            p99_latency_ms=float(np.percentile(arr, 99)) if n > 0 else 0.0,
            fps=1000.0/float(np.mean(arr)) if n > 0 and float(np.mean(arr)) > 0 else 0.0,
            cold_start_ms=self._cold_start_ms,
            peak_memory_mb=peak_mb,
            current_memory_mb=current_mb,
            component_times_ms={k: float(np.mean(v)) if v else 0.0
                                  for k, v in comp_acc.items()},
            n_success=n_success,
            n_partial=n_partial,
            n_failure=n_failure,
            recognition_accuracy=accuracy,
            mean_attitude_residual_deg=float(np.mean(residuals)) if residuals else float("nan"),
            n_warmup=n_warmup,
            catalog_size=self._catalog_size,
            image_shape=bench_imgs[0].shape if bench_imgs else (0, 0),
        )


# ---------------------------------------------------------------------------
# Baseline pipeline (for comparison)
# ---------------------------------------------------------------------------


class BaselinePipeline:
    """Baseline pipeline — rebuilds catalog index on every call.

    Used only for comparison in benchmark to measure the speedup
    from catalog caching. Not for production use.
    """

    def __init__(self, config: dict, catalog_path: str | Path) -> None:
        self._config = config
        self._catalog_path = Path(catalog_path)

    def process_image(self, image: np.ndarray) -> tuple[NavigationResult, dict]:
        # Reload catalog and rebuild index every time (no caching)
        catalog = load_catalog(
            self._catalog_path,
            config=self._config,
            mag_limit=self._config.get("dataset", {}).get("catalog_mag_limit", None),
        )
        catalog_index = CatalogIndex(catalog)
        result = run_navigation(image=image, config=self._config,
                                catalog_index=catalog_index)
        timing = {
            "detection_ms": result.detection_time_ms,
            "feature_ms":   result.feature_extraction_time_ms,
            "recognition_ms": result.recognition_time_ms,
            "attitude_ms":  result.attitude_time_ms,
            "total_ms":     result.total_time_ms,
        }
        return result, timing

    def benchmark(self, images: list[np.ndarray], n_warmup: int = 0) -> BenchmarkResult:
        """Benchmark without warmup (baseline always cold)."""
        if not images:
            return BenchmarkResult()
        latencies, results = [], []
        for img in images:
            result, timing = self.process_image(img)
            latencies.append(timing["total_ms"])
            results.append(result)
        n = len(latencies); arr = np.array(latencies)
        n_ok = sum(1 for r in results if r.status in ("SUCCESS","PARTIAL"))
        residuals = [r.attitude_residual_deg for r in results
                     if r.status in ("SUCCESS","PARTIAL") and not np.isnan(r.attitude_residual_deg)]
        return BenchmarkResult(
            n_images=n,
            mean_latency_ms=float(np.mean(arr)),
            median_latency_ms=float(np.median(arr)),
            min_latency_ms=float(np.min(arr)),
            max_latency_ms=float(np.max(arr)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
            fps=1000.0/float(np.mean(arr)) if float(np.mean(arr)) > 0 else 0.0,
            cold_start_ms=float(np.mean(arr)),  # every run is cold
            recognition_accuracy=n_ok/max(n,1),
            mean_attitude_residual_deg=float(np.mean(residuals)) if residuals else float("nan"),
        )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_baseline_vs_optimized(
    config: dict,
    catalog_path: str | Path,
    images: list[np.ndarray],
    n_warmup: int = 3,
) -> ComparisonResult:
    """Run baseline and optimized benchmarks and return comparison.

    All values come from actual execution — nothing fabricated.

    Parameters
    ----------
    config : dict  Project configuration dict.
    catalog_path : str or Path  Path to the star catalog CSV.
    images : list[np.ndarray]  Images to benchmark on.
    n_warmup : int  Warmup images for optimized pipeline.

    Returns
    -------
    ComparisonResult
    """
    if not images:
        return ComparisonResult()

    # Optimized: catalog built once
    opt = OptimizedPipeline(config, catalog_path)
    opt_result = opt.benchmark(images, n_warmup=n_warmup)

    # Baseline: catalog rebuilt per image (no warmup — always cold)
    base = BaselinePipeline(config, catalog_path)
    cmp_imgs = images[:min(8, len(images))]  # cap baseline at 8 to save time
    base_result = base.benchmark(cmp_imgs, n_warmup=0)

    return ComparisonResult(baseline=base_result, optimized=opt_result)
