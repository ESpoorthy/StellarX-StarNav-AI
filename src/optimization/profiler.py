"""
profiler.py — Phase 6
======================
Pipeline profiler: measures per-component latency and memory usage,
identifies bottlenecks, and produces a formatted breakdown report.

Usage
-----
    from src.optimization.profiler import PipelineProfiler
    profiler = PipelineProfiler(config, catalog_path)
    report = profiler.profile(images, n_runs=10)
    profiler.print_report(report)
"""

from __future__ import annotations

import math
import tracemalloc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.catalog.catalog_loader import load_catalog
from src.navigation.navigator import _preprocess_image, run_navigation
from src.recognition.catalog_index import CatalogIndex


@dataclass
class ProfileReport:
    """Per-component timing breakdown from the profiler.

    All times in milliseconds.
    """

    n_runs: int = 0

    # Per-component mean (ms)
    preprocessing_ms: float = 0.0
    detection_ms: float = 0.0
    feature_ms: float = 0.0
    recognition_ms: float = 0.0
    attitude_ms: float = 0.0
    total_ms: float = 0.0

    # Per-component P95 (ms)
    p95_preprocessing_ms: float = 0.0
    p95_detection_ms: float = 0.0
    p95_recognition_ms: float = 0.0
    p95_attitude_ms: float = 0.0
    p95_total_ms: float = 0.0

    # Memory (MB)
    peak_memory_mb: float = 0.0

    # Bottleneck (component with highest mean time)
    bottleneck_component: str = ""
    bottleneck_pct: float = 0.0

    # Per-component percentage of total
    component_pct: dict = field(default_factory=dict)


class PipelineProfiler:
    """Measures per-component latency of the navigation pipeline.

    Parameters
    ----------
    config : dict
        Project configuration dict.
    catalog_path : str or Path
        Path to the star catalog CSV.
    """

    def __init__(self, config: dict, catalog_path: str | Path) -> None:
        self._config = config
        catalog = load_catalog(catalog_path, config=config)
        self._catalog_index = CatalogIndex(catalog)

    def profile(
        self,
        images: list[np.ndarray],
        n_runs: int = 10,
        n_warmup: int = 2,
    ) -> ProfileReport:
        """Profile the pipeline on a set of images.

        Parameters
        ----------
        images : list[np.ndarray]  Images to profile.
        n_runs : int  Number of profiling runs.
        n_warmup : int  Warmup runs before timing.

        Returns
        -------
        ProfileReport
        """
        if not images:
            return ProfileReport()

        # Warmup
        for img in images[:n_warmup]:
            try:
                run_navigation(img, self._config, self._catalog_index)
            except Exception:
                pass

        # Profile each component separately
        detect_times, feat_times = [], []
        recog_times, attitude_times, total_times = [], [], []

        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        for i in range(n_runs):
            img = images[i % len(images)]

            # Full pipeline (captures accurate total + component times)
            from src.navigation.navigator import NavigationResult
            result = run_navigation(img, self._config, self._catalog_index)

            total_times.append(result.total_time_ms)
            detect_times.append(result.detection_time_ms)
            feat_times.append(result.feature_extraction_time_ms)
            recog_times.append(result.recognition_time_ms)
            attitude_times.append(result.attitude_time_ms)

        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        try:
            stats = snap_after.compare_to(snap_before, "lineno")
            peak_mb = sum(s.size_diff for s in stats if s.size_diff > 0) / (1024*1024)
        except Exception:
            peak_mb = 0.0

        def _mean(lst): return float(np.mean(lst)) if lst else 0.0
        def _p95(lst):  return float(np.percentile(lst, 95)) if lst else 0.0

        mean_total = _mean(total_times)
        comp_means = {
            "detection":   _mean(detect_times),
            "feature":     _mean(feat_times),
            "recognition": _mean(recog_times),
            "attitude":    _mean(attitude_times),
        }

        # Identify bottleneck
        bottleneck = max(comp_means, key=comp_means.get) if comp_means else ""
        bottleneck_pct = (comp_means.get(bottleneck, 0) / max(mean_total, 1e-9)) * 100

        # Component percentages
        comp_pct = {
            k: (v / max(mean_total, 1e-9)) * 100
            for k, v in comp_means.items()
        }

        return ProfileReport(
            n_runs=n_runs,
            preprocessing_ms=0.0,  # pre-preprocessed images
            detection_ms=_mean(detect_times),
            feature_ms=_mean(feat_times),
            recognition_ms=_mean(recog_times),
            attitude_ms=_mean(attitude_times),
            total_ms=mean_total,
            p95_preprocessing_ms=0.0,
            p95_detection_ms=_p95(detect_times),
            p95_recognition_ms=_p95(recog_times),
            p95_attitude_ms=_p95(attitude_times),
            p95_total_ms=_p95(total_times),
            peak_memory_mb=peak_mb,
            bottleneck_component=bottleneck,
            bottleneck_pct=bottleneck_pct,
            component_pct=comp_pct,
        )

    @staticmethod
    def print_report(report: ProfileReport) -> None:
        """Print a formatted profiling breakdown."""
        print("\n" + "="*60)
        print("  PIPELINE PERFORMANCE BREAKDOWN")
        print(f"  (n_runs={report.n_runs}, all times in ms)")
        print("="*60)
        print(f"  {'Component':<22} {'Mean (ms)':>10} {'P95 (ms)':>10} {'% Total':>8}")
        print("  " + "-"*54)

        rows = [
            ("Star detection",    report.detection_ms,   report.p95_detection_ms),
            ("Feature extraction",report.feature_ms,     0.0),
            ("Pattern recognition",report.recognition_ms, report.p95_recognition_ms),
            ("Attitude estimation",report.attitude_ms,    report.p95_attitude_ms),
        ]
        for name, mean, p95 in rows:
            pct = report.component_pct.get(
                name.lower().split()[0], 0.0
            )
            p95_str = f"{p95:.2f}" if p95 > 0 else "  N/A"
            print(f"  {name:<22} {mean:>10.2f} {p95_str:>10} {pct:>7.1f}%")

        print("  " + "-"*54)
        print(f"  {'TOTAL':<22} {report.total_ms:>10.1f} {report.p95_total_ms:>10.1f} {'100.0':>7}%")
        print()
        print(f"  Peak memory usage : {report.peak_memory_mb:.3f} MB")
        print(f"  Bottleneck        : {report.bottleneck_component} ({report.bottleneck_pct:.1f}% of total)")
        print("="*60)
