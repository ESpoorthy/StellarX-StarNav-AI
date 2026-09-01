#!/usr/bin/env python3
"""
benchmark.py — Phase 6
=======================
StellarX-StarNav-AI complete inference benchmark and optimization report.

Measures:
  - Cold start vs warm inference latency
  - Per-component latency breakdown
  - Peak memory usage
  - Recognition accuracy
  - Attitude estimation residual
  - Baseline vs optimized comparison
  - Edge deployment estimates

Usage:
    python benchmark.py [--n-images INT] [--n-warmup INT] [--config PATH]
                        [--catalog PATH] [--skip-baseline] [--edge]

All performance numbers come from actual execution — nothing fabricated.
Exit code 0 on success, 1 on fatal error.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def main() -> int:
    p = argparse.ArgumentParser(description="StellarX Phase 6 Benchmark")
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--catalog",      default=None)
    p.add_argument("--n-images",     type=int, default=20)
    p.add_argument("--n-warmup",     type=int, default=3)
    p.add_argument("--skip-baseline",action="store_true")
    p.add_argument("--edge",         action="store_true",
                   help="Show edge deployment estimates")
    p.add_argument("--seed",         type=int, default=30000)
    args = p.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    cat_path = Path(args.catalog or config.get("dataset", {}).get(
        "catalog_file", "data/catalog/hipparcos_bright.csv"))
    if not cat_path.exists():
        print(f"ERROR: catalog not found: {cat_path}", file=sys.stderr)
        return 1

    print("=" * 65)
    print("  StellarX Phase 6 — Inference Benchmark")
    print("=" * 65)
    print(f"  Config        : {cfg_path}")
    print(f"  Catalog       : {cat_path}")
    print(f"  Test images   : {args.n_images}")
    print(f"  Warmup images : {args.n_warmup}")
    print(f"  Random seed   : {args.seed}")
    print()

    from src.catalog.catalog_loader import load_catalog
    from src.preprocessing.star_field_generator import StarFieldGenerator
    from src.navigation.navigator import _preprocess_image
    from src.optimization.pipeline import (
        OptimizedPipeline, compare_baseline_vs_optimized
    )
    from src.optimization.profiler import PipelineProfiler
    from src.optimization.edge_config import print_deployment_summary

    # ── Build test images ─────────────────────────────────────────────────
    print("Building test images ...")
    t0 = time.perf_counter()
    catalog = load_catalog(cat_path, config=config)
    gen = StarFieldGenerator(catalog, config.get("dataset", {}))
    images = []
    for i in range(args.n_images + args.n_warmup):
        sf = gen.generate(seed=args.seed + i)
        images.append(_preprocess_image(sf.image, config))
    print(f"  {len(images)} images built in {(time.perf_counter()-t0)*1000:.0f} ms")
    print()

    # ── Profile ───────────────────────────────────────────────────────────
    print("Profiling pipeline components ...")
    profiler = PipelineProfiler(config, cat_path)
    profile_report = profiler.profile(images, n_runs=min(10, len(images)), n_warmup=2)
    PipelineProfiler.print_report(profile_report)

    # ── Optimized benchmark ───────────────────────────────────────────────
    print("\nRunning optimized pipeline benchmark ...")
    pipeline = OptimizedPipeline(config, cat_path)
    bench = pipeline.benchmark(images, n_warmup=args.n_warmup)

    print("\n" + "="*65)
    print("  OPTIMIZED PIPELINE BENCHMARK (warm inference)")
    print("="*65)
    print(f"  Images (post-warmup)  : {bench.n_images}")
    print(f"  Cold start            : {bench.cold_start_ms:.1f} ms  (catalog load + KD-tree build)")
    print()
    print(f"  Mean latency          : {bench.mean_latency_ms:.1f} ms")
    print(f"  Median latency        : {bench.median_latency_ms:.1f} ms")
    print(f"  Min latency           : {bench.min_latency_ms:.1f} ms")
    print(f"  Max latency           : {bench.max_latency_ms:.1f} ms")
    print(f"  P95 latency           : {bench.p95_latency_ms:.1f} ms")
    print(f"  P99 latency           : {bench.p99_latency_ms:.1f} ms")
    print(f"  FPS (warm)            : {bench.fps:.2f}")
    print()
    print(f"  Peak memory           : {bench.peak_memory_mb:.3f} MB")
    print(f"  Current memory        : {bench.current_memory_mb:.3f} MB")
    print()
    print(f"  SUCCESS               : {bench.n_success}")
    print(f"  PARTIAL               : {bench.n_partial}")
    print(f"  FAILURE/LOW           : {bench.n_failure}")
    print(f"  Recognition accuracy  : {bench.recognition_accuracy*100:.1f}%")
    if not math.isnan(bench.mean_attitude_residual_deg):
        print(f"  Mean att. residual    : {bench.mean_attitude_residual_deg:.4f} deg")
    else:
        print(f"  Mean att. residual    : N/A")
    print()
    print(f"  Per-component mean latency (ms):")
    for k, v in bench.component_times_ms.items():
        pct = (v / max(bench.mean_latency_ms, 1e-9)) * 100
        print(f"    {k:<22}: {v:>7.2f} ms  ({pct:.1f}%)")
    print("="*65)

    # ── Baseline comparison ───────────────────────────────────────────────
    if not args.skip_baseline:
        print("\nRunning baseline comparison (rebuilds catalog per image) ...")
        cmp_imgs = images[:min(6, len(images))]
        cmp = compare_baseline_vs_optimized(
            config, cat_path, cmp_imgs, n_warmup=args.n_warmup
        )
        cmp.print_table()

    # ── Edge deployment ───────────────────────────────────────────────────
    if args.edge:
        print_deployment_summary(measured_result=bench)

    print("\nBenchmark complete.")
    print(f"Run command: python benchmark.py --n-images {args.n_images}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
