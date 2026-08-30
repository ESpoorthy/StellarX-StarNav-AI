#!/usr/bin/env python3
"""
benchmark.py
============
StellarX-StarNav-AI Phase 4 — benchmark and accuracy evaluation.

Usage:
    python benchmark.py [--n-images INT] [--config PATH] [--catalog PATH]

Exit code 0 on success, 1 on fatal error.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, yaml


def main() -> int:
    p = argparse.ArgumentParser(description="StellarX Phase 4 benchmark")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--catalog", default=None)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip baseline comparison (faster)")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr); return 1
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    cat_path = Path(args.catalog or config.get("dataset",{}).get(
        "catalog_file","data/catalog/hipparcos_bright.csv"))
    if not cat_path.exists():
        print(f"ERROR: catalog not found: {cat_path}", file=sys.stderr); return 1

    n_images = args.n_images
    print(f"StellarX Phase 4 Benchmark")
    print(f"  Config   : {cfg_path}")
    print(f"  Catalog  : {cat_path}")
    print(f"  Images   : {n_images}  Warmup: {args.n_warmup}")
    print()

    from src.catalog.catalog_loader import load_catalog
    from src.recognition.catalog_index import CatalogIndex
    from src.preprocessing.star_field_generator import StarFieldGenerator
    from src.navigation.navigator import _preprocess_image
    from src.optimization.pipeline import OptimizedPipeline, compare_baseline_vs_optimized
    from src.evaluation.phase4_eval import run_phase4_evaluation

    # Build test images
    print("Building test images ...")
    t0 = time.perf_counter()
    catalog = load_catalog(cat_path, config=config)
    gen = StarFieldGenerator(catalog, config.get("dataset",{}))
    images = []
    for i in range(n_images + args.n_warmup):
        sf = gen.generate(seed=10000 + i)
        images.append(_preprocess_image(sf.image, config))
    print(f"  Built {len(images)} images in {(time.perf_counter()-t0)*1000:.0f} ms")
    print()

    # ── Optimized pipeline benchmark ─────────────────────────────────────
    print("Running OptimizedPipeline benchmark ...")
    pipeline = OptimizedPipeline(config, cat_path)
    bench = pipeline.benchmark(images, n_warmup=args.n_warmup)

    print("\n" + "="*60)
    print("  OPTIMIZED PIPELINE BENCHMARK")
    print("="*60)
    print(f"  Images (excl. warmup) : {bench.n_images}")
    print(f"  Mean latency          : {bench.mean_latency_ms:.1f} ms")
    print(f"  Median latency        : {bench.median_latency_ms:.1f} ms")
    print(f"  P95 latency           : {bench.p95_latency_ms:.1f} ms")
    print(f"  P99 latency           : {bench.p99_latency_ms:.1f} ms")
    print(f"  FPS                   : {bench.fps:.1f}")
    print(f"  Peak memory           : {bench.peak_memory_mb:.2f} MB")
    print(f"  SUCCESS               : {bench.n_success}")
    print(f"  PARTIAL               : {bench.n_partial}")
    print(f"  FAILURE/LOW           : {bench.n_failure}")
    print(f"  Recognition accuracy  : {bench.recognition_accuracy*100:.1f}%")
    mres = bench.mean_attitude_residual_deg
    import math
    print(f"  Mean attitude residual: {mres:.4f} deg" if not math.isnan(mres) else "  Mean attitude residual: N/A")
    if bench.component_times_ms:
        print("\n  Per-component mean latency (ms):")
        for k,v in bench.component_times_ms.items():
            print(f"    {k:20s}: {v:.2f} ms")
    print("="*60)

    # ── Baseline comparison ───────────────────────────────────────────────
    if not args.skip_baseline and len(images) >= 3:
        print("\nRunning baseline vs optimized comparison ...")
        cmp_images = images[:min(10, len(images))]  # cap at 10 for speed
        cmp = compare_baseline_vs_optimized(config, cat_path, cmp_images)

        print("\n" + "="*60)
        print("  BASELINE vs OPTIMIZED COMPARISON")
        print("="*60)
        opt = cmp["optimized"]; base = cmp["baseline"]
        print(f"  Optimized mean latency : {opt.get('mean_latency_ms',0):.1f} ms")
        print(f"  Baseline mean latency  : {base.get('mean_latency_ms',0):.1f} ms")
        print(f"  Speedup ratio          : {cmp['speedup_ratio']:.2f}x")
        print(f"  Optimized accuracy     : {opt.get('recognition_accuracy',0)*100:.1f}%")
        print(f"  Baseline accuracy      : {base.get('recognition_accuracy',0)*100:.1f}%")
        print("="*60)

    # ── Phase 4 robustness evaluation ────────────────────────────────────
    print("\nRunning Phase 4 robustness evaluation ...")
    eval_images = min(n_images, 15)
    eval_result = run_phase4_evaluation(config, cat_path, n_images=eval_images, verbose=True)

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
