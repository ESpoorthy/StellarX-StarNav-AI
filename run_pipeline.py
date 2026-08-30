#!/usr/bin/env python3
"""
run_pipeline.py
===============
StellarX-StarNav-AI Phase 4 — single-frame pipeline demo.

Usage:
    python run_pipeline.py [--seed INT] [--catalog PATH] [--config PATH]

Exit code 0 on success, 1 on fatal error.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np, yaml


def main() -> int:
    p = argparse.ArgumentParser(description="StellarX Phase 4 pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--catalog", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boresight-ra", type=float, default=None)
    p.add_argument("--boresight-dec", type=float, default=None)
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

    from src.catalog.catalog_loader import load_catalog
    from src.recognition.catalog_index import CatalogIndex
    from src.preprocessing.star_field_generator import StarFieldGenerator
    from src.navigation.navigator import run_navigation, _preprocess_image

    print(f"Loading catalog: {cat_path}")
    t0 = time.perf_counter()
    catalog = load_catalog(cat_path, config=config)
    cidx = CatalogIndex(catalog)
    print(f"  {len(cidx)} stars indexed in {(time.perf_counter()-t0)*1000:.1f} ms")

    gen = StarFieldGenerator(catalog, config.get("dataset",{}))
    print(f"\nGenerating synthetic star field (seed={args.seed}) ...")
    sf = gen.generate(seed=args.seed, boresight_ra_deg=args.boresight_ra,
                      boresight_dec_deg=args.boresight_dec)
    print(f"  Boresight RA : {sf.boresight_ra_deg:.4f} deg")
    print(f"  Boresight Dec: {sf.boresight_dec_deg:.4f} deg")
    print(f"  Roll         : {sf.roll_deg:.4f} deg")
    print(f"  GT stars     : {len(sf.stars)}")

    img = _preprocess_image(sf.image, config)
    print("\nRunning Phase 4 navigation pipeline ...")
    result = run_navigation(img, config, cidx, neural_model=None)

    print("\n" + "="*60)
    print("  PIPELINE RESULT")
    print("="*60)
    print(f"  Status                   : {result.status}")
    print(f"  Observed stars           : {result.n_observed_stars}")
    print(f"  Matched stars            : {result.n_matched_stars}")
    print(f"  Inlier stars             : {result.n_inlier_stars}")
    q = result.quaternion
    print(f"  Quaternion [qw,qx,qy,qz]: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
    e = result.euler_angles_deg
    print(f"  Euler [yaw,pitch,roll]   : [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}] deg")
    r = result.attitude_residual_deg
    print(f"  Mean residual            : {r:.4f} deg" if not math.isnan(r) else "  Mean residual            : N/A")
    print(f"  Attitude confidence      : {result.attitude_confidence:.4f}")
    print(f"  Detection time           : {result.detection_time_ms:.1f} ms")
    print(f"  Recognition time         : {result.recognition_time_ms:.1f} ms")
    print(f"  Attitude time            : {result.attitude_time_ms:.1f} ms")
    print(f"  Total latency            : {result.total_time_ms:.1f} ms")
    if result.identified_stars:
        print(f"\n  Identified stars ({len(result.identified_stars)}):")
        for s in result.identified_stars[:5]:
            print(f"    {s.catalog_id:12s}  RA={s.catalog_ra_deg:8.4f}  "
                  f"Dec={s.catalog_dec_deg:8.4f}  residual={s.angular_residual_deg:.4f} deg  conf={s.confidence:.3f}")
        if len(result.identified_stars) > 5:
            print(f"    ... and {len(result.identified_stars)-5} more")
    print(f"\n  Position note: {result.position_note}")
    print("="*60)

    if result.status == "ERROR":
        print(f"\nERROR: {result.error_message}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
