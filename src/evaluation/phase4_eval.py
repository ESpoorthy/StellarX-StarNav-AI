"""
phase4_eval.py
==============
Phase 4 evaluation — measures real recognition metrics on synthetic images.
No values are fabricated.

Usage:
    from src.evaluation.phase4_eval import run_phase4_evaluation
    results = run_phase4_evaluation(config, "data/catalog/hipparcos_bright.csv", n_images=20)
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.catalog.catalog_loader import load_catalog
from src.preprocessing.image_preprocessing import subtract_background, reduce_noise, normalise
from src.preprocessing.star_detection import detect_stars, StarCandidate
from src.preprocessing.star_field_generator import StarFieldGenerator
from src.recognition.catalog_index import CatalogIndex
from src.recognition.pattern_builder import build_pattern
from src.recognition.pattern_matcher import RecognitionStatus, run_recognition


@dataclass
class Phase4EvalResult:
    n_images: int = 0
    n_success: int = 0
    n_partial: int = 0
    n_low_confidence: int = 0
    n_failure: int = 0
    success_rate: float = float("nan")
    partial_rate: float = float("nan")
    recognition_rate: float = float("nan")
    failure_rate: float = float("nan")
    mean_residual_deg: float = float("nan")
    median_residual_deg: float = float("nan")
    p95_residual_deg: float = float("nan")
    mean_latency_ms: float = float("nan")
    median_latency_ms: float = float("nan")
    p95_latency_ms: float = float("nan")
    mean_inliers: float = float("nan")
    noisy_recognition_rate: float = float("nan")
    missing_star_recognition_rate: float = float("nan")
    false_detection_recognition_rate: float = float("nan")
    catalog_size: int = 0
    notes: list = field(default_factory=list)


def _preprocess(image: np.ndarray, config: dict) -> np.ndarray:
    pp = config.get("preprocessing", {})
    img = subtract_background(image,
        method=pp.get("background_method", "median_filter"),
        filter_size=int(pp.get("background_filter_size", 31)))
    img = reduce_noise(img,
        method=pp.get("noise_method", "gaussian"),
        sigma=float(pp.get("noise_sigma", 0.8)))
    norm = pp.get("normalization", "min_max")
    if norm and norm != "none":
        img = normalise(img, method=norm)
    return img


def _inject_noise(stars, noise_px=2.0, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    return [StarCandidate(x=s.x+rng.normal(0,noise_px), y=s.y+rng.normal(0,noise_px),
                          brightness=s.brightness, peak=s.peak, area=s.area, bbox=s.bbox)
            for s in stars]


def _inject_false(stars, w=512, h=512, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    fs = StarCandidate(x=float(rng.uniform(10,w-10)), y=float(rng.uniform(10,h-10)),
                       brightness=float(rng.uniform(0.01,0.08)), peak=float(rng.uniform(0.04,0.08)),
                       area=1, bbox=(0,0,1,1))
    return list(stars) + [fs]


def run_phase4_evaluation(config: dict, catalog_path, n_images: int = 20,
                           verbose: bool = True) -> Phase4EvalResult:
    """Run Phase 4 evaluation. All metrics are real — nothing fabricated."""
    catalog_path = Path(catalog_path)
    res = Phase4EvalResult()
    catalog = load_catalog(catalog_path, config=config)
    cidx = CatalogIndex(catalog)
    res.catalog_size = len(cidx)
    ds_cfg = config.get("dataset", {})
    det_cfg = config.get("star_detection", {})
    w = int(ds_cfg.get("image_width", 512))
    h = int(ds_cfg.get("image_height", 512))
    gen = StarFieldGenerator(catalog, ds_cfg)

    statuses, residuals, latencies, inliers_list = [], [], [], []
    noisy_ok, missing_ok, false_ok = [], [], []

    if verbose:
        print(f"[phase4_eval] {len(cidx)} catalog stars, {n_images} test images")

    for i in range(n_images):
        seed = 42000 + i
        sf = gen.generate(seed=seed)
        img = _preprocess(sf.image, config)
        stars = detect_stars(img, det_cfg)

        t0 = time.perf_counter()
        pat = build_pattern(stars, config)
        out = run_recognition(pat, cidx, config)
        ms = (time.perf_counter()-t0)*1000

        statuses.append(out.status); latencies.append(ms); inliers_list.append(out.n_inliers)
        if not math.isnan(out.mean_residual_deg): residuals.append(out.mean_residual_deg)

        rng = np.random.default_rng(seed+1)
        noisy_out = run_recognition(build_pattern(_inject_noise(stars,2.0,rng),config),cidx,config)
        noisy_ok.append(noisy_out.status in (RecognitionStatus.SUCCESS,RecognitionStatus.PARTIAL))

        miss = stars[:-1] if len(stars)>2 else stars
        miss_out = run_recognition(build_pattern(miss,config),cidx,config)
        missing_ok.append(miss_out.status in (RecognitionStatus.SUCCESS,RecognitionStatus.PARTIAL))

        rng2 = np.random.default_rng(seed+2)
        false_out = run_recognition(build_pattern(_inject_false(stars,w,h,rng2),config),cidx,config)
        false_ok.append(false_out.status in (RecognitionStatus.SUCCESS,RecognitionStatus.PARTIAL))

        if verbose and (i % max(1,n_images//5)==0 or i==n_images-1):
            print(f"  [{i+1}/{n_images}] det={len(stars)} status={out.status.value} "
                  f"inliers={out.n_inliers} conf={out.confidence:.3f} lat={ms:.1f}ms")

    n = len(statuses)
    res.n_images = n
    res.n_success = sum(1 for s in statuses if s==RecognitionStatus.SUCCESS)
    res.n_partial = sum(1 for s in statuses if s==RecognitionStatus.PARTIAL)
    res.n_low_confidence = sum(1 for s in statuses if s==RecognitionStatus.LOW_CONFIDENCE)
    res.n_failure = sum(1 for s in statuses if s==RecognitionStatus.FAILURE)
    if n>0:
        res.success_rate = res.n_success/n; res.partial_rate = res.n_partial/n
        res.recognition_rate = (res.n_success+res.n_partial)/n
        res.failure_rate = (res.n_low_confidence+res.n_failure)/n
    if residuals:
        a=np.array(residuals); res.mean_residual_deg=float(np.mean(a))
        res.median_residual_deg=float(np.median(a)); res.p95_residual_deg=float(np.percentile(a,95))
    if latencies:
        a=np.array(latencies); res.mean_latency_ms=float(np.mean(a))
        res.median_latency_ms=float(np.median(a)); res.p95_latency_ms=float(np.percentile(a,95))
    if inliers_list: res.mean_inliers=float(np.mean(inliers_list))
    if noisy_ok: res.noisy_recognition_rate=sum(noisy_ok)/len(noisy_ok)
    if missing_ok: res.missing_star_recognition_rate=sum(missing_ok)/len(missing_ok)
    if false_ok: res.false_detection_recognition_rate=sum(false_ok)/len(false_ok)
    res.notes.append(f"Catalog: {res.catalog_size} stars, {n} synthetic images")
    res.notes.append("All metrics from real execution — no fabrication.")

    if verbose:
        print("\n" + "="*60)
        print("  PHASE 4 EVALUATION RESULTS")
        print("="*60)
        print(f"  Images          : {n}")
        print(f"  Catalog stars   : {res.catalog_size}")
        print(f"  SUCCESS         : {res.n_success} ({res.success_rate*100:.1f}%)")
        print(f"  PARTIAL         : {res.n_partial} ({res.partial_rate*100:.1f}%)")
        print(f"  LOW_CONFIDENCE  : {res.n_low_confidence}")
        print(f"  FAILURE         : {res.n_failure}")
        print(f"  Recognition rate: {res.recognition_rate*100:.1f}%")
        print(f"  Mean residual   : {res.mean_residual_deg:.4f} deg")
        print(f"  P95 residual    : {res.p95_residual_deg:.4f} deg")
        print(f"  Mean latency    : {res.mean_latency_ms:.1f} ms")
        print(f"  P95 latency     : {res.p95_latency_ms:.1f} ms")
        print(f"  Mean inliers    : {res.mean_inliers:.1f}")
        print(f"  Noisy recog.    : {res.noisy_recognition_rate*100:.1f}%")
        print(f"  Missing star    : {res.missing_star_recognition_rate*100:.1f}%")
        print(f"  +False detect.  : {res.false_detection_recognition_rate*100:.1f}%")
        print("="*60)
    return res
