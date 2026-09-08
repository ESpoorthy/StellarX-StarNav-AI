# Results

> **Status:** Results template — no experiments have been conducted yet.
> This document will be populated during and after the evaluation phases.
> Do not add fabricated or estimated results.

---

## 1. Star Detection Performance

*To be completed after Phase 2 implementation and evaluation.*

| Metric | Value | Notes |
|--------|-------|-------|
| Detection rate (%) | — | |
| False-positive rate (%) | — | |
| Mean centroid error (px) | — | |
| Detection rate vs. star brightness | — | |
| Evaluation dataset | — | |

**Analysis:** *(to be written after evaluation)*

---

## 2. Pattern Recognition Accuracy

*To be completed after Phase 3–4 implementation and evaluation.*

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 accuracy (%) | — | |
| Top-5 accuracy (%) | — | |
| Precision | — | |
| Recall | — | |
| F1 score | — | |
| Accuracy vs. number of detected stars | — | |
| Accuracy vs. star density | — | |

**Confusion analysis:** *(to be written after evaluation)*

---

## 3. Catalog Matching Accuracy

*To be completed after Phase 4 implementation and evaluation.*

| Metric | Value | Notes |
|--------|-------|-------|
| Match rate (%) | — | |
| False-match rate (%) | — | |
| Mean match confidence | — | |
| Match rate at confidence threshold | — | |

**Analysis:** *(to be written after evaluation)*

---

## 4. Navigation Estimation Performance

*To be completed after Phase 5 implementation and evaluation.*

### 4.1 Attitude Estimation

| Metric | Value | Notes |
|--------|-------|-------|
| Mean angular error (deg) | — | |
| Median angular error (deg) | — | |
| 90th-percentile angular error (deg) | — | |
| Error distribution | — | |

### 4.2 Position Estimation

| Metric | Value | Notes |
|--------|-------|-------|
| Applicable (yes/no) | — | Depends on methodology selected in Phase 5 |
| Mean position error | — | |

**Analysis:** *(to be written after evaluation)*

---

## 5. Confidence Analysis

*To be completed after Phase 4 implementation.*

| Metric | Value | Notes |
|--------|-------|-------|
| Confidence calibration curve | — | |
| Mean confidence (correct matches) | — | |
| Mean confidence (incorrect matches) | — | |
| Optimal confidence threshold | — | |
| Coverage at optimal threshold (%) | — | |

**Analysis:** *(to be written after evaluation)*

---

## 6. Inference Latency

*To be completed after Phase 6 benchmarking.*

| Hardware | Mean latency (ms) | P95 latency (ms) | Notes |
|----------|-------------------|------------------|-------|
| Development machine (CPU) | — | — | |
| Development machine (GPU) | — | — | |
| Target edge platform | — | — | TBD |

**Analysis:** *(to be written after benchmarking)*

---

## 7. Resource Utilization

*To be completed after Phase 6 optimization.*

| Metric | Value | Notes |
|--------|-------|-------|
| Model size (MB) | — | |
| Peak RAM usage (MB) | — | |
| Peak GPU memory (MB) | — | If applicable |
| FLOP count (approximate) | — | |

**Analysis:** *(to be written after benchmarking)*

---

## 8. Comparison with Baseline Methods

*To be completed once baseline methods are defined and implemented for comparison.*

| Method | Top-1 Acc (%) | Match Rate (%) | Mean Latency (ms) | Notes |
|--------|--------------|----------------|-------------------|-------|
| StellarX (proposed) | — | — | — | |
| Baseline 1 | — | — | — | TBD |
| Baseline 2 | — | — | — | TBD |

**Baseline methods:** to be selected during Phase 3–4 to provide a meaningful comparison.

---

## 9. Limitations

*To be documented after evaluation.*

Known anticipated limitations:
- Performance may degrade under very low star density (few detectable stars in the field of view)
- Generalisation to unseen sensor types has not yet been evaluated
- Real spacecraft imagery with ground-truth may be limited; results on synthetic data may not fully characterise real-world performance
- Computational benchmarks are hardware-dependent

Additional limitations identified during evaluation will be recorded here.

---

## 10. Future Improvements

*To be proposed after initial evaluation is complete.*

| Improvement | Motivation | Priority |
|-------------|------------|----------|
| *(to be added)* | | |

---

## Reproducibility

All reported results are linked to specific experiment records in [`docs/experiments.md`](experiments.md). To reproduce a result:
1. Check out the commit corresponding to the experiment date.
2. Use the configuration recorded in the experiment entry.
3. Use the dataset version recorded in the experiment entry.
4. Run training or evaluation as described in `CONTRIBUTING.md`.

---

## SIH 2026 Sprint — Performance Fix Summary

### Phase 5 Root Cause (identified and fixed)

The apparent "hang at Phase 5" was caused by multiple compounding inefficiencies
in the Phase 4 recognition pipeline, which blocked the Streamlit UI thread before
Phase 5 even started.

**Root causes:**

1. **Vote accumulation** — 45 Python calls × 1225-entry Python dict linear scan
   = ~55,000 Python operations per image. Fixed: vectorized NumPy array comparison.

2. **RANSAC inner loop** — 30 iterations × N Python per-correspondence operations.
   Fixed: `ransac_inlier_count_vectorized` using matrix multiply.

3. **Wahba B-matrix** — Python for-loop over correspondences.
   Fixed: `(cat * w[:,None]).T @ obs` vectorized broadcast.

4. **Residual computation** — Python for-loop.
   Fixed: vectorized `arccos(dot products)`.

5. **Double preprocessing** — `app.py` preprocessed raw, then called `run_full_pipeline`
   which preprocessed raw again. Fixed: single preprocessing pass.

6. **No input validation** — NaN/Inf vectors could enter the SVD solver silently.
   Fixed: strip before every solve.

**Result:** pipeline terminates deterministically with bounded RANSAC iterations (30)
and bounded outlier rejection iterations (3). No infinite loops possible.

### Integrity Check Added

Every attitude solution is now verified post-SVD:
- `|det(R) − 1| < 0.001`
- `‖RᵀR − I‖_max < 1e-5`
- Quaternion norm in [0.999, 1.001]
- Mean residual < 2.0°
- Max per-star residual < 3.0°
- RANSAC inliers ≥ 2

Status: `VALID` / `DEGRADED` / `REJECTED` with explicit rejection reason.
