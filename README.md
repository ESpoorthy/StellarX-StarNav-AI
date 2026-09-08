# StellarX StarNav-AI

**Autonomous Lost-in-Space Attitude Determination for Small Spacecraft**

SIH 2026 · Team StellarX · Software · Space Technology · AICTE

---

## Problem

A spacecraft in orbit does not always know its orientation.  GPS provides position, not orientation.  Ground-contact windows are limited.  When neither is available, the spacecraft must determine its own attitude — which way it is pointing — from the one sensor that always works: its star tracker.

StellarX StarNav-AI solves this problem from a single star-field image, with no GPS, no ground contact, and no prior attitude knowledge.

---

## What It Does

```
Star-field image (onboard camera)
         │
         ▼
DETECT    Stars extracted via connected-component analysis
         │
         ▼
RECOGNISE AI-assisted pattern recognition (vote matrix + RANSAC)
         │         matches pairwise angular separations against
         ▼         the Hipparcos star catalogue
SOLVE     Wahba/SVD weighted least-squares → rotation matrix R
         │
         ▼
VERIFY    Integrity check: det(R)≈+1, |q|≈1, residual < threshold
         │
         ▼
OUTPUT    Quaternion + Euler angles + integrity status
```

**Outputs attitude. Does not output position.**
Single-image star geometry determines orientation (3 DoF), not location.
This is scientifically correct and honestly documented throughout the system.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# Run the full test suite
pytest tests/ -v

# Run the pipeline on a single image (CLI)
python run_pipeline.py

# Generate synthetic training dataset
python -c "
import yaml
from src.preprocessing.dataset_builder import build_dataset
config = yaml.safe_load(open('config.yaml'))
build_dataset(config)
"

# Run the performance benchmark
python benchmark.py
```

---

## Repository Structure

```
StellarX-StarNav-AI/
│
├── app.py                        ← Streamlit dashboard (Phase 7)
├── benchmark.py                  ← Phase 6 benchmarking CLI
├── run_pipeline.py               ← Single-image CLI runner
├── config.yaml                   ← All runtime parameters
├── requirements.txt
│
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py     Phase 2: bg subtraction, noise, normalise
│   │   ├── star_detection.py          Phase 2: CC analysis, centroiding, feature extraction
│   │   ├── star_field_generator.py    Phase 1: synthetic image generation
│   │   ├── dataset_builder.py         Phase 1: batch generation + metadata JSON
│   │   └── feature_dataset.py         Phase 3: feature matrix builder
│   │
│   ├── catalog/
│   │   └── catalog_loader.py          Phase 1: Hipparcos CSV loader + queries
│   │
│   ├── recognition/
│   │   ├── catalog_index.py           Phase 4: KD-tree + vectorized pair lookup
│   │   ├── pattern_builder.py         Phase 4: pixel → unit-vector conversion
│   │   ├── pattern_matcher.py         Phase 4: vote matrix + RANSAC + Wahba
│   │   └── pattern_matcher_optimized.py  Phase 6: vectorized ops (matmul/einsum)
│   │
│   ├── models/
│   │   ├── sklearn_classifier.py      Phase 3: RandomForest/KNN/MLP classifier
│   │   ├── inference.py               Phase 3: load_model / run_inference dispatcher
│   │   └── star_pattern_model.py      PyTorch stub (awaiting Python 3.14 wheel)
│   │
│   ├── navigation/
│   │   ├── navigator.py               Phase 5: end-to-end orchestrator
│   │   ├── attitude_estimator.py      Phase 5: Wahba/SVD + outlier rejection
│   │   ├── camera_model.py            Phase 5: pinhole model, pixel↔unit-vector
│   │   ├── position_estimator.py      Phase 5: honest UNAVAILABLE stub
│   │   └── pipeline_result.py         Phase 5: structured result + integrity check
│   │
│   ├── optimization/
│   │   ├── pipeline.py                Phase 6: OptimizedPipeline + benchmark
│   │   ├── profiler.py                Phase 6: per-component latency profiler
│   │   └── edge_config.py             Phase 6: edge deployment profiles
│   │
│   └── evaluation/
│       ├── phase4_eval.py             Phase 4: recognition evaluation harness
│       ├── phase5_eval.py             Phase 5: attitude evaluation harness
│       └── ground_truth_validator.py  Phase 5: geodesic SO(3) error validation
│
├── demo/
│   └── demo_assets.py                Phase 7: pre-generated demo assets
│
├── data/
│   ├── catalog/
│   │   └── hipparcos_bright.csv       50 brightest stars (Hipparcos, ESA 1997)
│   ├── raw/                           Generated images (gitignored)
│   └── processed/                     Feature datasets (gitignored)
│
├── models/                            Trained checkpoints (gitignored)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_star_detection.ipynb
│   ├── 03_pattern_generation.ipynb
│   └── 04_model_training.ipynb
│
├── tests/
│   ├── test_catalog_loader.py
│   ├── test_star_detection.py
│   ├── test_star_field_generator.py
│   ├── test_pattern_matching.py
│   ├── test_models.py
│   ├── test_navigation.py
│   ├── test_phase4_recognition.py
│   ├── test_phase5_navigation.py
│   ├── test_phase6_optimization.py
│   └── test_pipeline_robustness.py    NaN guards / degenerate / GT / perf
│
└── docs/
    ├── architecture.md
    ├── methodology.md
    ├── dataset.md
    ├── experiments.md
    └── results.md
```

---

## Implementation Status

| Phase | Description | Status |
|---|---|---|
| 1 | Synthetic data pipeline, Hipparcos catalogue | ✅ Complete |
| 2 | Star detection, background subtraction, centroiding | ✅ Complete |
| 3 | Feature extraction (pairwise descriptors), sklearn classifier | ✅ Complete |
| 4 | Vote-matrix + RANSAC catalogue matching | ✅ Complete |
| 5 | Wahba/SVD attitude estimation, integrity check | ✅ Complete |
| 6 | Vectorized NumPy optimizations, benchmarking | ✅ Complete |
| 7 | Streamlit mission dashboard | ✅ Complete |

---

## Technical Architecture

### Star Detection
Connected-component analysis on background-subtracted, normalised images.
Intensity-weighted centroiding achieves sub-pixel accuracy (< 0.15 px, noiseless synthetic).

### AI-Assisted Pattern Recognition
1. **AI prior** (optional): sklearn RandomForest classifies the observed star distribution into a sky-cell label. Provides a prior for the geometric matcher.
2. **Vote matrix**: for every observed star pair, finds all catalogue pairs with matching angular separation (within `angle_tolerance_deg`). Votes accumulate to build star→catalogue correspondences.
3. **RANSAC**: 30 iterations (bounded). Each hypothesis uses TRIAD rotation from 2 correspondences; inlier counting is vectorized (`ransac_inlier_count_vectorized`).
4. **Wahba/SVD refinement**: weighted least-squares rotation fit over all RANSAC inliers (vectorized B-matrix via einsum).

### Attitude Estimation (Wahba/SVD)
Solves the weighted Wahba problem:

```
minimise  Σᵢ wᵢ ‖ref_i − R obs_i‖²

B = Σᵢ (wᵢ · outer(ref_i, obs_i))  =  (ref * w[:,None]).T @ obs
B = U S Vᵀ
R = U diag(1, 1, det(U Vᵀ)) Vᵀ
```

Input validation before every solve:
- NaN / Inf rows silently removed
- Zero-length vectors removed
- Inputs normalised to unit length
- Weights clipped to [ε, ∞)

Post-solve validation:
- `|det(R) − 1| < ε`
- `‖RᵀR − I‖ < ε`
- Quaternion normalised, `|‖q‖ − 1| < ε`

### Integrity Check
Every solution is independently verified:

| Check | Threshold |
|---|---|
| `det(R)` | ≥ 0.999 |
| Quaternion norm | [0.999, 1.001] |
| Mean residual | ≤ 2.0° |
| Max per-star residual | ≤ 3.0° |
| RANSAC inliers | ≥ 2 |

Result: `VALID` / `DEGRADED` / `REJECTED` with an explicit rejection reason.

### Position
Single-image star geometry provides **attitude (3 DoF orientation) only**.
Absolute position requires multi-image triangulation, orbital mechanics, or additional sensors.
This is scientifically correct and is stated explicitly throughout the system.

---

## Phase 5 Performance Fix (SIH 2026 Sprint)

Root cause of the previous Phase 5 hang was identified and fixed:

| Component | Before | After |
|---|---|---|
| Vote accumulation | Python dict scan (45 × 1225 iterations) | Vectorized NumPy array comparison |
| RANSAC inlier count | Python for-loop per hypothesis | `matmul`-based vectorized counting |
| Wahba B-matrix | Python for-loop over correspondences | `einsum` / broadcast |
| Residual computation | Python for-loop | Vectorized `arccos(dot)` |
| Pairwise angle precomputation | Python double for-loop | Single matrix multiply |
| Double preprocessing | Raw preprocessed twice | Single pass, result shared |
| Input validation | None | NaN/Inf/zero stripped before every SVD |

---

## Test Coverage

```
pytest tests/ -v
```

**~420 tests** across 12 modules covering:
- Catalog loading, queries, unit vectors
- Star detection, centroiding, feature extraction
- Pattern matching, RANSAC, Wahba/SVD
- Navigation pipeline end-to-end
- NaN/Inf/zero input guards
- Degenerate geometry (collinear, 0/1/2 stars)
- Synthetic ground-truth attitude recovery
- Intentional failure scenarios
- Integrity check (bad det, zero quaternion, excessive residual)
- Performance regression (pipeline < 30s, solver < 50ms avg)
- Vectorized ops numerical correctness

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Star detection | scipy.ndimage (connected components) |
| Feature descriptors | NumPy (pairwise distances + ratios) |
| AI classifier | scikit-learn RandomForest |
| Attitude solver | NumPy linalg.svd (Wahba/SVD) |
| Star catalogue | Hipparcos ESA 1997 (public domain) |
| Deep learning | PyTorch stub — no Python 3.14 wheel yet |
| Dashboard | Streamlit 1.56 |

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| 50-star prototype catalogue | Many pointing directions yield 0 visible stars | Extend to full Hipparcos (~118K) before production |
| Synthetic data only | Real sensor PSF, stray light not modelled | Validate on real FITS imagery (Phase 2+) |
| No PyTorch on Python 3.14 | AI backend is sklearn only | PyTorch planned once wheel is available |
| Position not computed | Attitude only | Scientifically correct; stated explicitly |

---

## Team StellarX

| Name | GitHub |
|---|---|
| Sai Spoorthy Eturu | [@ESpoorthy](https://github.com/ESpoorthy) |
| Kommera Harihanika | [@placedeliteverifypotxnicufu](https://github.com/placedeliteverifypotxnicufu) |
| Duddala Srija | [@Duddalasrija](https://github.com/Duddalasrija) |
| Glory Pranavi B | [@glory-pranavi](https://github.com/glory-pranavi) |
| Katakam Sahithi Rithvika | [@sahithrithvika](https://github.com/sahithrithvika) |
| Shamithri Gowravarapu | [@sham12398](https://github.com/sham12398) |

---

## IP Notice

Engineering implementation of concepts from a published invention.
IP rights governed by the applicable patent and institutional agreements.
