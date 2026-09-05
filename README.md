# StellarX StarNav-AI

**Autonomous Spacecraft Navigation via AI Star Pattern Recognition**

> 🔬 All 7 phases complete · Team StellarX

---

## What It Does

StellarX StarNav-AI determines spacecraft orientation (attitude) from a single star-field image — no GPS, no ground contact required. A star-field image goes through a 7-stage AI pipeline and produces a quaternion + Euler angle output in under 100 ms.

---

## Pipeline

```
Star-Field Image
      │
      ▼
Phase 1 ── Data Foundation       Hipparcos catalog · Synthetic image generator
      │
      ▼
Phase 2 ── Star Detection         Background subtraction · Centroiding
      │
      ▼
Phase 3 ── Feature Extraction     Pairwise distances + brightness ratios (90-dim)
      │
      ▼
Phase 4 ── Pattern Recognition    Vote matrix · RANSAC · Catalog matching
      │
      ▼
Phase 5 ── Attitude Estimation    Wahba/SVD weighted least-squares → Quaternion
      │
      ▼
Phase 6 ── Optimization           Vectorized NumPy · Benchmarking · Edge config
      │
      ▼
Phase 7 ── Demonstration          Streamlit dashboard (app.py)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Run all tests
pytest tests/ -v

# 4. Generate synthetic dataset
python -c "
import yaml
from src.preprocessing.dataset_builder import build_dataset
config = yaml.safe_load(open('config.yaml'))
build_dataset(config)
"

# 5. Run the full pipeline on a single image
python run_pipeline.py
```

---

## Repository Structure

```
StellarX-StarNav-AI/
│
├── app.py                      ← Streamlit dashboard (Phase 7)
├── benchmark.py                ← Phase 6 inference benchmarking
├── run_pipeline.py             ← CLI pipeline runner
├── config.yaml                 ← All runtime parameters
├── requirements.txt
│
├── src/
│   ├── preprocessing/          ← Phase 1-2: image loading, detection, features
│   │   ├── image_preprocessing.py
│   │   ├── star_detection.py
│   │   ├── star_field_generator.py
│   │   ├── dataset_builder.py
│   │   └── feature_dataset.py
│   │
│   ├── catalog/                ← Phase 1: Hipparcos catalog loader
│   │   ├── catalog_loader.py
│   │   └── pattern_matcher.py
│   │
│   ├── recognition/            ← Phase 4: pattern matching pipeline
│   │   ├── catalog_index.py
│   │   ├── pattern_builder.py
│   │   ├── pattern_matcher.py
│   │   └── pattern_matcher_optimized.py
│   │
│   ├── models/                 ← Phase 3: feature classifier + inference
│   │   ├── sklearn_classifier.py
│   │   ├── inference.py
│   │   └── star_pattern_model.py   ← PyTorch stub (Python 3.14 not yet supported)
│   │
│   ├── navigation/             ← Phase 5: attitude + position estimation
│   │   ├── navigator.py
│   │   ├── attitude_estimator.py
│   │   ├── position_estimator.py
│   │   └── camera_model.py
│   │
│   ├── optimization/           ← Phase 6: benchmarking + edge deployment
│   │   ├── pipeline.py
│   │   ├── profiler.py
│   │   └── edge_config.py
│   │
│   ├── evaluation/             ← Phase 4-5 evaluation metrics
│   │   ├── phase4_eval.py
│   │   └── phase5_eval.py
│   │
│   └── utils/
│       └── visualization.py
│
├── demo/
│   └── demo_assets.py          ← Pre-generated demo data for the dashboard
│
├── data/
│   ├── catalog/
│   │   └── hipparcos_bright.csv   ← 50 brightest stars (Hipparcos, ESA 1997)
│   ├── raw/                    ← Generated images (gitignored)
│   └── processed/              ← Feature datasets (gitignored)
│
├── models/                     ← Trained checkpoints (gitignored)
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
│   └── test_phase6_optimization.py
│
└── docs/
    ├── architecture.md
    ├── methodology.md
    ├── dataset.md
    ├── experiments.md
    └── results.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Image processing | OpenCV, SciPy, Pillow |
| Star detection | Connected-component analysis (scipy.ndimage) |
| ML classifier | scikit-learn (RandomForest / KNN / MLP) |
| Attitude solver | Wahba/SVD weighted least-squares |
| Catalog | Hipparcos (ESA 1997, public domain) |
| Dashboard | Streamlit 1.56 |
| Deep learning | PyTorch (stub — awaiting Python 3.14 wheel) |

---

## Test Results

```
363 passed · 15 skipped · 0 failures
```

Skipped tests are expected (sparse-catalog frames with 0 visible stars).

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

This repository is an engineering implementation of concepts from a published invention. IP rights are governed by the applicable patent and institutional agreements.
