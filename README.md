# StellarX — AI-Powered Star Pattern Recognition for Autonomous Spacecraft Navigation

> **Team:** StellarX
> **Status:** 🔬 Phase 1 complete — synthetic data pipeline implemented and tested. Phase 2 (Star Detection) is next.

---

## Overview

StellarX-StarNav-AI explores a neural-network-based approach to recognizing star patterns from onboard star-field imagery, with the goal of enabling autonomous spacecraft navigation without dependence on external positioning infrastructure.

The invention underlying this project describes a system that:

- Processes star-field images captured by an onboard imaging device
- Extracts characteristic features from detected stars
- Identifies star patterns from the extracted features
- Matches identified patterns against a stored star catalog
- Supports spacecraft position and attitude determination from catalog matches
- Reduces computational complexity and resource consumption compared to conventional methods
- Targets resource-constrained platforms such as CubeSats and small satellites

> **Note:** The descriptions above reflect the proposed system as described in the related invention. Phase 1 (dataset pipeline) is now complete; subsequent phases will implement the detection, recognition, and navigation components.

---

## Problem Statement

Determining a spacecraft's orientation and position from onboard sensors is a fundamental challenge in autonomous navigation. Star-based navigation — identifying stars in imagery captured by an onboard camera and matching them against a known catalog — is one of the most accurate methods available. However, doing this reliably under real operational conditions is hard:

- Star-field images are noisy, low-contrast, and affected by sensor artifacts
- The number of visible stars and their brightness vary with sensor field of view and exposure
- Conventional catalog-matching algorithms can be computationally expensive
- Resource-constrained spacecraft (CubeSats, nanosats) have limited CPU, memory, and power budgets
- The system must operate autonomously without ground-in-the-loop corrections

This project targets a neural-network-based pipeline that can perform robust star pattern recognition efficiently enough for deployment on constrained spacecraft hardware.

---

## Proposed Solution

The planned end-to-end pipeline:

```
Star Field Image
        ↓
Image Preprocessing
        ↓
Star Detection
        ↓
Star Feature Extraction
        ↓
Neural Network
        ↓
Star Pattern Recognition
        ↓
Star Catalog Matching
        ↓
Position / Attitude Estimation
        ↓
Navigation Output
```

Each stage is a distinct, independently testable component. The pipeline is designed to be modular so that individual stages can be swapped, benchmarked, and optimized without breaking the others.

---

## Key Objectives

| Objective | Description |
|---|---|
| Star detection | Reliably locate stars in noisy star-field images |
| Pattern recognition | Identify star patterns using a trained neural network |
| Catalog matching | Efficiently match recognized patterns against a star catalog |
| Attitude estimation | Determine spacecraft orientation from matched patterns |
| Position estimation | Estimate position where the methodology supports it |
| Confidence-aware predictions | Attach confidence scores to recognition outputs |
| Computational efficiency | Keep inference within the budget of constrained hardware |
| Edge deployment | Support lightweight deployment on resource-limited platforms |

---

## Planned Development Phases

### Phase 1 — Foundation ✅ Complete
- Repository architecture and project scaffolding
- Hipparcos bright-star catalog integrated (`data/catalog/hipparcos_bright.csv`)
- Synthetic star-field generator implemented (`src/preprocessing/star_field_generator.py`)
- Dataset builder with reproducible 800/100/100 train/val/test splits (`src/preprocessing/dataset_builder.py`)
- 16-bit PNG image I/O implemented (`src/preprocessing/image_preprocessing.py`)
- Full catalog loader with spatial and magnitude queries (`src/catalog/catalog_loader.py`)
- Ground-truth metadata schema defined and validated
- Unit test suites: ~100 real assertions across 4 test modules
- Exploratory notebook `notebooks/01_data_exploration.ipynb` fully executable
- `docs/dataset.md` fully documented with source, schema, methodology, limitations

### Phase 2 — Star Detection *(next)*
- Background estimation and subtraction
- Star detection algorithm (blob detection or matched filter)
- Sub-pixel centroiding
- Feature extraction from detected stars

### Phase 3 — Neural Network
- Training data generation from synthetic pipeline
- Neural network architecture design and development
- Model training and evaluation

### Phase 4 — Pattern Recognition
- Star pattern identification
- Star catalog integration
- Confidence estimation for recognition outputs

### Phase 5 — Navigation
- Spacecraft attitude estimation from recognised patterns
- Position estimation where applicable
- Navigation output formatting

### Phase 6 — Optimization
- Inference benchmarking
- CPU and memory optimization
- Lightweight / edge deployment experiments

### Phase 7 — Demonstration
- Streamlit interactive interface
- Image upload and visualization
- Prediction results with confidence and processing-time metrics

---

## Repository Structure

```text
StellarX-StarNav-AI/
│
├── README.md               # Project overview (this file)
├── LICENSE                 # License file
├── .gitignore              # Python-oriented ignore rules
├── requirements.txt        # Core Python dependencies
├── config.yaml             # Centralised configuration template
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Community standards
│
├── data/
│   ├── raw/                # Original unmodified source data (not versioned)
│   ├── processed/          # Preprocessed data ready for training (not versioned)
│   └── catalog/            # Star catalog files
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_star_detection.ipynb
│   ├── 03_pattern_generation.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── preprocessing/      # Image loading, noise reduction, normalization
│   ├── models/             # Neural network definitions and inference
│   ├── catalog/            # Catalog loading and pattern matching
│   ├── navigation/         # Attitude and position estimation
│   └── utils/              # Shared utilities and visualization helpers
│
├── models/                 # Saved model checkpoints (not versioned)
├── app/                    # Streamlit demonstration application
├── tests/                  # Unit tests
└── docs/                   # Architecture, methodology, and results documentation
```

---

## Technology Stack

| Technology | Role | Status |
|---|---|---|
| Python 3.11+ | Primary language | ✅ In use |
| NumPy | Numerical arrays and computation | ✅ In use |
| Pandas | Data handling and catalog operations | ✅ In use |
| SciPy | Gaussian PSF, image convolution | ✅ In use |
| Pillow | 16-bit PNG image I/O | ✅ In use |
| Matplotlib | Visualization and result plotting | ✅ In use |
| PyYAML | Configuration management | ✅ In use |
| pytest | Unit testing | ✅ In use |
| OpenCV | Star detection (Phase 2) | Planned |
| PyTorch | Deep learning framework (Phase 3) | Planned |
| Scikit-learn | ML utilities and metrics (Phase 3+) | Planned |
| Streamlit | Interactive demonstration interface (Phase 7) | Planned |

---

## Quick Start — Generate the Dataset

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the full synthetic dataset (800 train / 100 val / 100 test)
python -c "
import yaml
from src.preprocessing.dataset_builder import build_dataset
with open('config.yaml') as f:
    config = yaml.safe_load(f)
build_dataset(config)
"

# 3. Run the unit tests
pytest tests/ -v

# 4. Explore the data
jupyter notebook notebooks/01_data_exploration.ipynb
```

> Generated images are saved to `data/raw/` and are **not committed** to the repository (see `.gitignore`).

---



The project will be evaluated against the following targets, to be defined concretely during implementation:

- **Accuracy** — pattern recognition and catalog match rate
- **Robustness** — performance under noise, partial occlusion, and variable star density
- **Computational efficiency** — inference time on target hardware class
- **Inference latency** — time from image input to navigation output
- **Memory usage** — peak RAM and model size
- **Reproducibility** — fixed seeds, versioned configs, documented experiments

---

## Team

**Team: StellarX**

| GitHub Handle | Name | Role |
|---|---|---|
| [@ESpoorthy](https://github.com/ESpoorthy) | Sai Spoorthy Eturu | Repository Owner |
| [@placedeliteverifypotxnicufu](https://github.com/placedeliteverifypotxnicufu) | Kommera Harihanika | Collaborator |
| [@Duddalasrija](https://github.com/Duddalasrija) | Duddala Srija | Collaborator |
| [@glory-pranavi](https://github.com/glory-pranavi) | Glory Pranavi B | Collaborator |
| [@Katakam Sahithi Rithvika](https://github.com/sahithrithvika) | Katakam Sahithi Rithvika | Collaborator |
| [@Shamithri Gowravarapu](https://github.com/sham12398) | Shamithri Gowravarapu | Collaborator |

---

## Intellectual Property Notice

This repository contains an engineering and research implementation of concepts related to a published invention. Intellectual-property rights and licensing are governed by the applicable patent and institutional agreements. This notice does not constitute a claim of ownership, a grant of license, or an opinion on freedom to operate.

---

## Documentation

Detailed documentation is located in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Planned system architecture and component diagram |
| [`docs/methodology.md`](docs/methodology.md) | Intended methodology for each pipeline stage |
| [`docs/dataset.md`](docs/dataset.md) | Catalog source, synthetic generation methodology, metadata schema |
| [`docs/experiments.md`](docs/experiments.md) | Experiment tracking template |
| [`docs/results.md`](docs/results.md) | Results reporting template |
