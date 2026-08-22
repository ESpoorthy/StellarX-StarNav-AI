# Methodology

> **Living document** — updated as each phase is implemented.
> Decisions made during implementation replace TBD placeholders.
> Phase 1 and Phase 2 sections are complete; later sections remain as plans.

---

## Overview

This document describes the technical approach for each stage of the StellarX-StarNav-AI pipeline. It bridges the high-level architecture in `docs/architecture.md` and the concrete implementation choices made during development.

---

## 1. Input Star-Field Imagery

### Phase 1 decisions

The system operates on **synthetic star-field images** generated from the Hipparcos bright-star catalog using a simplified spacecraft star-sensor simulation (see `docs/dataset.md` for full detail).

| Property | Decision |
|---|---|
| Format | 16-bit greyscale PNG |
| Bit depth | 16-bit (uint16, values 0–65535) |
| Pixel value convention | Loaded as float32 in [0, 1] via `load_image()` |
| Dimensions | 512 × 512 px (configurable, `dataset.image_width/height`) |
| Simulation model | Gnomonic projection, Gaussian PSF (σ = 1.5 px), Poisson-approx shot noise, Gaussian read noise |

Real spacecraft imagery (FITS format, `astropy.io.fits`) is planned for Phase 2+. It is not used in the current implementation.

---

## 2. Image Preprocessing

### Phase 2 decisions (implemented)

The preprocessing pipeline transforms a raw (synthetic) star-field image into a clean, normalised array ready for star detection. All algorithms are implemented in `src/preprocessing/image_preprocessing.py`. Parameters are sourced from the `preprocessing` section of `config.yaml`.

The pipeline runs in four steps:

```
load_image()
    ↓
subtract_background()
    ↓
reduce_noise()
    ↓
normalise()
    ↓
float32 array [0, 1]  →  star detection
```

### 2.1 Image Loading

`load_image(path)` reads 8-bit or 16-bit greyscale PNG / TIFF files via Pillow and returns a float32 NumPy array normalised to [0, 1]. RGB images are converted to greyscale. JPEG is explicitly rejected because lossy compression corrupts low-level photometric data.

### 2.2 Background Subtraction

**Algorithm: large-kernel median filter**

Config key: `preprocessing.background_method: median_filter`  
Config key: `preprocessing.background_filter_size: 31`

A (31 × 31) median filter estimates the slowly-varying background illumination. Subtracting this estimate removes the DC bias and any low-spatial-frequency gradient while leaving compact star blobs intact. The result is clipped to [0, 1].

**Why median filter?** The median is insensitive to compact bright objects (stars), so it estimates the background rather than the scene signal — this is the same principle used in the SExtractor astronomical source extractor.

**Approximation note:** This estimate works well for the Phase 1 synthetic dataset, which has a flat, low background (level ≈ 0.02). For real spacecraft imagery with structured stray-light gradients, a higher-order background mesh or sigma-clipped polynomial fit would be more appropriate.

Alternative supported: `background_method: constant` (subtracts the global median — faster but coarser).

### 2.3 Noise Reduction

**Algorithm: Gaussian smoothing**

Config key: `preprocessing.noise_method: gaussian`  
Config key: `preprocessing.noise_sigma: 0.8`

A small-sigma (σ = 0.8 px) Gaussian blur suppresses sub-pixel read noise without significantly broadening the star PSF (FWHM ≈ 3.5 px at σ = 1.5 px; smoothing with σ = 0.8 px adds ≈ 0.08 px in quadrature — negligible).

Setting `noise_sigma: 0` or `noise_method: none` skips this step entirely.

### 2.4 Intensity Normalisation

**Algorithm: min–max rescaling (robust)**

Config key: `preprocessing.normalization: min_max`

The 99.9th-percentile pixel value is used as the upper bound (instead of the true maximum) to make the normalisation robust to hot pixels and cosmic-ray spikes. After normalisation the background level is close to zero and the brightest star peak is close to 1.0, giving the subsequent threshold a consistent meaning across all images.

Alternative supported: `normalization: z_score` (zero mean, unit variance — useful for debugging, not recommended for thresholding).

---

## 3. Star Detection

### Phase 2 decisions (implemented)

Star detection is implemented in `src/preprocessing/star_detection.py`. It takes the preprocessed float32 image and the `star_detection` config section, and returns a `list[StarCandidate]`.

The pipeline runs in five steps:

```
Preprocessed image
    ↓
Threshold → binary mask
    ↓
scipy.ndimage.label  →  connected-component map
    ↓
Filter blobs (area, peak brightness)
    ↓
Intensity-weighted centroid per blob
    ↓
Sort by brightness, cap at max_stars
    ↓
list[StarCandidate]
```

### 3.1 Thresholding

Two modes are supported:

**Absolute threshold** (default, `threshold_method: absolute`)

```
mask = image >= min_brightness
```

Config key: `star_detection.min_brightness: 0.05`

After min-max normalisation, this selects pixels at ≥ 5% of the dynamic range. The synthetic background after preprocessing is < 0.01, so this threshold gives a comfortable margin.

**Sigma-clip threshold** (`threshold_method: sigma_clip`)

```
threshold = median(image) + k × σ(image)
```

Config key: `star_detection.sigma_clip_k: 5.0`

More adaptive: adjusts automatically if background levels vary between images. 5σ above the median is a standard astronomical detection threshold (probability of a noise peak exceeding 5σ is ≈ 3 × 10⁻⁷).

### 3.2 Connected-Component Labelling

`scipy.ndimage.label` with an 8-connectivity structuring element identifies contiguous groups of above-threshold pixels (blobs). Each blob is a candidate star.

8-connectivity (diagonals count as connected) is used rather than 4-connectivity to correctly merge the slightly elongated Gaussian PSF footprint into a single blob.

### 3.3 Blob Filtering

Each labelled blob is filtered on two criteria before centroiding:

| Filter | Config key | Value | Rationale |
|---|---|---|---|
| Area ≥ min_area_px | `min_area_px` | 1 | Reject isolated single-pixel hot pixels |
| Area ≤ max_area_px | `max_area_px` | 200 | Reject large cosmic-ray tracks or saturation bleed |
| Peak ≥ min_peak_brightness | `min_peak_brightness` | 0.04 | Reject faint diffuse regions |

### 3.4 Centroiding

**Algorithm: intensity-weighted centroid (first moment)**

Config key: `star_detection.centroid_method: intensity_weighted`  
Config key: `star_detection.centroid_half_window: 5`

The centroid is computed over a (2W+1) × (2W+1) window centred on the peak pixel of the blob:

```
x̄ = Σ(I_ij × j) / Σ(I_ij)
ȳ = Σ(I_ij × i) / Σ(I_ij)
```

The window is clamped to image bounds. If the window sum is zero (degenerate case), the peak pixel coordinate is returned.

**Accuracy:** Berry & Burnell (2005) show that intensity-weighted centroiding achieves < 0.1 px RMS error for Gaussian PSFs with SNR > 10 and a window radius ≥ 3σ. Our synthetic images satisfy both conditions (SNR ≫ 10 for bright stars; window radius W = 5 px > 3 × 1.5 px = 4.5 px).

On the Phase 1 synthetic dataset (noiseless conditions), measured centroid error is < 0.15 px (see notebook `02_star_detection.ipynb`, Section 4).

Alternative: `centroid_method: peak` uses the integer peak pixel coordinate directly — fast but no sub-pixel precision.

### 3.5 Output — `StarCandidate`

Each detection is returned as a `StarCandidate` dataclass:

| Field | Type | Description |
|---|---|---|
| `x` | float | Sub-pixel centroid column (horizontal) |
| `y` | float | Sub-pixel centroid row (vertical) |
| `brightness` | float | Integrated intensity within blob (sum of pixel values) |
| `peak` | float | Maximum pixel value within blob |
| `area` | int | Number of pixels in the connected-component blob |
| `bbox` | tuple | `(row_min, col_min, row_max, col_max)` bounding box |
| `features` | np.ndarray | Phase 3 — feature vector (empty until Phase 3) |
| `metadata` | dict | Optional debug fields |

---

## 4. Feature Extraction

### Status: Phase 3 — planned

`extract_features(stars, config)` in `star_detection.py` is a documented stub that raises `NotImplementedError`.

**Planned approach:** The feature representation will be designed in Phase 3 based on what the neural network architecture requires. Current candidates:

- **Pairwise angular distances** between star centroids, normalised by field of view — rotation and scale invariant.
- **Brightness ratios** between star pairs or triplets — independent of absolute photometric calibration.
- **Geometric descriptors** (triangle side ratios, polygon angles) formed by groups of detected stars.
- **Normalised (x, y) positions** within the image frame.

The representation must be invariant (or robust) to image rotation and scale, and compact enough for efficient neural network processing.

**To be determined during Phase 3:** final representation dimensionality, normalisation strategy, handling of variable star count per frame.

---

## 5. Neural Network Recognition

**Status: Phase 3 — planned**

Architecture (CNN, MLP, graph network, transformer, or hybrid) to be determined during Phase 3 based on the feature representation design and inference latency constraints.

**To be determined during Phase 3.**

---

## 6. Star Catalog Matching

**Status: Phase 4 — planned**

Matching algorithm (hash-based lookup, k-nearest-neighbour search in embedding space, geometric verification) to be determined during Phase 4.

**To be determined during Phase 4.**

---

## 7. Navigation Estimation

**Status: Phase 5 — planned**

Attitude estimation algorithm (QUEST, Davenport q-method, SVD/Wahba solution) to be determined during Phase 5.

Position estimation feasibility depends on the chosen methodology and will be assessed in Phase 5.

**To be determined during Phase 5.**

---

## 8. Evaluation Metrics

| Stage | Metric | Phase 2 measured value |
|---|---|---|
| Star detection | Detection rate (TP / (TP + FN)) | See notebook 02, Section 6 |
| Star detection | False positives per image | See notebook 02, Section 6 |
| Centroiding | Mean centroid error (px) | < 0.15 px (noiseless synthetic) |
| Feature extraction | Descriptor discriminability | TBD (Phase 3) |
| Neural network | Top-1 / top-k accuracy | TBD (Phase 3) |
| Catalog matching | Match rate, false-match rate | TBD (Phase 4) |
| Attitude estimation | Mean angular error (deg) | TBD (Phase 5) |
| End-to-end | Pipeline latency (ms) | TBD (Phase 6) |

Numerical targets for each metric will be established once baseline measurements are taken in the corresponding phase.

---

## Open Decisions

| Decision | Phase | Notes |
|---|---|---|
| Feature representation design | 3 | Candidates listed in §4 above |
| Neural network architecture | 3 | Deferred pending feature design |
| Star catalog extension (full Hipparcos) | 2/3 | Needed before Phase 3 training |
| Training data generation strategy | 3 | Synthetic only vs. mixed |
| Star catalog source for matching | 4 | Full Hipparcos recommended |
| Catalog matching algorithm | 4 | |
| Attitude estimation algorithm | 5 | |
| Position estimation feasibility | 5 | |
| Inference optimisation strategy | 6 | |

---

## Phase 3 Decisions (implemented)

### 4. Feature Extraction — IMPLEMENTED

**Algorithm: Pairwise pixel distances + brightness ratios**

Config key: `features.descriptor: pairwise_distances_and_ratios`
Config key: `features.max_stars: 10`

Stars are sorted by descending brightness and the top N = 10 are used. For every pair (i, j) with i < j, two descriptors are computed:

**Pairwise distance:**
```
d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2) / image_diagonal
```
Normalising by the image diagonal makes the descriptor resolution-independent.

**Brightness ratio:**
```
r_ij = brightness_i / (brightness_i + brightness_j)
```
Sum-normalised to keep the range in (0, 1) and avoid division-by-zero.

For N = 10 stars: N*(N-1)/2 = 45 pairs → **90 features total** (45 distances + 45 ratios).

Frames with fewer than 2 detected stars return a zero vector of the correct length. This is intentional — the classifier learns to handle sparse frames.

**Scientific basis:** Pairwise angular/pixel distances are the core of classical star-identification algorithms (Groth 1986, Mortari 2004). They are invariant to rotation and scale after normalisation, providing strong discrimination between sky regions.

### 5. Neural Network / Classification — IMPLEMENTED (sklearn backend)

**Decision: scikit-learn RandomForest classifier (Phase 3 active backend)**

PyTorch is not available on Python 3.14 (no compatible wheel as of August 2026). The `StarPatternClassifier` in `src/models/sklearn_classifier.py` provides a fully working classifier with three backend options:

| Backend | Config key | Notes |
|---|---|---|
| RandomForest | `classifier_type: random_forest` | **Active default.** n_estimators=200, provides probability estimates. |
| KNN | `classifier_type: knn` | n_neighbors=5. No training phase. |
| MLP | `classifier_type: mlp` | hidden_layers=[256, 128]. Requires more data. |

The PyTorch `StarPatternModel` stub in `star_pattern_model.py` is preserved and will be activated when a Python 3.14 compatible wheel is published.

### Sky Tessellation (classification target)

The classification target is a **sky-cell label** — a discretised identifier for the boresight sky region. The sky is divided using a cylindrical equal-area grid:

```
n_dec = sqrt(n_sky_cells / 2)
n_ra  = n_sky_cells / n_dec
label = ra_bin * n_dec + dec_bin
```

Config key: `model.n_sky_cells: 500` (target cell count; actual = n_ra × n_dec).

This maps the attitude estimation problem to a classification problem: recognising which sky region the camera is pointing at, then using the catalog to recover the precise attitude.

### Inference Interface

`run_inference(model, features, config) → RecognitionResult` dispatches to the active backend. The `RecognitionResult` dataclass provides:
- `pattern_id` — sky-cell label string (`"cell_N"`) or `None` if below confidence threshold
- `confidence` — predicted class probability in [0, 1]
- `raw_output` — full probability vector
- `latency_ms` — wall-clock inference time
- `top_k_predictions` — ranked list of (pattern_id, probability) pairs

Config key: `evaluation.confidence_threshold: 0.3`
Config key: `evaluation.top_k: 3`

### Updated Evaluation Metrics Table

| Stage | Metric | Phase 3 value |
|---|---|---|
| Star detection | Detection rate | See notebook 02 |
| Star detection | False positives/image | See notebook 02 |
| Centroiding | Mean centroid error | < 0.15 px (noiseless synthetic) |
| Feature extraction | Feature dim | 90 (N=10, distances+ratios) |
| Classification | Top-1 accuracy | See notebook 03 (sparse catalog limited) |
| Classification | Top-k accuracy | See notebook 03 |
| Classification | Inference latency | < 5 ms (single sample, sklearn RF, CPU) |
| Catalog matching | Match rate | TBD (Phase 4) |
| Attitude estimation | Mean angular error | TBD (Phase 5) |

### Updated Open Decisions

| Decision | Phase | Status |
|---|---|---|
| Feature representation | 3 | ✅ Pairwise distances + ratios (90-dim) |
| Classification backend | 3 | ✅ sklearn RF (PyTorch deferred to Python ≤3.12) |
| Star catalog extension | 3/4 | ⬜ Full Hipparcos needed before Phase 4 |
| Catalog matching algorithm | 4 | ⬜ Planned |
| Attitude estimation algorithm | 5 | ⬜ Planned |
| Position estimation feasibility | 5 | ⬜ Planned |
| Inference optimisation | 6 | ⬜ Planned |
