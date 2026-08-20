# Methodology

> **Status:** Intended methodology — implementation decisions are still open.
> Sections marked **To be determined during implementation** will be completed as each phase progresses.

---

## Overview

This document describes the intended technical approach for each stage of the StellarX-StarNav-AI pipeline. It serves as a living specification that bridges the high-level architecture description and the concrete implementation choices made during development.

---

## 1. Input Star-Field Imagery

The system operates on images of the star field captured by an onboard imaging device. These images are expected to be:

- Greyscale or single-channel intensity images (colour support to be evaluated)
- Captured under low-light, space-environment conditions
- Subject to sensor noise, cosmic-ray events, and stray-light artifacts

**Image format and resolution:** To be determined during dataset preparation (Phase 1).

**Supported input sources:** Real spacecraft imagery, synthetic simulations, or publicly available astronomical datasets — to be determined during Phase 1.

---

## 2. Image Preprocessing

The goal of preprocessing is to transform the raw sensor image into a clean, normalised representation suitable for reliable star detection.

### 2.1 Background Estimation and Subtraction

Space imagery typically contains a slowly varying background illumination component from scattered light, sensor bias, and dark current. A background estimation method will be applied to separate the background from point-source objects.

**Algorithm:** To be determined during implementation.

### 2.2 Noise Reduction

Point-spread noise and read noise from the sensor must be suppressed without smearing the point-source star signals.

**Algorithm:** To be determined during implementation. Candidates include Gaussian smoothing, median filtering, and wavelet-based denoising.

### 2.3 Intensity Normalisation

Pixel intensities will be normalised to a consistent range to ensure stable neural network input and to allow comparison across images from different sensors or exposures.

**Method (e.g. min-max, z-score, percentile clipping):** To be determined during implementation.

---

## 3. Star Detection

The star detection step identifies pixel regions in the preprocessed image that correspond to individual stars.

### 3.1 Detection Approach

Stars appear as point-source objects — compact, approximately Gaussian intensity profiles above the background level. Detection involves:
1. Applying an intensity threshold to identify candidate regions
2. Localising the centroid of each candidate
3. Filtering candidates by size, brightness, and morphology

**Detection algorithm (e.g. blob detection, connected-component analysis, matched filter):** To be determined during Phase 2.

### 3.2 Centroiding

Subpixel centroid estimation improves the precision of star position measurements. A centroiding method will be selected to balance accuracy and computational cost.

**Centroiding method:** To be determined during implementation.

### 3.3 Output

Each detected star is represented as a structured record containing:
- Centroid coordinates (x, y) in image space
- Integrated brightness estimate
- (Optional) additional morphological descriptors

---

## 4. Feature Extraction

The feature extraction step constructs a compact, discriminative representation of the observed star field from the list of detected stars. This representation is used as input to the neural network.

Potential feature types include:
- Pairwise angular distances between stars
- Brightness ratios between star pairs or triplets
- Geometric patterns formed by star groupings (triangles, polygons)
- Normalised positional descriptors

**Feature representation design and dimensionality:** To be determined during Phase 3.

The representation must be:
- Invariant (or robust) to image rotation and scale
- Compact enough for efficient neural network inference
- Discriminative enough to distinguish similar-looking patterns

---

## 5. Neural Network-Based Recognition

The neural network is the central classification component. It takes the extracted feature vector as input and outputs either a class label corresponding to a known star pattern, or a similarity score for retrieval-based matching.

### 5.1 Architecture

**Architecture (e.g. fully-connected MLP, CNN operating on feature maps, graph neural network, transformer):** To be determined during Phase 3.

The choice of architecture will be guided by:
- The dimensionality and structure of the input features
- The required classification accuracy
- Inference latency and memory constraints for the target platform

### 5.2 Training Data

The model requires labeled training examples pairing observed star-field features with known catalog identities. Training data may be sourced from:
- Synthetic simulations of star fields generated from a catalog
- Real imagery with known ground-truth orientations (if available)
- Data augmentation to improve robustness

**Training data strategy:** To be determined during Phase 3.

### 5.3 Loss Function and Optimisation

**Loss function:** To be determined during implementation.

**Optimiser and learning-rate schedule:** To be determined during implementation.

### 5.4 Evaluation

The model will be evaluated on a held-out test set using metrics including:
- Top-1 and top-k classification accuracy
- Precision and recall
- Confusion matrix analysis
- Inference latency on target hardware

---

## 6. Star Catalog Matching

The output of the neural network identifies a candidate star pattern. The catalog matching step verifies and refines this identification by comparing the candidate against entries in a stored star catalog.

### 6.1 Catalog

The star catalog is a reference database of known stars with accurate celestial coordinates (right ascension, declination), apparent magnitudes, and identifiers.

**Catalog source and format:** To be determined during Phase 1 dataset preparation.

### 6.2 Matching Algorithm

**Matching algorithm (e.g. hash-based lookup, k-nearest-neighbor search, geometric verification):** To be determined during Phase 4.

### 6.3 Confidence Estimation

Each match will be accompanied by a confidence score reflecting the reliability of the identification. A minimum confidence threshold (configurable via `config.yaml`) will gate the output to avoid low-quality navigation estimates.

**Confidence model:** To be determined during implementation.

---

## 7. Navigation Estimation

Given a verified catalog match, the navigation estimation step computes the spacecraft's attitude and, where the methodology supports it, its position.

### 7.1 Attitude Estimation

Spacecraft attitude describes its orientation in inertial space (typically expressed as a quaternion or as Euler angles). Attitude is computed from the correspondence between observed star directions (in the camera frame) and catalog star directions (in the inertial frame).

**Estimation algorithm (e.g. QUEST, Davenport q-method, SVD-based solution):** To be determined during Phase 5.

### 7.2 Position Estimation

Position estimation from star imagery alone is generally not possible without additional information (e.g. knowledge of observable star magnitudes as a function of distance, or multi-camera parallax). Whether and how position estimation is supported will depend on the chosen methodology.

**Position estimation approach:** To be determined during Phase 5.

---

## 8. Evaluation Metrics

The following metrics will be used to evaluate system performance across pipeline stages:

| Stage | Metrics |
|---|---|
| Star detection | Detection rate, false-positive rate, centroid accuracy |
| Feature extraction | Descriptor discriminability, sensitivity analysis |
| Neural network | Top-1 / top-k accuracy, precision, recall, F1 |
| Catalog matching | Match rate, false-match rate, confidence calibration |
| Attitude estimation | Angular error (degrees), statistical distribution |
| Position estimation | Position error (km or arc-minutes, if applicable) |
| End-to-end | Pipeline latency, memory usage, accuracy on test set |

Specific numerical targets for each metric are to be established during implementation once baseline performance is characterised.

---

## Open Questions

The following decisions remain open and will be resolved during the corresponding implementation phases:

- Image format, resolution, and sensor model (Phase 1)
- Preprocessing algorithms for noise and background (Phase 2)
- Star detection algorithm and centroiding method (Phase 2)
- Feature representation design (Phase 3)
- Neural network architecture (Phase 3)
- Training data generation strategy (Phase 3)
- Star catalog source and format (Phase 1/4)
- Catalog matching algorithm (Phase 4)
- Attitude estimation algorithm (Phase 5)
- Position estimation feasibility and approach (Phase 5)
