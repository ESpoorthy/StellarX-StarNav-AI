# Dataset Documentation

> **Phase 1 — Complete.**
> This document reflects the implemented Phase 1 data pipeline.
> Sections marked **Phase 2+** describe planned future work.

---

## 1. Overview

StellarX-StarNav-AI uses **synthetic star-field images** as its primary training and evaluation data source. These images are generated programmatically from a real published star catalog using a simplified spacecraft star-sensor simulation. No real spacecraft imagery has been used in Phase 1.

The synthetic approach was chosen because:

- Labeled real spacecraft star-sensor imagery with accurate ground-truth attitude is rare and often restricted.
- Synthetic generation gives full control over ground-truth (exact star positions, magnitudes, attitude).
- Generation is fully reproducible given a random seed.
- The pipeline can be scaled to any dataset size without data collection effort.

All generated data is clearly labeled as **Synthetic Star-Field Data** throughout the codebase and documentation.

---

## 2. Star Catalog

### 2.1 Source

| Field | Value |
|---|---|
| Catalog name | Hipparcos Catalogue |
| Publication | ESA SP-1200, 1997 |
| Mission | Hipparcos satellite, ESA, 1989–1993 |
| URL | https://www.cosmos.esa.int/web/hipparcos/catalogues |
| License | Public domain for scientific/educational use |
| Coordinate epoch | J2000.0 ICRS |

### 2.2 Phase 1 Subset

For the Phase 1 prototype, a curated subset of **50 bright stars** is bundled directly in the repository at:

```
data/catalog/hipparcos_bright.csv
```

This subset covers stars with V magnitude roughly ≤ 2.1 (the exact faint limit in the working CSV), which are the most reliably detectable stars in a simplified sensor simulation. The mag_limit filter in `config.yaml` (`catalog_mag_limit: 6.5`) applies at load time and can be tightened to select only the very brightest stars.

**Why 50 stars?** The prototype catalog is intentionally small for fast development and iteration. It is sufficient to generate synthetic frames with a small number of stars, validate the full pipeline end-to-end, and test the detection and matching logic. Extending to the full Hipparcos catalog (~118,218 stars) is the first recommended step before Phase 2.

### 2.3 Catalog File Format

```
data/catalog/hipparcos_bright.csv
```

Lines beginning with `#` are comments and are skipped by the loader.

| Column | Type | Units | Description |
|---|---|---|---|
| `hip_id` | int | — | Hipparcos catalogue number |
| `ra_deg` | float | degrees | Right ascension, J2000.0 ICRS |
| `dec_deg` | float | degrees | Declination, J2000.0 ICRS |
| `vmag` | float | mag | Johnson V-band apparent magnitude |
| `spectral_type` | str | — | MK spectral classification |
| `common_name` | str | — | Common name (empty string if none) |

The loader (`src/catalog/catalog_loader.py`) maps these to a `CatalogStar` dataclass with `star_id = "HIP_<hip_id>"`.

### 2.4 Coordinate System

- Right ascension and declination are in **degrees, J2000.0 ICRS**.
- The `CatalogStar.unit_vector()` method converts (RA, Dec) to a 3-D unit vector using the standard astronomical convention:

  ```
  x = cos(dec) * cos(ra)
  y = cos(dec) * sin(ra)
  z = sin(dec)
  ```

### 2.5 Extending the Catalog

To extend to the full Hipparcos catalog:

1. Download `hip_main.dat` from CDS VizieR (catalogue I/239): https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=I/239
2. Run (Phase 1+):
   ```bash
   python src/catalog/catalog_downloader.py  # to be implemented in Phase 2
   ```

Alternatively, replace `data/catalog/hipparcos_bright.csv` with any CSV file that contains at minimum the four required columns: `hip_id`, `ra_deg`, `dec_deg`, `vmag`.

---

## 3. Synthetic Star-Field Generation

### 3.1 Simulation Model

The generator (`src/preprocessing/star_field_generator.py`) simulates a simplified spacecraft star tracker. The simulation is a deliberate approximation — it captures the essential characteristics of star-sensor imagery without modelling physical effects that are not yet needed for Phase 1.

**Pipeline:**

```
Random attitude (RA, Dec, roll)
        ↓
Select catalog stars within FoV (half-angle cone query)
        ↓
Gnomonic projection → pixel coordinates
        ↓
Magnitude → linear flux (Pogson's law)
        ↓
Render 2-D isotropic Gaussian PSF per star
        ↓
Add constant background
        ↓
Add Gaussian shot noise (approx.) + read noise
        ↓
Add random hot-pixel / cosmic-ray artifacts (optional)
        ↓
Clip to [0, 1] → float32 image
```

### 3.2 Attitude Sampling

- Boresight (RA, Dec) is sampled **uniformly on the unit sphere**: `Dec = arcsin(U(-1, 1))`, `RA = U(0, 360)`.
- Roll angle is sampled uniformly in [0°, 360°).
- Fixed boresight and roll can be passed to override random sampling (for testing).

### 3.3 Star Projection

Stars are projected onto the focal plane using the **gnomonic (tangent-plane) projection**:

```
cos_c = sin(dec_c)*sin(dec) + cos(dec_c)*cos(dec)*cos(ra - ra_c)
x_tan = cos(dec)*sin(ra - ra_c) / cos_c
y_tan = (cos(dec_c)*sin(dec) - sin(dec_c)*cos(dec)*cos(ra - ra_c)) / cos_c
```

The focal length in pixels is derived from the configured field of view:

```
f_px = (image_width / 2) / tan(fov_rad / 2)
```

Roll is applied as a 2-D rotation in the tangent plane before pixel coordinate conversion.

**Approximation note:** Gnomonic projection is accurate to within ~0.1% for angular offsets up to ~10° from the boresight. The default 20° full FoV keeps this well within acceptable limits for prototype use.

### 3.4 Magnitude to Flux Conversion

Linear flux is derived from V magnitude using **Pogson's law**:

```
flux = 10^(-0.4 * (vmag - vmag_ref))
```

where `vmag_ref = 0.0`. All fluxes are then **normalised** so the brightest star in the frame has flux = 1.0. Stars below the `min_star_flux` threshold are discarded before rendering.

### 3.5 Point Spread Function (PSF)

Each star is rendered as a **2-D isotropic Gaussian** with sigma `psf_sigma_px` (configurable). Only the bounding box within radius 4σ is updated — stars outside the image boundary are silently skipped.

**Approximation note:** Real star-tracker PSFs are not perfectly Gaussian and may be asymmetric, wavelength-dependent, or defocus-distorted. This is adequate for Phase 1.

### 3.6 Noise Model

| Component | Model | Config key |
|---|---|---|
| Background | Constant additive offset | `background_level` |
| Shot noise | Gaussian approximation: σ = sqrt(signal) × 0.05 | `shot_noise` |
| Read noise | Zero-mean Gaussian | `read_noise_sigma` |
| Hot pixels / cosmic rays | Poisson count of events, random positions | `artifact_probability` |

**Approximation note:** True photon shot noise follows a Poisson distribution. The Gaussian approximation used here is valid when the expected photon count is large (signal >> 1). For very faint stars this overestimates the noise symmetry but is acceptable at this stage.

### 3.7 Reproducibility

Every call to `generator.generate(seed=N)` produces **identical** output for the same seed and catalog. The seed for sample `i` (counting across all splits in train→val→test order) is:

```
seed_i = base_seed × 100_000 + i
```

This ensures seeds are unique, far apart in the random sequence, and stable across split-size changes (as long as `base_seed` and `i` are unchanged).

---

## 4. Dataset Output

### 4.1 File Layout

```
data/raw/
├── train/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...  (num_train files)
├── val/
│   └── ...  (num_val files)
├── test/
│   └── ...  (num_test files)
└── metadata.json
```

All image files are **16-bit greyscale PNG** (lossless). The 16-bit depth preserves the full dynamic range of the float32 simulation output (quantisation error < 0.002).

### 4.2 Image Format

| Property | Value |
|---|---|
| Format | PNG (lossless) |
| Bit depth | 16-bit greyscale |
| Pixel value range | 0–65535 (maps to [0.0, 1.0] after normalisation) |
| Dimensions | 512 × 512 px (Phase 1 default, configurable) |
| Colour mode | Greyscale (single channel) |

When loaded via `src.preprocessing.image_preprocessing.load_image()`, the image is returned as a **float32 NumPy array in [0, 1]**.

### 4.3 Metadata Schema

All ground-truth information is stored in a single `metadata.json` file. Each entry corresponds to one image.

**Top-level fields:**

| Field | Type | Units | Description |
|---|---|---|---|
| `sample_id` | str | — | Unique identifier, e.g. `"train_000042"` |
| `split` | str | — | `"train"` \| `"val"` \| `"test"` |
| `image_file` | str | — | Relative path from `output_dir`, e.g. `"train/000042.png"` |
| `seed` | int | — | Random seed used to generate this image |
| `image_width` | int | px | Image width in pixels |
| `image_height` | int | px | Image height in pixels |
| `fov_deg` | float | deg | Full field-of-view used for this image |
| `boresight_ra_deg` | float | deg | Camera boresight right ascension (J2000 ICRS) |
| `boresight_dec_deg` | float | deg | Camera boresight declination (J2000 ICRS) |
| `roll_deg` | float | deg | Camera roll angle around the boresight |
| `n_stars` | int | — | Number of stars rendered into this image |
| `stars` | list | — | Per-star ground-truth (see below) |

**Per-star fields (`stars[]`):**

| Field | Type | Units | Description |
|---|---|---|---|
| `star_id` | str | — | Hipparcos identifier, `"HIP_<n>"` |
| `x_px` | float | px | Star centroid column (horizontal), sub-pixel |
| `y_px` | float | px | Star centroid row (vertical), sub-pixel |
| `flux` | float | — | Normalised flux in (0, 1]; 1.0 = brightest in frame |
| `vmag` | float | mag | Johnson V-band apparent magnitude from catalog |
| `ra_deg` | float | deg | Catalog right ascension (J2000 ICRS) |
| `dec_deg` | float | deg | Catalog declination (J2000 ICRS) |

**Example entry:**

```json
{
  "sample_id": "train_000000",
  "split": "train",
  "image_file": "train/000000.png",
  "seed": 4200000,
  "image_width": 512,
  "image_height": 512,
  "fov_deg": 20.0,
  "boresight_ra_deg": 237.541,
  "boresight_dec_deg": -14.832,
  "roll_deg": 312.7,
  "n_stars": 3,
  "stars": [
    {
      "star_id": "HIP_32349",
      "x_px": 263.4,
      "y_px": 198.7,
      "flux": 1.0,
      "vmag": -1.46,
      "ra_deg": 101.287,
      "dec_deg": -16.716
    }
  ]
}
```

---

## 5. Train / Validation / Test Split

| Split | Count (Phase 1 default) | Purpose |
|---|---|---|
| `train` | 800 | Model training |
| `val` | 100 | Hyperparameter tuning and early stopping |
| `test` | 100 | Final held-out evaluation |

**Split strategy:** Splits are defined purely by seed range. Because the boresight is sampled uniformly on the sphere, splits are naturally independent — there is no concept of "similar orientations" that could leak between splits. All 1,000 samples use distinct seeds.

The held-out test set must not be used during model development. It is reserved for the final Phase 3/4 evaluation reported in `docs/results.md`.

---

## 6. Data Versioning and Storage

- `data/raw/` is listed in `.gitignore` — generated images are **never committed to the repository**.
- `data/processed/` is also ignored — future preprocessed arrays go here.
- `data/catalog/hipparcos_bright.csv` **is** committed — it is a small static reference file.
- `data/raw/metadata.json` is ignored (it is generated alongside the images).
- Dataset version is tracked via the `random_seed` and split counts in `config.yaml`. Any change to these values produces a different dataset; increment the project `version` field in `config.yaml` when doing so.

---

## 7. Reproducibility

To regenerate the full dataset from scratch:

```bash
# From the repository root
python -c "
import yaml
from src.preprocessing.dataset_builder import build_dataset
with open('config.yaml') as f:
    config = yaml.safe_load(f)
build_dataset(config)
"
```

The same command with the same `config.yaml` will always produce byte-identical images.

---

## 8. Known Limitations

| Limitation | Impact | Planned Mitigation |
|---|---|---|
| Only 50 stars in prototype catalog | Many boresight pointings yield 0–1 visible stars; unrealistic star density | Extend to full Hipparcos catalog (~118K stars) before Phase 2 |
| Isotropic Gaussian PSF | Does not match real sensor optics (asymmetry, wavelength, focus) | Model refinement in Phase 2+ |
| Gaussian shot noise approximation | Slightly incorrect for very faint stars | Use true Poisson sampling in Phase 2 |
| No atmospheric refraction | Not applicable for spacecraft (space environment) | N/A |
| No proper motion or parallax | Negligible for star-tracker application at this accuracy level | Document assumption in Phase 3 evaluation |
| No vignetting or flat-field variation | Real sensors have non-uniform response | Add in Phase 2 if star detection accuracy is affected |
| All images are single-exposure | No time-series or multi-exposure stacking | Out of scope for Phase 1 |
| No real spacecraft imagery | Synthetic-to-real transfer gap unknown | Validate against real data in Phase 5+ |

---

## 9. Phase 2+ Extensions

The following extensions are planned and documented here for traceability:

- **Full Hipparcos catalog**: replace `hipparcos_bright.csv` with a full catalog download (see §2.5).
- **FITS support**: implement FITS image loading in `image_preprocessing.load_image()` using `astropy.io.fits` for compatibility with real astronomical data.
- **Realistic PSF**: replace the Gaussian PSF with a measured or parameterised sensor PSF.
- **True Poisson noise**: replace the Gaussian shot-noise approximation.
- **Vignetting and flat-field**: add spatially varying response.
- **Real-data augmentation**: mix synthetic and real images during training once real data is available.
