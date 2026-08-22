"""
feature_dataset.py
==================
Builds the feature matrix and label array used to train and evaluate the
:class:`~src.models.sklearn_classifier.StarPatternClassifier`.

Phase 3 implementation
-----------------------
For each generated :class:`~src.preprocessing.star_field_generator.SyntheticStarField`:

1. Apply the Phase 2 preprocessing pipeline (background subtraction,
   noise reduction, normalisation).
2. Run star detection to get a ``list[StarCandidate]``.
3. Call :func:`~src.preprocessing.star_detection.extract_features` to
   produce a fixed-length feature vector.
4. Convert the boresight (RA, Dec) to a sky-cell label using
   :func:`~src.models.sklearn_classifier.boresight_to_label`.
5. Accumulate feature vectors and labels into NumPy arrays.

Frames with fewer than 2 detected stars produce a zero feature vector and
are included in the dataset (the classifier learns to handle sparse frames).
Frames where detection fails entirely are skipped with a warning.

The function :func:`build_feature_dataset` is the primary entry point.
It can optionally save the dataset to ``data/processed/`` as NumPy ``.npz``
files for fast re-loading without re-running the full pipeline.

Reproducibility
---------------
All randomness is controlled by ``config["dataset"]["random_seed"]``.
The same seed always produces the same dataset.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from src.catalog.catalog_loader import load_catalog
from src.models.sklearn_classifier import boresight_to_label
from src.preprocessing.image_preprocessing import (
    subtract_background,
    reduce_noise,
    normalise,
)
from src.preprocessing.star_detection import detect_stars, extract_features
from src.preprocessing.star_field_generator import StarFieldGenerator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_dataset(
    config: dict,
    n_samples: int | None = None,
    split: str = "all",
    verbose: bool = True,
    save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate a feature matrix and label vector for classifier training.

    Parameters
    ----------
    config:
        Full project configuration dict (from config.yaml).
    n_samples:
        Total number of samples to generate.  If ``None``, uses
        ``num_train + num_val + num_test`` from ``config["dataset"]``.
    split:
        Which split to generate: ``"train"``, ``"val"``, ``"test"``,
        or ``"all"`` (default — generates all splits in order).
    verbose:
        If ``True``, prints progress to stdout.
    save_path:
        If given, saves ``X.npy``, ``y.npy``, and ``meta.json`` under
        this directory for fast re-loading.

    Returns
    -------
    X : np.ndarray
        Float32 feature matrix of shape (N, feature_dim).
    y : np.ndarray
        Integer label array of shape (N,) — sky-cell IDs.
    meta : list[dict]
        Per-sample metadata (boresight, seed, n_stars_gt, n_stars_detected).

    Raises
    ------
    FileNotFoundError
        If the catalog file does not exist.
    """
    t_start = time.time()
    ds_cfg   = config["dataset"]
    pp_cfg   = config.get("preprocessing", {})
    det_cfg  = config.get("star_detection", {})
    feat_cfg = config.get("features", {})
    model_cfg = config.get("model", {})

    n_sky_cells = int(model_cfg.get("n_sky_cells", 500))
    base_seed   = int(ds_cfg["random_seed"])

    # ── Determine how many samples to generate ────────────────────────────
    split_counts = {
        "train": int(ds_cfg["num_train"]),
        "val":   int(ds_cfg["num_val"]),
        "test":  int(ds_cfg["num_test"]),
    }

    if split == "all":
        splits_to_use = list(split_counts.items())
    elif split in split_counts:
        splits_to_use = [(split, split_counts[split])]
    else:
        raise ValueError(f"Unknown split '{split}'. Choose 'train', 'val', 'test', or 'all'.")

    if n_samples is not None:
        # Override — take first n_samples from the first split only
        splits_to_use = [(splits_to_use[0][0], min(n_samples, splits_to_use[0][1]))]

    # ── Catalog + generator ───────────────────────────────────────────────
    catalog = load_catalog(Path(ds_cfg["catalog_file"]), config)
    generator = StarFieldGenerator(catalog, ds_cfg)

    if verbose:
        print(f"[feature_dataset] Catalog: {catalog}")

    # ── Determine feature dimension ───────────────────────────────────────
    max_n    = int(feat_cfg.get("max_stars", 10))
    n_pairs  = max_n * (max_n - 1) // 2
    use_ratios = feat_cfg.get("descriptor", "pairwise_distances_and_ratios") \
                 == "pairwise_distances_and_ratios"
    feature_dim = 2 * n_pairs if use_ratios else n_pairs

    # ── Main generation loop ──────────────────────────────────────────────
    X_list:    list[np.ndarray] = []
    y_list:    list[int]        = []
    meta_list: list[dict]       = []

    global_index = 0

    for split_name, count in splits_to_use:
        if verbose:
            print(f"[feature_dataset] Processing '{split_name}' ({count} samples)…")

        for local_i in range(count):
            seed = base_seed * 100_000 + global_index
            global_index += 1

            # Generate star field
            sf = generator.generate(seed=seed)

            # Preprocessing (in-memory — no disk I/O needed here)
            try:
                img = subtract_background(
                    sf.image,
                    method=pp_cfg.get("background_method", "median_filter"),
                    filter_size=int(pp_cfg.get("background_filter_size", 31)),
                )
                img = reduce_noise(
                    img,
                    method=pp_cfg.get("noise_method", "gaussian"),
                    sigma=float(pp_cfg.get("noise_sigma", 0.8)),
                )
                norm_method = pp_cfg.get("normalization", "min_max")
                if norm_method and norm_method != "none":
                    img = normalise(img, method=norm_method)
            except Exception as exc:
                warnings.warn(
                    f"Preprocessing failed for seed={seed}: {exc}. Skipping.",
                    stacklevel=2,
                )
                continue

            # Star detection
            try:
                stars = detect_stars(img, det_cfg)
            except Exception as exc:
                warnings.warn(
                    f"Detection failed for seed={seed}: {exc}. Skipping.",
                    stacklevel=2,
                )
                continue

            # Feature extraction
            feat = extract_features(stars, config)
            if feat.shape[0] != feature_dim:
                # Dimension mismatch — pad or truncate to expected length
                vec = np.zeros(feature_dim, dtype=np.float32)
                vec[: min(len(feat), feature_dim)] = feat[: feature_dim]
                feat = vec

            # Label
            label = boresight_to_label(
                sf.boresight_ra_deg,
                sf.boresight_dec_deg,
                n_sky_cells=n_sky_cells,
            )

            X_list.append(feat)
            y_list.append(label)
            meta_list.append({
                "seed":              seed,
                "split":             split_name,
                "boresight_ra_deg":  round(sf.boresight_ra_deg, 4),
                "boresight_dec_deg": round(sf.boresight_dec_deg, 4),
                "roll_deg":          round(sf.roll_deg, 4),
                "n_stars_gt":        len(sf.stars),
                "n_stars_detected":  len(stars),
                "label":             label,
            })

            if verbose and (local_i % max(1, count // 5) == 0):
                print(
                    f"  [{split_name}] {local_i + 1}/{count}  "
                    f"seed={seed}  gt={len(sf.stars)}  "
                    f"det={len(stars)}  label={label}"
                )

    X = np.array(X_list, dtype=np.float32) if X_list else np.zeros((0, feature_dim), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    elapsed = time.time() - t_start
    if verbose:
        n_unique = len(np.unique(y)) if len(y) > 0 else 0
        print(
            f"\n[feature_dataset] Done: {len(X)} samples, "
            f"{n_unique} unique classes, "
            f"{feature_dim} features, "
            f"{elapsed:.1f}s ({len(X) / elapsed:.0f} samples/s)"
        )

    # ── Optional save ──────────────────────────────────────────────────────
    if save_path is not None:
        import json
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "X.npy", X)
        np.save(save_path / "y.npy", y)
        with open(save_path / "meta.json", "w") as fh:
            json.dump(meta_list, fh, indent=2)
        if verbose:
            print(f"[feature_dataset] Saved to {save_path.resolve()}")

    return X, y, meta_list


def load_feature_dataset(
    load_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load a previously saved feature dataset from *load_path*.

    Parameters
    ----------
    load_path:
        Directory containing ``X.npy``, ``y.npy``, and optionally
        ``meta.json`` (produced by :func:`build_feature_dataset`).

    Returns
    -------
    X, y, meta
        Same types as :func:`build_feature_dataset`.

    Raises
    ------
    FileNotFoundError
        If the required ``.npy`` files do not exist.
    """
    import json
    load_path = Path(load_path)

    x_path = load_path / "X.npy"
    y_path = load_path / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Feature dataset files not found in {load_path}. "
            "Run build_feature_dataset() first."
        )

    X = np.load(x_path)
    y = np.load(y_path)

    meta_path = load_path / "meta.json"
    meta: list[dict] = []
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)

    return X, y, meta
