"""
dataset_builder.py
==================
Generates a full labelled synthetic dataset of star-field images and writes
ground-truth metadata to disk.

Responsibility
--------------
1. Load the star catalog via :func:`~src.catalog.catalog_loader.load_catalog`.
2. Instantiate a :class:`~src.preprocessing.star_field_generator.StarFieldGenerator`.
3. Generate ``num_train + num_val + num_test`` images with unique, reproducible
   seeds derived from the global ``random_seed`` in config.
4. Save each image as a lossless PNG (or TIFF) under ``data/raw/<split>/``.
5. Write a single ``metadata.json`` file alongside the images that records
   ground-truth information for every sample.
6. Print a short progress summary to stdout.

Output layout
-------------
::

    data/raw/
    ├── train/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── val/
    │   └── ...
    ├── test/
    │   └── ...
    └── metadata.json          ← one JSON array, all splits combined

Metadata schema
---------------
Each entry in ``metadata.json`` is a JSON object with these fields:

.. code-block:: json

    {
        "sample_id":          "train_000000",
        "split":              "train",
        "image_file":         "train/000000.png",
        "seed":               42,
        "image_width":        512,
        "image_height":       512,
        "fov_deg":            20.0,
        "boresight_ra_deg":   123.456,
        "boresight_dec_deg":  -34.567,
        "roll_deg":           78.9,
        "n_stars":            12,
        "stars": [
            {
                "star_id":   "HIP_32349",
                "x_px":      256.3,
                "y_px":      198.7,
                "flux":      1.0,
                "vmag":      -1.46,
                "ra_deg":    101.287,
                "dec_deg":   -16.716
            }
        ]
    }

Reproducibility
---------------
The seed for sample ``i`` in split ``split_name`` is deterministic::

    seed_i = base_seed * 100_000 + global_index

where ``global_index`` increments across train → val → test in order.
This means seeds are stable even if split sizes change, as long as the
base seed and total count do not conflict.

All parameters are sourced from ``config["dataset"]``.  No paths or counts
are hard-coded in this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.catalog.catalog_loader import load_catalog
from src.preprocessing.star_field_generator import StarFieldGenerator, SyntheticStarField


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dataset(config: dict, verbose: bool = True) -> dict[str, Any]:
    """Generate the full synthetic dataset and save to disk.

    Parameters
    ----------
    config:
        Full project configuration dict (loaded from config.yaml).
        The ``dataset`` sub-dict drives all parameters.
    verbose:
        If ``True``, print progress to stdout.

    Returns
    -------
    dict
        Summary dict with keys ``n_train``, ``n_val``, ``n_test``,
        ``output_dir``, ``metadata_file``, ``elapsed_sec``.

    Raises
    ------
    FileNotFoundError
        If the catalog file specified in config does not exist.
    """
    t_start = time.time()
    ds_cfg = config["dataset"]

    # ── Paths ──────────────────────────────────────────────────────────
    output_dir = Path(ds_cfg["output_dir"])
    metadata_path = Path(ds_cfg["metadata_file"])
    catalog_path = Path(ds_cfg["catalog_file"])

    # ── Splits ─────────────────────────────────────────────────────────
    split_counts = {
        "train": int(ds_cfg["num_train"]),
        "val":   int(ds_cfg["num_val"]),
        "test":  int(ds_cfg["num_test"]),
    }
    base_seed = int(ds_cfg["random_seed"])

    # ── Catalog ─────────────────────────────────────────────────────────
    catalog = load_catalog(catalog_path, config)
    if verbose:
        print(f"[dataset_builder] Catalog loaded: {catalog}")

    # ── Generator ───────────────────────────────────────────────────────
    generator = StarFieldGenerator(catalog, ds_cfg)

    # ── Create split directories ─────────────────────────────────────────
    for split in split_counts:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # ── Generate images ──────────────────────────────────────────────────
    all_metadata: list[dict] = []
    global_index = 0

    for split, count in split_counts.items():
        if verbose:
            print(f"[dataset_builder] Generating {count} '{split}' images …")

        for local_index in range(count):
            seed = base_seed * 100_000 + global_index

            star_field: SyntheticStarField = generator.generate(seed=seed)

            # Build filename and paths
            filename = f"{local_index:06d}.png"
            rel_path = f"{split}/{filename}"
            abs_path = output_dir / split / filename

            # Save image
            _save_image(star_field.image, abs_path)

            # Build metadata entry
            entry = _build_metadata_entry(star_field, split, rel_path, seed)
            all_metadata.append(entry)

            global_index += 1

            if verbose and (local_index % max(1, count // 10) == 0):
                print(
                    f"  [{split}] {local_index + 1}/{count} — "
                    f"seed={seed}, n_stars={entry['n_stars']}"
                )

    # ── Write metadata JSON ───────────────────────────────────────────────
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(all_metadata, fh, indent=2)

    elapsed = time.time() - t_start
    total = sum(split_counts.values())

    if verbose:
        print(
            f"\n[dataset_builder] Done. "
            f"{total} images in {elapsed:.1f}s "
            f"({total / elapsed:.0f} img/s)"
        )
        print(f"  Output dir    : {output_dir.resolve()}")
        print(f"  Metadata file : {metadata_path.resolve()}")

    return {
        "n_train": split_counts["train"],
        "n_val":   split_counts["val"],
        "n_test":  split_counts["test"],
        "output_dir": str(output_dir.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "elapsed_sec": elapsed,
    }


def load_metadata(metadata_path: str | Path) -> list[dict]:
    """Load a previously generated ``metadata.json`` file.

    Parameters
    ----------
    metadata_path:
        Path to the metadata JSON file produced by :func:`build_dataset`.

    Returns
    -------
    list[dict]
        List of metadata entries, one per generated image.

    Raises
    ------
    FileNotFoundError
        If the metadata file does not exist.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_image(image: np.ndarray, path: Path) -> None:
    """Save a float32 image array to disk as a lossless 16-bit greyscale PNG.

    The array is scaled from [0, 1] to [0, 65535] and saved using OpenCV
    which writes 16-bit greyscale PNG without deprecation issues.

    Parameters
    ----------
    image:
        Float32 array of shape (H, W) with values in [0, 1].
    path:
        Destination file path.  Parent directory must already exist.
    """
    import cv2
    img_uint16 = (np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), img_uint16)


def _build_metadata_entry(
    sf: SyntheticStarField,
    split: str,
    rel_path: str,
    seed: int,
) -> dict:
    """Convert a :class:`SyntheticStarField` into a serialisable metadata dict.

    Parameters
    ----------
    sf:
        Generated star-field result.
    split:
        Dataset split name (``"train"``, ``"val"``, ``"test"``).
    rel_path:
        Relative path to the saved image file from ``output_dir``.
    seed:
        Seed used to generate this image.

    Returns
    -------
    dict
        JSON-serialisable metadata entry.
    """
    sample_id = f"{split}_{rel_path.split('/')[-1].replace('.png', '')}"
    stars_list = [
        {
            "star_id":  s.star_id,
            "x_px":     round(s.x_px, 4),
            "y_px":     round(s.y_px, 4),
            "flux":     round(s.flux, 6),
            "vmag":     round(s.vmag, 3),
            "ra_deg":   round(s.ra_deg, 6),
            "dec_deg":  round(s.dec_deg, 6),
        }
        for s in sf.stars
    ]
    return {
        "sample_id":          sample_id,
        "split":              split,
        "image_file":         rel_path,
        "seed":               seed,
        "image_width":        sf.image_width,
        "image_height":       sf.image_height,
        "fov_deg":            sf.fov_deg,
        "boresight_ra_deg":   round(sf.boresight_ra_deg, 6),
        "boresight_dec_deg":  round(sf.boresight_dec_deg, 6),
        "roll_deg":           round(sf.roll_deg, 4),
        "n_stars":            len(sf.stars),
        "stars":              stars_list,
    }
