"""
test_star_field_generator.py
============================
Unit tests for:
  - src.preprocessing.star_field_generator
  - src.preprocessing.image_preprocessing  (load_image)
  - src.preprocessing.dataset_builder      (build_dataset, load_metadata)

Covers:
  - Pure function correctness (_gnomonic_project, _mag_to_flux,
    _render_gaussian_star, _random_boresight)
  - StarFieldGenerator.generate() output shape, dtype, value range
  - Determinism: same seed → identical image and star list
  - Distinctness: different seeds → different images
  - Stars stay within image boundaries
  - Ground-truth metadata is consistent with the image
  - load_image round-trip (save → load → compare)
  - build_dataset produces the right file count and metadata file
  - load_metadata reads back all entries correctly
  - Dataset split counts are respected

Run with:
    pytest tests/test_star_field_generator.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.catalog.catalog_loader import load_catalog, StarCatalog
from src.preprocessing.star_field_generator import (
    StarFieldGenerator,
    SyntheticStarField,
    _gnomonic_project,
    _mag_to_flux,
    _render_gaussian_star,
    _random_boresight,
)
from src.preprocessing.image_preprocessing import load_image
from src.preprocessing.dataset_builder import build_dataset, load_metadata


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CATALOG_PATH = Path("data/catalog/hipparcos_bright.csv")

MINIMAL_CONFIG = {
    "dataset": {
        "catalog_file": str(CATALOG_PATH),
        "catalog_mag_limit": 6.5,
        "image_width": 128,
        "image_height": 128,
        "field_of_view_deg": 20.0,
        "max_stars_per_image": 20,
        "psf_sigma_px": 1.5,
        "min_star_flux": 0.02,
        "background_level": 0.02,
        "read_noise_sigma": 0.005,
        "shot_noise": True,
        "artifact_probability": 0.0,   # determinism: no random artifacts
        "num_train": 4,
        "num_val": 2,
        "num_test": 2,
        "random_seed": 42,
        "output_dir": "",               # overridden per test via tmp_path
        "metadata_file": "",            # overridden per test via tmp_path
        "image_format": "png",
    }
}


@pytest.fixture(scope="module")
def catalog() -> StarCatalog:
    return load_catalog(CATALOG_PATH)


@pytest.fixture(scope="module")
def generator(catalog) -> StarFieldGenerator:
    return StarFieldGenerator(catalog, MINIMAL_CONFIG["dataset"])


@pytest.fixture(scope="module")
def star_field(generator) -> SyntheticStarField:
    """A single generated star field, shared across tests in this module."""
    return generator.generate(seed=42)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestGnomonicProject:
    def test_boresight_projects_to_origin(self):
        x, y = _gnomonic_project(45.0, 30.0, 45.0, 30.0)
        assert math.isclose(x, 0.0, abs_tol=1e-9)
        assert math.isclose(y, 0.0, abs_tol=1e-9)

    def test_behind_plane_returns_nan(self):
        """A point 180° away is behind the tangent plane — must return nan."""
        x, y = _gnomonic_project(180.0, 0.0, 0.0, 0.0)
        assert math.isnan(x) and math.isnan(y)

    def test_small_offset_ra(self):
        """A star slightly east of boresight should have x > 0."""
        x, y = _gnomonic_project(1.0, 0.0, 0.0, 0.0)
        assert x > 0.0

    def test_small_offset_dec(self):
        """A star slightly north of boresight should have y > 0."""
        x, y = _gnomonic_project(0.0, 1.0, 0.0, 0.0)
        assert y > 0.0


class TestMagToFlux:
    def test_reference_magnitude_gives_unit_flux(self):
        assert math.isclose(_mag_to_flux(0.0, vmag_ref=0.0), 1.0, rel_tol=1e-9)

    def test_fainter_star_has_lower_flux(self):
        f_bright = _mag_to_flux(0.0)
        f_faint = _mag_to_flux(5.0)
        assert f_faint < f_bright

    def test_five_magnitude_difference(self):
        """5 magnitudes = factor 100 in flux (Pogson's law)."""
        ratio = _mag_to_flux(0.0) / _mag_to_flux(5.0)
        assert math.isclose(ratio, 100.0, rel_tol=1e-6)

    def test_negative_magnitude_brighter_than_zero(self):
        assert _mag_to_flux(-1.0) > _mag_to_flux(0.0)

    def test_flux_always_positive(self):
        for vmag in [-2.0, 0.0, 2.0, 5.0, 10.0]:
            assert _mag_to_flux(vmag) > 0.0


class TestRenderGaussianStar:
    def test_adds_flux_to_image(self):
        img = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img, 32.0, 32.0, 1.0, sigma=2.0)
        assert img.max() > 0.0

    def test_peak_near_centre(self):
        img = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img, 32.0, 32.0, 1.0, sigma=1.5)
        row, col = np.unravel_index(img.argmax(), img.shape)
        assert abs(row - 32) <= 1 and abs(col - 32) <= 1

    def test_star_at_edge_does_not_raise(self):
        img = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img, 0.0, 0.0, 1.0, sigma=1.5)   # corner
        _render_gaussian_star(img, 63.0, 63.0, 1.0, sigma=1.5)  # opposite corner

    def test_out_of_bounds_star_ignored(self):
        img = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img, -100.0, -100.0, 1.0, sigma=1.5)
        assert img.max() == 0.0

    def test_higher_flux_gives_brighter_peak(self):
        img_lo = np.zeros((64, 64), dtype=np.float64)
        img_hi = np.zeros((64, 64), dtype=np.float64)
        _render_gaussian_star(img_lo, 32.0, 32.0, 0.5, sigma=1.5)
        _render_gaussian_star(img_hi, 32.0, 32.0, 1.0, sigma=1.5)
        assert img_hi.max() > img_lo.max()


class TestRandomBoresight:
    def test_returns_tuple_of_two_floats(self):
        rng = np.random.default_rng(0)
        ra, dec = _random_boresight(rng)
        assert isinstance(ra, float) and isinstance(dec, float)

    def test_ra_in_range(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            ra, _ = _random_boresight(rng)
            assert 0.0 <= ra < 360.0

    def test_dec_in_range(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            _, dec = _random_boresight(rng)
            assert -90.0 <= dec <= 90.0


# ---------------------------------------------------------------------------
# StarFieldGenerator.generate()
# ---------------------------------------------------------------------------


class TestStarFieldGeneratorOutput:
    def test_returns_synthetic_star_field(self, star_field):
        assert isinstance(star_field, SyntheticStarField)

    def test_image_is_float32(self, star_field):
        assert star_field.image.dtype == np.float32

    def test_image_shape(self, star_field):
        assert star_field.image.shape == (128, 128)

    def test_image_values_in_unit_range(self, star_field):
        assert star_field.image.min() >= 0.0
        assert star_field.image.max() <= 1.0

    def test_image_not_all_zeros(self, star_field):
        """A generated image should have non-zero pixels (background + stars)."""
        assert star_field.image.max() > 0.0

    def test_stars_list_is_list(self, star_field):
        assert isinstance(star_field.stars, list)

    def test_metadata_seed_matches(self, generator):
        sf = generator.generate(seed=7)
        assert sf.seed == 7

    def test_fov_stored_correctly(self, star_field):
        assert math.isclose(star_field.fov_deg, 20.0)

    def test_image_dimensions_stored(self, star_field):
        assert star_field.image_width == 128
        assert star_field.image_height == 128


class TestStarFieldDeterminism:
    def test_same_seed_identical_image(self, generator):
        sf1 = generator.generate(seed=100)
        sf2 = generator.generate(seed=100)
        assert np.array_equal(sf1.image, sf2.image)

    def test_same_seed_identical_star_count(self, generator):
        sf1 = generator.generate(seed=200)
        sf2 = generator.generate(seed=200)
        assert len(sf1.stars) == len(sf2.stars)

    def test_same_seed_identical_boresight(self, generator):
        sf1 = generator.generate(seed=300)
        sf2 = generator.generate(seed=300)
        assert math.isclose(sf1.boresight_ra_deg, sf2.boresight_ra_deg)
        assert math.isclose(sf1.boresight_dec_deg, sf2.boresight_dec_deg)

    def test_different_seeds_different_images(self, generator):
        sf1 = generator.generate(seed=1)
        sf2 = generator.generate(seed=2)
        # Images should differ (with overwhelming probability)
        assert not np.array_equal(sf1.image, sf2.image)

    def test_fixed_boresight_overrides_random(self, generator):
        sf = generator.generate(seed=99, boresight_ra_deg=45.0, boresight_dec_deg=10.0)
        assert math.isclose(sf.boresight_ra_deg, 45.0)
        assert math.isclose(sf.boresight_dec_deg, 10.0)


class TestRenderedStarPositions:
    def test_stars_within_image_bounds(self, star_field):
        """All rendered stars must have pixel coordinates inside the image."""
        for star in star_field.stars:
            assert 0.0 <= star.x_px < star_field.image_width, (
                f"Star {star.star_id} x={star.x_px} outside [0, {star_field.image_width})"
            )
            assert 0.0 <= star.y_px < star_field.image_height, (
                f"Star {star.star_id} y={star.y_px} outside [0, {star_field.image_height})"
            )

    def test_star_flux_in_unit_range(self, star_field):
        for star in star_field.stars:
            assert 0.0 < star.flux <= 1.0 + 1e-9, (
                f"Star {star.star_id} flux={star.flux} out of (0, 1]"
            )

    def test_brightest_star_has_unit_flux(self, star_field):
        if star_field.stars:
            max_flux = max(s.flux for s in star_field.stars)
            assert math.isclose(max_flux, 1.0, rel_tol=1e-5)

    def test_star_count_does_not_exceed_max(self, star_field):
        assert len(star_field.stars) <= MINIMAL_CONFIG["dataset"]["max_stars_per_image"]

    def test_star_ids_are_strings(self, star_field):
        for star in star_field.stars:
            assert isinstance(star.star_id, str)
            assert len(star.star_id) > 0


# ---------------------------------------------------------------------------
# load_image round-trip
# ---------------------------------------------------------------------------


class TestLoadImageRoundTrip:
    def test_save_and_reload_shape(self, star_field, tmp_path):
        import cv2
        path = tmp_path / "test.png"
        img_uint16 = (np.clip(star_field.image, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(path), img_uint16)
        loaded = load_image(path)
        assert loaded.shape == star_field.image.shape

    def test_save_and_reload_dtype(self, star_field, tmp_path):
        import cv2
        path = tmp_path / "test.png"
        img_uint16 = (np.clip(star_field.image, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(path), img_uint16)
        loaded = load_image(path)
        assert loaded.dtype == np.float32

    def test_save_and_reload_values_close(self, star_field, tmp_path):
        """Round-trip through 16-bit PNG should lose < 0.002 in absolute terms."""
        import cv2
        path = tmp_path / "test.png"
        img_uint16 = (np.clip(star_field.image, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(path), img_uint16)
        loaded = load_image(path)
        max_err = float(np.abs(loaded - star_field.image).max())
        assert max_err < 0.002, f"Round-trip max error {max_err} exceeds tolerance"

    def test_load_image_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.png")

    def test_load_image_unsupported_format(self, tmp_path):
        bad = tmp_path / "test.jpg"
        bad.write_bytes(b"\xff\xd8\xff")   # JPEG magic bytes
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(bad)


# ---------------------------------------------------------------------------
# build_dataset + load_metadata
# ---------------------------------------------------------------------------


class TestBuildDataset:
    @pytest.fixture()
    def small_config(self, tmp_path):
        cfg = {
            "dataset": {
                **MINIMAL_CONFIG["dataset"],
                "num_train": 3,
                "num_val": 1,
                "num_test": 1,
                "output_dir": str(tmp_path / "raw"),
                "metadata_file": str(tmp_path / "raw" / "metadata.json"),
            }
        }
        return cfg

    def test_returns_summary_dict(self, small_config):
        summary = build_dataset(small_config, verbose=False)
        assert isinstance(summary, dict)
        assert {"n_train", "n_val", "n_test", "output_dir", "metadata_file"} <= summary.keys()

    def test_correct_image_counts(self, small_config):
        build_dataset(small_config, verbose=False)
        raw = Path(small_config["dataset"]["output_dir"])
        assert len(list((raw / "train").glob("*.png"))) == 3
        assert len(list((raw / "val").glob("*.png"))) == 1
        assert len(list((raw / "test").glob("*.png"))) == 1

    def test_metadata_file_created(self, small_config):
        build_dataset(small_config, verbose=False)
        assert Path(small_config["dataset"]["metadata_file"]).exists()

    def test_metadata_entry_count(self, small_config):
        build_dataset(small_config, verbose=False)
        entries = load_metadata(small_config["dataset"]["metadata_file"])
        assert len(entries) == 5   # 3 train + 1 val + 1 test

    def test_metadata_required_keys(self, small_config):
        build_dataset(small_config, verbose=False)
        entries = load_metadata(small_config["dataset"]["metadata_file"])
        required = {
            "sample_id", "split", "image_file", "seed",
            "image_width", "image_height", "fov_deg",
            "boresight_ra_deg", "boresight_dec_deg", "roll_deg",
            "n_stars", "stars",
        }
        for entry in entries:
            assert required <= entry.keys(), (
                f"Entry {entry.get('sample_id')} missing keys: "
                f"{required - entry.keys()}"
            )

    def test_metadata_splits_correct(self, small_config):
        build_dataset(small_config, verbose=False)
        entries = load_metadata(small_config["dataset"]["metadata_file"])
        splits = [e["split"] for e in entries]
        assert splits.count("train") == 3
        assert splits.count("val") == 1
        assert splits.count("test") == 1

    def test_metadata_image_files_exist(self, small_config):
        build_dataset(small_config, verbose=False)
        raw = Path(small_config["dataset"]["output_dir"])
        entries = load_metadata(small_config["dataset"]["metadata_file"])
        for entry in entries:
            img_path = raw / entry["image_file"]
            assert img_path.exists(), f"Image file missing: {img_path}"

    def test_metadata_star_positions_within_bounds(self, small_config):
        build_dataset(small_config, verbose=False)
        entries = load_metadata(small_config["dataset"]["metadata_file"])
        for entry in entries:
            w, h = entry["image_width"], entry["image_height"]
            for star in entry["stars"]:
                assert 0.0 <= star["x_px"] < w, (
                    f"x_px={star['x_px']} out of bounds [0, {w})"
                )
                assert 0.0 <= star["y_px"] < h, (
                    f"y_px={star['y_px']} out of bounds [0, {h})"
                )

    def test_determinism_across_rebuild(self, small_config):
        """Building twice with the same config must produce identical metadata seeds."""
        build_dataset(small_config, verbose=False)
        entries1 = load_metadata(small_config["dataset"]["metadata_file"])
        build_dataset(small_config, verbose=False)
        entries2 = load_metadata(small_config["dataset"]["metadata_file"])
        seeds1 = [e["seed"] for e in entries1]
        seeds2 = [e["seed"] for e in entries2]
        assert seeds1 == seeds2

    def test_metadata_n_stars_matches_stars_list(self, small_config):
        build_dataset(small_config, verbose=False)
        entries = load_metadata(small_config["dataset"]["metadata_file"])
        for entry in entries:
            assert entry["n_stars"] == len(entry["stars"])


class TestLoadMetadata:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path / "nonexistent.json")

    def test_returns_list(self, tmp_path):
        p = tmp_path / "meta.json"
        p.write_text(json.dumps([{"a": 1}]))
        result = load_metadata(p)
        assert isinstance(result, list)
        assert result[0]["a"] == 1
