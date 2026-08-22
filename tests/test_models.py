"""
test_models.py
==============
Unit tests for:
  - src.models.sklearn_classifier  (StarPatternClassifier, boresight_to_label,
                                    train_classifier, evaluate_classifier)
  - src.models.inference            (RecognitionResult, load_model, run_inference)
  - src.preprocessing.feature_dataset (build_feature_dataset, load_feature_dataset)
  - src.models.star_pattern_model   (PyTorch deferral — ImportError expected)

Run with:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.models.sklearn_classifier import (
    StarPatternClassifier,
    TrainingResult,
    boresight_to_label,
    evaluate_classifier,
    train_classifier,
)
from src.models.inference import RecognitionResult, load_model, run_inference


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

FULL_CONFIG = {
    "dataset": {
        "catalog_file": "data/catalog/hipparcos_bright.csv",
        "catalog_mag_limit": 6.5,
        "image_width": 128, "image_height": 128,
        "field_of_view_deg": 20.0, "max_stars_per_image": 30,
        "psf_sigma_px": 1.5, "min_star_flux": 0.05,
        "background_level": 0.02, "read_noise_sigma": 0.005,
        "shot_noise": False, "artifact_probability": 0.0,
        "num_train": 60, "num_val": 20, "num_test": 20,
        "random_seed": 42,
        "output_dir": "data/raw",
        "metadata_file": "data/raw/metadata.json",
        "image_format": "png",
    },
    "preprocessing": {
        "background_subtraction": True,
        "background_method": "median_filter",
        "background_filter_size": 31,
        "noise_reduction": True,
        "noise_method": "gaussian",
        "noise_sigma": 0.8,
        "normalization": "min_max",
    },
    "star_detection": {
        "threshold_method": "absolute",
        "min_brightness": 0.05,
        "sigma_clip_k": 5.0,
        "min_area_px": 1, "max_area_px": 200,
        "min_peak_brightness": 0.04,
        "max_stars": 50,
        "centroid_method": "intensity_weighted",
        "centroid_half_window": 5,
    },
    "features": {
        "max_stars": 10,
        "descriptor": "pairwise_distances_and_ratios",
        "image_width": 128, "image_height": 128,
    },
    "model": {
        "backend": "sklearn",
        "classifier_type": "random_forest",
        "n_estimators": 20,       # small for fast tests
        "max_depth": 5,
        "n_neighbors": 3,
        "mlp_hidden_layers": [32, 16],
        "checkpoint_dir": "models",
        "checkpoint_name": "star_pattern_classifier.pkl",
        "n_sky_cells": 50,        # small for fast tests
    },
    "evaluation": {
        "confidence_threshold": 0.1,
        "top_k": 3,
    },
    "training": {"seed": 42},
}

FEAT_DIM = 90   # 2 * 10*(10-1)/2


def _make_synthetic_dataset(
    n_samples: int = 80,
    n_classes: int = 4,
    feat_dim: int = FEAT_DIM,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic multi-class dataset for fast classifier tests."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, feat_dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)
    return X, y


@pytest.fixture(scope="module")
def small_dataset():
    return _make_synthetic_dataset(n_samples=80, n_classes=4)


@pytest.fixture(scope="module")
def fitted_rf_clf(small_dataset):
    X, y = small_dataset
    clf = StarPatternClassifier(FULL_CONFIG)
    clf.fit(X, y, seed=42)
    return clf


# ===========================================================================
# boresight_to_label
# ===========================================================================

class TestBoresightToLabel:

    def test_returns_int(self):
        lbl = boresight_to_label(0.0, 0.0)
        assert isinstance(lbl, int)

    def test_label_non_negative(self):
        for ra, dec in [(0, 0), (90, 45), (180, -45), (359.9, 89.9)]:
            assert boresight_to_label(ra, dec) >= 0

    def test_label_within_range(self):
        n = 50
        for ra in range(0, 360, 30):
            for dec in range(-90, 91, 30):
                lbl = boresight_to_label(float(ra), float(dec), n_sky_cells=n)
                assert lbl < n * 4, f"label {lbl} >= n_sky_cells*4 for n={n}"

    def test_same_boresight_same_label(self):
        l1 = boresight_to_label(123.4, -56.7, n_sky_cells=200)
        l2 = boresight_to_label(123.4, -56.7, n_sky_cells=200)
        assert l1 == l2

    def test_different_boresights_different_labels(self):
        l1 = boresight_to_label(0.0,   0.0,  n_sky_cells=200)
        l2 = boresight_to_label(180.0, 0.0,  n_sky_cells=200)
        l3 = boresight_to_label(0.0,   45.0, n_sky_cells=200)
        assert l1 != l2
        assert l1 != l3

    def test_poles_covered(self):
        """RA=0 at both poles should return valid labels."""
        assert boresight_to_label(0.0,  90.0) >= 0
        assert boresight_to_label(0.0, -90.0) >= 0


# ===========================================================================
# StarPatternClassifier — construction
# ===========================================================================

class TestStarPatternClassifierConstruction:

    def test_not_fitted_initially(self):
        clf = StarPatternClassifier(FULL_CONFIG)
        assert not clf.is_fitted

    def test_n_classes_zero_before_fit(self):
        clf = StarPatternClassifier(FULL_CONFIG)
        assert clf.n_classes == 0

    def test_feature_dim_zero_before_fit(self):
        clf = StarPatternClassifier(FULL_CONFIG)
        assert clf.feature_dim == 0

    def test_unknown_classifier_type_raises(self):
        cfg = {**FULL_CONFIG, "model": {**FULL_CONFIG["model"], "classifier_type": "svm_magic"}}
        clf = StarPatternClassifier(cfg)
        X, y = _make_synthetic_dataset(10, 2)
        with pytest.raises(ValueError, match="Unknown"):
            clf.fit(X, y)

    def test_insufficient_samples_raises(self):
        clf = StarPatternClassifier(FULL_CONFIG)
        X = np.zeros((1, FEAT_DIM), dtype=np.float32)
        y = np.array([0])
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_mismatched_X_y_raises(self):
        clf = StarPatternClassifier(FULL_CONFIG)
        X = np.zeros((10, FEAT_DIM), dtype=np.float32)
        y = np.zeros(8, dtype=np.int64)
        with pytest.raises(ValueError):
            clf.fit(X, y)


# ===========================================================================
# StarPatternClassifier — fit / TrainingResult
# ===========================================================================

class TestStarPatternClassifierFit:

    def test_fit_returns_training_result(self, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        assert isinstance(result, TrainingResult)

    def test_is_fitted_after_fit(self, fitted_rf_clf):
        assert fitted_rf_clf.is_fitted

    def test_training_result_n_train(self, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        assert result.n_train == len(X)

    def test_training_result_feature_dim(self, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        assert result.feature_dim == FEAT_DIM

    def test_training_result_n_classes(self, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        assert result.n_classes == len(np.unique(y))

    def test_train_accuracy_in_unit_range(self, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_elapsed_sec_positive(self, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        assert result.elapsed_sec > 0.0

    def test_n_classes_property_after_fit(self, fitted_rf_clf, small_dataset):
        X, y = small_dataset
        assert fitted_rf_clf.n_classes == len(np.unique(y))

    def test_feature_dim_property_after_fit(self, fitted_rf_clf):
        assert fitted_rf_clf.feature_dim == FEAT_DIM

    def test_deterministic_same_seed(self, small_dataset):
        """Two fits with the same seed must produce the same predictions."""
        X, y = small_dataset
        clf1 = StarPatternClassifier(FULL_CONFIG)
        clf1.fit(X, y, seed=0)
        clf2 = StarPatternClassifier(FULL_CONFIG)
        clf2.fit(X, y, seed=0)
        labels1, _ = clf1.predict_batch(X)
        labels2, _ = clf2.predict_batch(X)
        assert np.array_equal(labels1, labels2)

    def test_knn_classifier_trains(self, small_dataset):
        X, y = small_dataset
        cfg = {**FULL_CONFIG, "model": {**FULL_CONFIG["model"], "classifier_type": "knn"}}
        clf = StarPatternClassifier(cfg)
        result = clf.fit(X, y, seed=42)
        assert clf.is_fitted
        assert result.n_train == len(X)

    def test_mlp_classifier_trains(self, small_dataset):
        X, y = small_dataset
        cfg = {**FULL_CONFIG, "model": {**FULL_CONFIG["model"], "classifier_type": "mlp"}}
        clf = StarPatternClassifier(cfg)
        result = clf.fit(X, y, seed=42)
        assert clf.is_fitted


# ===========================================================================
# StarPatternClassifier — predict / predict_batch
# ===========================================================================

class TestStarPatternClassifierPredict:

    def test_predict_returns_tuple(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        result = fitted_rf_clf.predict(X[0])
        assert isinstance(result, tuple) and len(result) == 2

    def test_predict_label_is_int(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        label, _ = fitted_rf_clf.predict(X[0])
        assert isinstance(label, int)

    def test_predict_confidence_in_unit_range(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        for x in X[:10]:
            _, conf = fitted_rf_clf.predict(x)
            assert 0.0 <= conf <= 1.0

    def test_predict_batch_returns_two_arrays(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        labels, confs = fitted_rf_clf.predict_batch(X)
        assert labels.shape == (len(X),)
        assert confs.shape == (len(X),)

    def test_predict_batch_labels_are_int(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        labels, _ = fitted_rf_clf.predict_batch(X)
        assert labels.dtype in (np.int32, np.int64, int)

    def test_predict_batch_confidences_in_unit_range(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        _, confs = fitted_rf_clf.predict_batch(X)
        assert confs.min() >= 0.0
        assert confs.max() <= 1.0 + 1e-6

    def test_predict_not_fitted_raises(self, small_dataset):
        clf = StarPatternClassifier(FULL_CONFIG)
        X, _ = small_dataset
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(X[0])

    def test_predict_batch_not_fitted_raises(self, small_dataset):
        clf = StarPatternClassifier(FULL_CONFIG)
        X, _ = small_dataset
        with pytest.raises(RuntimeError):
            clf.predict_batch(X)

    def test_predict_proba_topk_length(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        topk = fitted_rf_clf.predict_proba_topk(X[0], k=3)
        assert len(topk) <= 3

    def test_predict_proba_topk_sorted_descending(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        topk = fitted_rf_clf.predict_proba_topk(X[0], k=3)
        probs = [p for _, p in topk]
        assert probs == sorted(probs, reverse=True)

    def test_predict_proba_topk_sums_le_one(self, fitted_rf_clf, small_dataset):
        X, _ = small_dataset
        topk = fitted_rf_clf.predict_proba_topk(X[0], k=3)
        total = sum(p for _, p in topk)
        assert total <= 1.0 + 1e-6

    def test_train_accuracy_above_random_baseline(self, small_dataset):
        """In-sample accuracy must beat random guessing (25% for 4 classes)."""
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        result = clf.fit(X, y, seed=42)
        n_classes = len(np.unique(y))
        random_baseline = 1.0 / n_classes
        assert result.train_accuracy > random_baseline


# ===========================================================================
# StarPatternClassifier — save / load
# ===========================================================================

class TestStarPatternClassifierSaveLoad:

    def test_save_creates_file(self, fitted_rf_clf, tmp_path):
        path = tmp_path / "clf.pkl"
        fitted_rf_clf.save(path)
        assert path.exists()

    def test_loaded_classifier_is_fitted(self, fitted_rf_clf, tmp_path):
        path = tmp_path / "clf.pkl"
        fitted_rf_clf.save(path)
        loaded = StarPatternClassifier.load(path)
        assert loaded.is_fitted

    def test_loaded_classifier_same_predictions(self, fitted_rf_clf, small_dataset, tmp_path):
        X, _ = small_dataset
        path = tmp_path / "clf.pkl"
        fitted_rf_clf.save(path)
        loaded = StarPatternClassifier.load(path)
        orig_labels, _  = fitted_rf_clf.predict_batch(X)
        load_labels, _  = loaded.predict_batch(X)
        assert np.array_equal(orig_labels, load_labels)

    def test_loaded_classifier_feature_dim(self, fitted_rf_clf, tmp_path):
        path = tmp_path / "clf.pkl"
        fitted_rf_clf.save(path)
        loaded = StarPatternClassifier.load(path)
        assert loaded.feature_dim == fitted_rf_clf.feature_dim

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            StarPatternClassifier.load(tmp_path / "missing.pkl")

    def test_load_creates_parent_dir(self, fitted_rf_clf, tmp_path):
        path = tmp_path / "nested" / "dir" / "clf.pkl"
        fitted_rf_clf.save(path)
        assert path.exists()


# ===========================================================================
# train_classifier / evaluate_classifier helpers
# ===========================================================================

class TestTrainClassifierHelper:

    def test_returns_classifier_and_result(self, small_dataset):
        X, y = small_dataset
        clf, result = train_classifier(X, y, FULL_CONFIG, seed=42)
        assert isinstance(clf, StarPatternClassifier)
        assert isinstance(result, TrainingResult)
        assert clf.is_fitted

    def test_evaluate_returns_dict(self, small_dataset):
        X, y = small_dataset
        clf, _ = train_classifier(X, y, FULL_CONFIG, seed=42)
        metrics = evaluate_classifier(clf, X, y, FULL_CONFIG)
        assert isinstance(metrics, dict)

    def test_evaluate_required_keys(self, small_dataset):
        X, y = small_dataset
        clf, _ = train_classifier(X, y, FULL_CONFIG, seed=42)
        metrics = evaluate_classifier(clf, X, y, FULL_CONFIG)
        required = {"top1_accuracy", "topk_accuracy", "n_test", "n_classes",
                    "confidence_threshold", "n_above_threshold",
                    "accuracy_above_threshold"}
        assert required <= metrics.keys()

    def test_evaluate_top1_in_unit_range(self, small_dataset):
        X, y = small_dataset
        clf, _ = train_classifier(X, y, FULL_CONFIG, seed=42)
        metrics = evaluate_classifier(clf, X, y, FULL_CONFIG)
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0

    def test_evaluate_n_test_correct(self, small_dataset):
        X, y = small_dataset
        clf, _ = train_classifier(X, y, FULL_CONFIG, seed=42)
        metrics = evaluate_classifier(clf, X, y, FULL_CONFIG)
        assert metrics["n_test"] == len(X)


# ===========================================================================
# RecognitionResult
# ===========================================================================

class TestRecognitionResult:

    def test_default_pattern_id_is_none(self):
        r = RecognitionResult()
        assert r.pattern_id is None

    def test_default_confidence_is_zero(self):
        r = RecognitionResult()
        assert r.confidence == 0.0

    def test_default_top_k_is_empty_list(self):
        r = RecognitionResult()
        assert r.top_k_predictions == []

    def test_default_latency_is_zero(self):
        r = RecognitionResult()
        assert r.latency_ms == 0.0

    def test_construct_with_values(self):
        r = RecognitionResult(
            pattern_id="cell_42",
            confidence=0.85,
            raw_output=np.array([0.1, 0.85, 0.05]),
            latency_ms=3.2,
            top_k_predictions=[("cell_42", 0.85), ("cell_43", 0.10)],
        )
        assert r.pattern_id == "cell_42"
        assert math.isclose(r.confidence, 0.85)
        assert len(r.top_k_predictions) == 2


# ===========================================================================
# run_inference (end-to-end with sklearn)
# ===========================================================================

_RUN_INFERENCE_CLF = None
_RUN_INFERENCE_FEAT = None

def _get_run_inference_clf():
    global _RUN_INFERENCE_CLF, _RUN_INFERENCE_FEAT
    if _RUN_INFERENCE_CLF is None:
        X, y = _make_synthetic_dataset(n_samples=60, n_classes=4, seed=1)
        clf = StarPatternClassifier(FULL_CONFIG)
        clf.fit(X, y, seed=1)
        _RUN_INFERENCE_CLF = clf
        _RUN_INFERENCE_FEAT = X[0]
    return _RUN_INFERENCE_CLF, _RUN_INFERENCE_FEAT


class TestRunInference:

    def test_returns_recognition_result(self):
        clf, feat = _get_run_inference_clf()
        result = run_inference(clf, feat, FULL_CONFIG)
        assert isinstance(result, RecognitionResult)

    def test_confidence_in_unit_range(self):
        clf, feat = _get_run_inference_clf()
        result = run_inference(clf, feat, FULL_CONFIG)
        assert 0.0 <= result.confidence <= 1.0

    def test_latency_positive(self):
        clf, feat = _get_run_inference_clf()
        result = run_inference(clf, feat, FULL_CONFIG)
        assert result.latency_ms >= 0.0

    def test_pattern_id_when_above_threshold(self):
        """High-confidence prediction → pattern_id is not None."""
        clf, _ = _get_run_inference_clf()
        X, y = _make_synthetic_dataset(n_samples=60, n_classes=4, seed=1)
        # Predict on training data (high confidence)
        for x in X[:5]:
            _, conf = clf.predict(x)
            result = run_inference(clf, x, FULL_CONFIG)
            if conf >= FULL_CONFIG["evaluation"]["confidence_threshold"]:
                assert result.pattern_id is not None
                assert result.pattern_id.startswith("cell_")

    def test_pattern_id_none_below_threshold(self):
        """Force threshold=1.0 → all predictions below → pattern_id=None."""
        clf, feat = _get_run_inference_clf()
        cfg = {**FULL_CONFIG, "evaluation": {"confidence_threshold": 1.01, "top_k": 3}}
        result = run_inference(clf, feat, cfg)
        assert result.pattern_id is None

    def test_top_k_predictions_length(self):
        clf, feat = _get_run_inference_clf()
        result = run_inference(clf, feat, FULL_CONFIG)
        assert len(result.top_k_predictions) <= FULL_CONFIG["evaluation"]["top_k"]

    def test_top_k_sorted_descending(self):
        clf, feat = _get_run_inference_clf()
        result = run_inference(clf, feat, FULL_CONFIG)
        probs = [p for _, p in result.top_k_predictions]
        assert probs == sorted(probs, reverse=True)

    def test_unsupported_model_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            run_inference("not_a_model", np.zeros(FEAT_DIM, dtype=np.float32), FULL_CONFIG)


# ===========================================================================
# load_model
# ===========================================================================

class TestLoadModel:

    def test_loads_pkl_returns_classifier(self, tmp_path, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        clf.fit(X, y, seed=42)
        path = tmp_path / "clf.pkl"
        clf.save(path)
        loaded = load_model(path, FULL_CONFIG)
        assert isinstance(loaded, StarPatternClassifier)
        assert loaded.is_fitted

    def test_loaded_model_produces_same_results(self, tmp_path, small_dataset):
        X, y = small_dataset
        clf = StarPatternClassifier(FULL_CONFIG)
        clf.fit(X, y, seed=42)
        path = tmp_path / "clf.pkl"
        clf.save(path)
        loaded = load_model(path, FULL_CONFIG)
        orig_labels, _ = clf.predict_batch(X)
        load_labels, _ = loaded.predict_batch(X)
        assert np.array_equal(orig_labels, load_labels)

    def test_load_nonexistent_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "missing.pkl", FULL_CONFIG)

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        p = tmp_path / "model.onnx"
        p.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unrecognised"):
            load_model(p, FULL_CONFIG)

    def test_pytorch_checkpoint_raises_import_error(self, tmp_path):
        """A .pt file should raise ImportError (torch not installed)."""
        p = tmp_path / "model.pt"
        p.write_bytes(b"fake pt data")
        with pytest.raises(ImportError):
            load_model(p, FULL_CONFIG)


# ===========================================================================
# build_feature_dataset + load_feature_dataset
# ===========================================================================

class TestBuildFeatureDataset:

    @pytest.fixture()
    def tiny_config(self):
        """Config with very small counts for fast dataset building."""
        cfg = {k: v for k, v in FULL_CONFIG.items()}
        cfg["dataset"] = {**FULL_CONFIG["dataset"], "num_train": 10, "num_val": 4, "num_test": 4}
        return cfg

    def test_returns_three_items(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, y, meta = build_feature_dataset(tiny_config, verbose=False)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(meta, list)

    def test_X_dtype_float32(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, _, _ = build_feature_dataset(tiny_config, verbose=False)
        assert X.dtype == np.float32

    def test_X_y_same_length(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, y, meta = build_feature_dataset(tiny_config, verbose=False)
        assert len(X) == len(y) == len(meta)

    def test_feature_dim_correct(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, _, _ = build_feature_dataset(tiny_config, verbose=False)
        assert X.shape[1] == 90

    def test_total_samples(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, y, meta = build_feature_dataset(tiny_config, verbose=False)
        expected = (tiny_config["dataset"]["num_train"] +
                    tiny_config["dataset"]["num_val"] +
                    tiny_config["dataset"]["num_test"])
        assert len(X) == expected

    def test_train_split_only(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, y, meta = build_feature_dataset(tiny_config, split="train", verbose=False)
        assert len(X) == tiny_config["dataset"]["num_train"]
        assert all(m["split"] == "train" for m in meta)

    def test_meta_required_keys(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        _, _, meta = build_feature_dataset(tiny_config, verbose=False)
        required = {"seed", "split", "boresight_ra_deg", "boresight_dec_deg",
                    "n_stars_gt", "n_stars_detected", "label"}
        for entry in meta:
            assert required <= entry.keys()

    def test_labels_are_non_negative_integers(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        _, y, _ = build_feature_dataset(tiny_config, verbose=False)
        assert (y >= 0).all()

    def test_n_samples_override(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        X, y, meta = build_feature_dataset(tiny_config, n_samples=5, verbose=False)
        assert len(X) <= 5

    def test_save_and_load(self, tiny_config, tmp_path):
        from src.preprocessing.feature_dataset import build_feature_dataset, load_feature_dataset
        X, y, meta = build_feature_dataset(tiny_config, save_path=tmp_path, verbose=False)
        X2, y2, meta2 = load_feature_dataset(tmp_path)
        assert np.array_equal(X, X2)
        assert np.array_equal(y, y2)
        assert len(meta) == len(meta2)

    def test_load_missing_raises(self, tmp_path):
        from src.preprocessing.feature_dataset import load_feature_dataset
        with pytest.raises(FileNotFoundError):
            load_feature_dataset(tmp_path / "nonexistent")

    def test_unknown_split_raises(self, tiny_config):
        from src.preprocessing.feature_dataset import build_feature_dataset
        with pytest.raises(ValueError, match="Unknown split"):
            build_feature_dataset(tiny_config, split="holdout", verbose=False)


# ===========================================================================
# StarPatternModel — PyTorch deferral
# ===========================================================================

class TestStarPatternModelDeferral:

    def test_build_model_raises_import_error(self):
        from src.models.star_pattern_model import build_model
        with pytest.raises(ImportError):
            build_model(FULL_CONFIG)

    def test_star_pattern_model_raises_import_error(self):
        from src.models.star_pattern_model import StarPatternModel
        with pytest.raises(ImportError):
            StarPatternModel(FULL_CONFIG["model"])

    def test_pytorch_available_returns_false(self):
        from src.models.star_pattern_model import _pytorch_available
        assert _pytorch_available() is False
