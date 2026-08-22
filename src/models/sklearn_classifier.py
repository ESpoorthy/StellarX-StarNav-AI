"""
sklearn_classifier.py
=====================
Star pattern recognition using scikit-learn classifiers.

Phase 3 active implementation
------------------------------
PyTorch is not available on Python 3.14 (no compatible wheel exists as of
August 2026).  This module provides a fully working classification pipeline
using scikit-learn, which is already installed.  The :class:`StarPatternClassifier`
is the production classifier for Phase 3 and Phase 4.

The PyTorch-based :class:`~src.models.star_pattern_model.StarPatternModel` stub
is preserved in ``star_pattern_model.py`` and will replace this module once
PyTorch supports Python 3.14.

Classification task
-------------------
Each star-field frame yields a feature vector (see
:func:`~src.preprocessing.star_detection.extract_features`) and a label.
The label is a **boresight region ID** — a discretised sky cell that
identifies which part of the sky the camera was pointing at.  Two frames
pointing at nearly the same boresight will have the same label and similar
feature vectors; frames pointing at different sky regions will have
different labels and dissimilar feature vectors.

Sky tessellation
~~~~~~~~~~~~~~~~
The sky is divided into a fixed grid of cells using a simple (RA, Dec)
discretisation:

    ra_bin  = floor(ra_deg  / ra_step)   mod n_ra_cells
    dec_bin = floor((dec_deg + 90) / dec_step)  mod n_dec_cells
    label   = ra_bin * n_dec_cells + dec_bin

The number of cells is controlled by ``config["model"]["n_sky_cells"]``
(default 500, giving ~500 distinguishable sky regions).  With the 50-star
prototype catalog only a small fraction of cells will have training samples;
expanding the catalog will populate more cells.

Classifiers supported
~~~~~~~~~~~~~~~~~~~~~
- ``random_forest`` — ensemble of decision trees; good generalisation,
  handles noisy features, provides probability estimates via voting.
- ``knn`` — k-nearest neighbours; simple, interpretable, no training phase;
  requires all training vectors in memory at inference time.
- ``mlp`` — multi-layer perceptron; higher capacity, requires more data.

All hyperparameters are sourced from ``config["model"]``.
"""

from __future__ import annotations

import math
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Sky tessellation helper
# ---------------------------------------------------------------------------


def boresight_to_label(
    ra_deg: float,
    dec_deg: float,
    n_sky_cells: int = 500,
) -> int:
    """Convert a boresight (RA, Dec) to a discrete sky-cell label.

    The sky is divided into approximately ``n_sky_cells`` cells using a
    simple cylindrical equal-area tessellation:

    - RA axis: ``n_ra`` bins uniformly spaced in [0°, 360°)
    - Dec axis: ``n_dec`` bins uniformly spaced in [-90°, +90°]
    - ``n_ra * n_dec ≈ n_sky_cells`` with aspect ratio ≈ 2:1

    Parameters
    ----------
    ra_deg, dec_deg:
        Boresight in degrees.
    n_sky_cells:
        Target number of sky cells (actual = n_ra * n_dec).

    Returns
    -------
    int
        Label in [0, n_ra * n_dec).
    """
    n_dec = max(1, int(math.sqrt(n_sky_cells / 2)))
    n_ra  = max(1, int(n_sky_cells / n_dec))

    ra_step  = 360.0 / n_ra
    dec_step = 180.0 / n_dec

    ra_bin  = int(ra_deg  / ra_step)  % n_ra
    dec_bin = int((dec_deg + 90.0) / dec_step) % n_dec

    return ra_bin * n_dec + dec_bin


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Summary of a classifier training run.

    Attributes
    ----------
    classifier_type : str
        Name of the classifier used.
    n_train : int
        Number of training samples.
    n_classes : int
        Number of distinct class labels seen during training.
    feature_dim : int
        Dimensionality of the input feature vector.
    train_accuracy : float
        Accuracy on the training set (in-sample).
    elapsed_sec : float
        Wall-clock training time in seconds.
    config_used : dict
        Snapshot of the model config used.
    """

    classifier_type: str = ""
    n_train: int = 0
    n_classes: int = 0
    feature_dim: int = 0
    train_accuracy: float = 0.0
    elapsed_sec: float = 0.0
    config_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class StarPatternClassifier:
    """Scikit-learn classifier for star pattern recognition.

    Wraps a sklearn estimator with:
    - training via :meth:`fit`
    - single-sample prediction via :meth:`predict`
    - batch prediction via :meth:`predict_batch`
    - top-k probability prediction via :meth:`predict_proba_topk`
    - serialisation via :meth:`save` / :meth:`load`

    Parameters
    ----------
    config:
        Full project configuration dict (loaded from config.yaml).
        The ``model`` sub-dict is used.
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config.get("model", {})
        self._clf: Any = None
        self._label_encoder = LabelEncoder()
        self._is_fitted: bool = False
        self._feature_dim: int = 0
        self._n_sky_cells: int = int(self._cfg.get("n_sky_cells", 500))
        self._classifier_type: str = self._cfg.get("classifier_type", "random_forest")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int | None = None,
    ) -> TrainingResult:
        """Train the classifier on feature matrix *X* and labels *y*.

        Parameters
        ----------
        X:
            Float32 array of shape (N, feature_dim).
        y:
            Integer array of shape (N,) — sky-cell labels from
            :func:`boresight_to_label`.
        seed:
            Random seed for reproducibility.  Overrides config if given.

        Returns
        -------
        TrainingResult
            Training summary including accuracy and timing.

        Raises
        ------
        ValueError
            If *X* and *y* have incompatible shapes or fewer than 2 samples.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X has {len(X)} rows but y has {len(y)} entries."
            )
        if len(X) < 2:
            raise ValueError("Need at least 2 training samples.")

        rng_seed = seed if seed is not None else int(
            self._cfg.get("seed", 42)
        )

        # Encode labels to consecutive integers
        y_enc = self._label_encoder.fit_transform(y)

        self._feature_dim = X.shape[1]
        self._clf = self._build_estimator(rng_seed)

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf.fit(X, y_enc)
        elapsed = time.time() - t0

        self._is_fitted = True

        train_acc = float(accuracy_score(y_enc, self._clf.predict(X)))

        return TrainingResult(
            classifier_type=self._classifier_type,
            n_train=len(X),
            n_classes=len(self._label_encoder.classes_),
            feature_dim=self._feature_dim,
            train_accuracy=train_acc,
            elapsed_sec=elapsed,
            config_used=dict(self._cfg),
        )

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Predict the sky-cell label for a single feature vector.

        Parameters
        ----------
        features:
            1-D float32 feature vector of shape (feature_dim,).

        Returns
        -------
        tuple[int, float]
            ``(label, confidence)`` where *label* is the original sky-cell
            integer and *confidence* is the predicted class probability.

        Raises
        ------
        RuntimeError
            If the classifier has not been trained yet.
        """
        self._check_fitted()
        x = features.reshape(1, -1)
        enc_pred = int(self._clf.predict(x)[0])
        label = int(self._label_encoder.inverse_transform([enc_pred])[0])
        confidence = self._get_confidence(x, enc_pred)
        return label, confidence

    def predict_batch(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict sky-cell labels for a batch of feature vectors.

        Parameters
        ----------
        X:
            Float32 array of shape (N, feature_dim).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(labels, confidences)`` each of shape (N,).

        Raises
        ------
        RuntimeError
            If the classifier has not been trained yet.
        """
        self._check_fitted()
        enc_preds = self._clf.predict(X)
        labels = self._label_encoder.inverse_transform(enc_preds).astype(int)

        if hasattr(self._clf, "predict_proba"):
            probas = self._clf.predict_proba(X)
            confidences = probas[np.arange(len(X)), enc_preds].astype(np.float32)
        else:
            confidences = np.ones(len(X), dtype=np.float32)

        return labels, confidences

    def predict_proba_topk(
        self, features: np.ndarray, k: int = 3
    ) -> list[tuple[int, float]]:
        """Return the top-k most probable sky-cell labels for one sample.

        Parameters
        ----------
        features:
            1-D float32 feature vector.
        k:
            Number of top predictions to return.

        Returns
        -------
        list of (label, probability)
            Sorted by descending probability.

        Raises
        ------
        RuntimeError
            If the classifier has not been trained yet or does not support
            probability estimates.
        """
        self._check_fitted()
        if not hasattr(self._clf, "predict_proba"):
            raise RuntimeError(
                f"Classifier '{self._classifier_type}' does not support "
                "probability estimates."
            )
        x = features.reshape(1, -1)
        probas = self._clf.predict_proba(x)[0]
        top_enc = np.argsort(probas)[::-1][:k]
        result = []
        for enc in top_enc:
            label = int(self._label_encoder.inverse_transform([enc])[0])
            result.append((label, float(probas[enc])))
        return result

    def save(self, path: str | Path) -> None:
        """Serialise the fitted classifier to *path* using pickle.

        Parameters
        ----------
        path:
            Destination ``.pkl`` file path.  Parent directory is created
            if it does not exist.

        Raises
        ------
        RuntimeError
            If the classifier has not been trained yet.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clf":           self._clf,
            "label_encoder": self._label_encoder,
            "feature_dim":   self._feature_dim,
            "cfg":           self._cfg,
            "n_sky_cells":   self._n_sky_cells,
            "classifier_type": self._classifier_type,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path, config: dict | None = None) -> "StarPatternClassifier":
        """Load a previously saved :class:`StarPatternClassifier` from *path*.

        Parameters
        ----------
        path:
            Path to a ``.pkl`` file saved by :meth:`save`.
        config:
            Optional config dict.  If ``None``, the config embedded in the
            checkpoint is used.

        Returns
        -------
        StarPatternClassifier
            Fitted classifier ready for prediction.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {path}")

        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        obj = cls(config={"model": payload["cfg"]} if config is None else config)
        obj._clf             = payload["clf"]
        obj._label_encoder   = payload["label_encoder"]
        obj._feature_dim     = payload["feature_dim"]
        obj._n_sky_cells     = payload["n_sky_cells"]
        obj._classifier_type = payload["classifier_type"]
        obj._is_fitted       = True
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if the classifier has been trained."""
        return self._is_fitted

    @property
    def n_classes(self) -> int:
        """Number of classes seen during training."""
        if not self._is_fitted:
            return 0
        return int(len(self._label_encoder.classes_))

    @property
    def feature_dim(self) -> int:
        """Expected feature vector dimensionality."""
        return self._feature_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_estimator(self, seed: int) -> Any:
        """Instantiate the underlying sklearn estimator from config."""
        ctype = self._classifier_type

        if ctype == "random_forest":
            return RandomForestClassifier(
                n_estimators=int(self._cfg.get("n_estimators", 200)),
                max_depth=self._cfg.get("max_depth") or None,
                random_state=seed,
                n_jobs=-1,
            )
        elif ctype == "knn":
            return KNeighborsClassifier(
                n_neighbors=int(self._cfg.get("n_neighbors", 5)),
                metric="euclidean",
                n_jobs=-1,
            )
        elif ctype == "mlp":
            hidden = tuple(self._cfg.get("mlp_hidden_layers", [256, 128]))
            return MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=500,
                random_state=seed,
            )
        else:
            raise ValueError(
                f"Unknown classifier_type '{ctype}'. "
                "Choose 'random_forest', 'knn', or 'mlp'."
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "StarPatternClassifier has not been trained yet. "
                "Call fit() before predict()."
            )

    def _get_confidence(self, x: np.ndarray, enc_pred: int) -> float:
        """Return the predicted class probability for a single sample."""
        if hasattr(self._clf, "predict_proba"):
            probas = self._clf.predict_proba(x)[0]
            return float(probas[enc_pred])
        return 1.0  # KNN without proba returns no confidence


# ---------------------------------------------------------------------------
# Convenience module-level functions
# ---------------------------------------------------------------------------


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    seed: int | None = None,
) -> tuple["StarPatternClassifier", TrainingResult]:
    """Train a :class:`StarPatternClassifier` and return it with training stats.

    Parameters
    ----------
    X_train:
        Float32 feature matrix of shape (N, feature_dim).
    y_train:
        Integer sky-cell labels of shape (N,).
    config:
        Full project configuration dict.
    seed:
        Random seed.  Overrides config training seed if given.

    Returns
    -------
    tuple[StarPatternClassifier, TrainingResult]
    """
    clf = StarPatternClassifier(config)
    result = clf.fit(X_train, y_train, seed=seed)
    return clf, result


def evaluate_classifier(
    clf: "StarPatternClassifier",
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> dict:
    """Evaluate a fitted classifier on a test set.

    Parameters
    ----------
    clf:
        Fitted :class:`StarPatternClassifier`.
    X_test:
        Float32 feature matrix of shape (N, feature_dim).
    y_test:
        Integer sky-cell labels of shape (N,).
    config:
        Full project configuration dict.

    Returns
    -------
    dict
        Keys: ``top1_accuracy``, ``topk_accuracy``, ``n_test``,
        ``n_classes``, ``confidence_threshold``,
        ``n_above_threshold``, ``accuracy_above_threshold``.
    """
    top_k             = int(config.get("evaluation", {}).get("top_k", 3))
    conf_threshold    = float(config.get("evaluation", {}).get("confidence_threshold", 0.3))

    labels, confs = clf.predict_batch(X_test)

    # Top-1 accuracy
    top1 = float(accuracy_score(y_test, labels))

    # Top-k accuracy
    if hasattr(clf._clf, "predict_proba"):
        probas   = clf._clf.predict_proba(X_test)
        y_enc    = clf._label_encoder.transform(y_test)
        topk_correct = 0
        for i in range(len(y_enc)):
            top_enc = np.argsort(probas[i])[::-1][:top_k]
            if y_enc[i] in top_enc:
                topk_correct += 1
        topk_acc = topk_correct / len(y_test)
    else:
        topk_acc = top1

    # Accuracy above confidence threshold
    mask = confs >= conf_threshold
    n_above = int(mask.sum())
    acc_above = float(accuracy_score(y_test[mask], labels[mask])) if n_above > 0 else float("nan")

    return {
        "top1_accuracy":          top1,
        "topk_accuracy":          topk_acc,
        "top_k":                  top_k,
        "n_test":                 len(y_test),
        "n_classes":              clf.n_classes,
        "confidence_threshold":   conf_threshold,
        "n_above_threshold":      n_above,
        "accuracy_above_threshold": acc_above,
    }
