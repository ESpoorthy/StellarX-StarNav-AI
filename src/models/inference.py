"""
inference.py
============
Production inference interface for star pattern recognition.

Phase 3 implementation
-----------------------
:func:`run_inference` now operates on the fitted
:class:`~src.models.sklearn_classifier.StarPatternClassifier`.

:func:`load_model` loads the serialised ``.pkl`` checkpoint produced by
:meth:`~src.models.sklearn_classifier.StarPatternClassifier.save`.

The PyTorch ``load_model`` path (loading ``.pt`` / ``.pth`` checkpoints) is
preserved as a documented stub and will be activated when PyTorch is
available.

Public contract
---------------
All downstream code (navigation pipeline, Streamlit app) should call only:

    model = load_model(checkpoint_path, config)
    result = run_inference(model, features, config)

The returned :class:`RecognitionResult` has a stable interface regardless
of which backend (sklearn / PyTorch) is in use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RecognitionResult:
    """Structured output from one inference call.

    Attributes
    ----------
    pattern_id : str | None
        Recognised sky-cell label as a string (``"cell_<N>"``), or ``None``
        if the model confidence is below ``evaluation.confidence_threshold``.
    confidence : float
        Predicted class probability in [0.0, 1.0].
    raw_output : np.ndarray | None
        Full probability vector over all classes before thresholding.
        ``None`` if the backend does not produce per-class probabilities.
    latency_ms : float
        Wall-clock inference time in milliseconds.
    top_k_predictions : list[tuple[str, float]]
        Top-k (pattern_id, probability) pairs sorted by descending probability.
        Empty list if the backend does not support probability estimates.
    """

    pattern_id: str | None = None
    confidence: float = 0.0
    raw_output: np.ndarray | None = None
    latency_ms: float = 0.0
    top_k_predictions: list[tuple[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.top_k_predictions is None:
            self.top_k_predictions = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str | Path, config: dict):
    """Load a trained model from a checkpoint file.

    Supports two checkpoint formats:

    - ``.pkl`` — scikit-learn :class:`~src.models.sklearn_classifier.StarPatternClassifier`
      (Phase 3 active backend).
    - ``.pt`` / ``.pth`` — PyTorch :class:`~src.models.star_pattern_model.StarPatternModel`
      (deferred until PyTorch is available on Python 3.14).

    Parameters
    ----------
    checkpoint_path:
        Path to the saved checkpoint file.
    config:
        Full project configuration dict.

    Returns
    -------
    StarPatternClassifier | StarPatternModel
        Fitted model ready for inference.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    ValueError
        If the checkpoint extension is not recognised.
    ImportError
        If a ``.pt`` checkpoint is requested but PyTorch is unavailable.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    suffix = checkpoint_path.suffix.lower()

    if suffix == ".pkl":
        from src.models.sklearn_classifier import StarPatternClassifier
        return StarPatternClassifier.load(checkpoint_path, config)

    elif suffix in {".pt", ".pth"}:
        # PyTorch path — deferred
        try:
            import torch  # noqa: F401
        except ImportError:
            from src.models.star_pattern_model import _PYTORCH_NOTE
            raise ImportError(_PYTORCH_NOTE)
        from src.models.star_pattern_model import build_model
        import torch
        model = build_model(config)
        state = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    else:
        raise ValueError(
            f"Unrecognised checkpoint extension '{suffix}'. "
            "Expected '.pkl' (sklearn) or '.pt' / '.pth' (PyTorch)."
        )


def run_inference(
    model,
    features: np.ndarray,
    config: dict,
) -> RecognitionResult:
    """Run inference on a feature vector and return a :class:`RecognitionResult`.

    Dispatches to the appropriate backend based on the model type.

    Parameters
    ----------
    model:
        A fitted :class:`~src.models.sklearn_classifier.StarPatternClassifier`
        or (future) ``StarPatternModel``.
    features:
        1-D float32 feature array of shape ``(feature_dim,)``.
    config:
        Full project configuration dict.

    Returns
    -------
    RecognitionResult
        Structured output with pattern ID, confidence, top-k predictions,
        and latency.
    """
    conf_threshold = float(
        config.get("evaluation", {}).get("confidence_threshold", 0.3)
    )
    top_k = int(config.get("evaluation", {}).get("top_k", 3))

    # ── sklearn backend ──────────────────────────────────────────────────────
    try:
        from src.models.sklearn_classifier import StarPatternClassifier
        if isinstance(model, StarPatternClassifier):
            return _run_sklearn_inference(model, features, conf_threshold, top_k)
    except ImportError:
        pass

    # ── PyTorch backend (deferred) ───────────────────────────────────────────
    try:
        import torch
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return _run_pytorch_inference(model, features, conf_threshold, top_k)
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        "Expected StarPatternClassifier or torch.nn.Module."
    )


# ---------------------------------------------------------------------------
# Backend-specific inference helpers
# ---------------------------------------------------------------------------


def _run_sklearn_inference(
    model,
    features: np.ndarray,
    conf_threshold: float,
    top_k: int,
) -> RecognitionResult:
    """Run inference using the sklearn classifier backend."""
    t0 = time.perf_counter()

    label, confidence = model.predict(features)
    pattern_id = f"cell_{label}" if confidence >= conf_threshold else None

    # Top-k predictions
    top_k_preds: list[tuple[str, float]] = []
    if hasattr(model._clf, "predict_proba"):
        topk_raw = model.predict_proba_topk(features, k=top_k)
        top_k_preds = [(f"cell_{lbl}", prob) for lbl, prob in topk_raw]

    # Full probability vector
    raw_output: np.ndarray | None = None
    if hasattr(model._clf, "predict_proba"):
        raw_output = model._clf.predict_proba(
            features.reshape(1, -1)
        )[0].astype(np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return RecognitionResult(
        pattern_id=pattern_id,
        confidence=confidence,
        raw_output=raw_output,
        latency_ms=latency_ms,
        top_k_predictions=top_k_preds,
    )


def _run_pytorch_inference(
    model,
    features: np.ndarray,
    conf_threshold: float,
    top_k: int,
) -> RecognitionResult:
    """Run inference using the PyTorch backend (deferred — not yet active)."""
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()

    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probas = F.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_class = int(probas.argmax())
    confidence = float(probas[pred_class])
    pattern_id = f"cell_{pred_class}" if confidence >= conf_threshold else None

    topk_idx = probas.argsort()[::-1][:top_k]
    top_k_preds = [(f"cell_{i}", float(probas[i])) for i in topk_idx]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return RecognitionResult(
        pattern_id=pattern_id,
        confidence=confidence,
        raw_output=probas,
        latency_ms=latency_ms,
        top_k_predictions=top_k_preds,
    )
