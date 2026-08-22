"""
star_pattern_model.py
=====================
PyTorch-based neural network for star pattern recognition.

Phase 3 status — DEFERRED
--------------------------
PyTorch does not publish a wheel for Python 3.14 (the version installed on
this machine) as of August 2026.  This module therefore remains a documented
stub.  The active Phase 3 classifier is
:class:`~src.models.sklearn_classifier.StarPatternClassifier`.

When PyTorch publishes a Python 3.14-compatible wheel (or when the project
migrates to Python 3.12), this file will be updated with a concrete
``torch.nn.Module`` architecture.  The class interface defined here is
intentionally stable so that downstream callers (``inference.py``,
``notebooks/04_model_training.ipynb``) require minimal changes.

Architecture plan
-----------------
The model will be a multi-layer perceptron (MLP) operating on the fixed-
length pairwise-distance + brightness-ratio feature vector produced by
:func:`~src.preprocessing.star_detection.extract_features`.

Planned layers (to be finalised during PyTorch integration):

    Input (feature_dim = 90)
        ↓ Linear → BatchNorm → ReLU → Dropout
        ↓ Linear → BatchNorm → ReLU → Dropout
        ↓ Linear → BatchNorm → ReLU
        ↓ Linear (output_classes)
    Softmax → probabilities

All hyperparameters (hidden sizes, dropout rate, output classes) will be
sourced from config.yaml — no hard-coded values.

Usage (once PyTorch is available)
----------------------------------
>>> from src.models.star_pattern_model import build_model
>>> model = build_model(config)
>>> logits = model(feature_tensor)      # shape (batch, output_classes)
"""

from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def _pytorch_available() -> bool:
    """Return True if torch can be imported on this Python version."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_PYTORCH_NOTE = (
    f"PyTorch is not available on Python {sys.version_info.major}."
    f"{sys.version_info.minor}. "
    "The active Phase 3 classifier is "
    "src.models.sklearn_classifier.StarPatternClassifier. "
    "Install PyTorch (requires Python ≤ 3.12 as of Aug 2026) to use this module."
)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class StarPatternModel:
    """Neural network model for star pattern recognition (PyTorch).

    This class is a **deferred stub** — see module docstring for context.
    It will subclass ``torch.nn.Module`` once PyTorch is available.

    Parameters
    ----------
    config:
        Model configuration dict (``config["model"]`` from config.yaml).
        Expected keys: ``feature_dim``, ``output_classes``,
        ``mlp_hidden_layers``.
    """

    def __init__(self, config: dict) -> None:
        if not _pytorch_available():
            raise ImportError(_PYTORCH_NOTE)

        # Reached only when torch is importable
        import torch.nn as nn  # noqa: F401 — imported here for future use

        self.config = config
        # TODO: define layers once PyTorch is available
        raise NotImplementedError(
            "StarPatternModel architecture is not yet implemented. "
            "See module docstring for the planned MLP design."
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor, shape (batch, feature_dim).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (batch, output_classes).
        """
        raise NotImplementedError("StarPatternModel.forward — not yet implemented.")


def build_model(config: dict) -> "StarPatternModel":
    """Instantiate a :class:`StarPatternModel` from *config*.

    Parameters
    ----------
    config:
        Full project configuration dict.

    Returns
    -------
    StarPatternModel

    Raises
    ------
    ImportError
        If PyTorch is not available on this Python version.
    NotImplementedError
        Until the architecture is implemented.
    """
    if not _pytorch_available():
        raise ImportError(_PYTORCH_NOTE)
    return StarPatternModel(config["model"])
