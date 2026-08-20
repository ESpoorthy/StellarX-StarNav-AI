"""
inference.py
============
Responsible for loading a trained model checkpoint and running inference
on extracted star features.

Responsibility (planned)
------------------------
1. Load a saved model checkpoint from the path specified in config.yaml.
2. Accept a feature array produced by src.preprocessing.star_detection.
3. Run the forward pass and post-process the output into a structured result.
4. Return a recognition result with a pattern identifier and confidence score.

Implementation note
-------------------
This module must not contain any training logic.  It is the production-facing
inference interface used by the navigation pipeline and the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.star_pattern_model import StarPatternModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RecognitionResult:
    """Structured output from one inference call.

    Attributes
    ----------
    pattern_id : str | None
        Identifier of the recognised star pattern.
        None if the model is not confident enough.
    confidence : float
        Confidence score in [0.0, 1.0].
    raw_output : np.ndarray
        Raw network output (logits or scores) before post-processing.
    latency_ms : float
        Wall-clock inference time in milliseconds.
    """

    pattern_id: str | None = None
    confidence: float = 0.0
    raw_output: np.ndarray = None  # type: ignore[assignment]
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str | Path, config: dict) -> StarPatternModel:
    """Load a trained model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to the saved ``.pt`` or ``.pth`` checkpoint.
    config:
        Model configuration dict (``model`` section of config.yaml).

    Returns
    -------
    StarPatternModel
        Model loaded in evaluation mode.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    NotImplementedError
        Until this function is implemented in Phase 3.
    """
    # TODO (Phase 3): implement checkpoint loading.
    #   Steps:
    #     1. build_model(config)
    #     2. model.load_state_dict(torch.load(checkpoint_path, map_location=...))
    #     3. model.eval()
    raise NotImplementedError("load_model is not yet implemented.")


def run_inference(
    model: StarPatternModel,
    features: np.ndarray,
    config: dict,
) -> RecognitionResult:
    """Run inference on a feature array and return a RecognitionResult.

    Parameters
    ----------
    model:
        Loaded StarPatternModel in evaluation mode.
    features:
        Feature array produced by ``src.preprocessing.star_detection.extract_features``.
    config:
        Evaluation configuration dict (``evaluation`` section of config.yaml).

    Returns
    -------
    RecognitionResult
        Structured recognition output including pattern ID, confidence,
        raw scores, and latency.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 3.
    """
    # TODO (Phase 3): implement inference.
    #   Steps:
    #     1. Convert features to torch.Tensor.
    #     2. Run model.forward() inside torch.no_grad().
    #     3. Apply softmax / sigmoid for confidence scores.
    #     4. Apply confidence threshold from config.
    #     5. Return RecognitionResult.
    raise NotImplementedError("run_inference is not yet implemented.")
