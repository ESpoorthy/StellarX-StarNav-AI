"""
pattern_matcher.py
==================
Responsible for matching a recognised star pattern (output of the neural
network inference step) against entries in the loaded star catalog.

Responsibility (planned)
------------------------
1. Accept a RecognitionResult from src.models.inference.
2. Query the StarCatalog for the corresponding catalog entry.
3. Apply a geometric verification step (optional, TBD).
4. Return a MatchResult with the catalog entry and a match confidence.

Implementation note
-------------------
The matching algorithm is to be determined during Phase 4.
All configurable thresholds must be sourced from config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.catalog.catalog_loader import CatalogStar, StarCatalog
from src.models.inference import RecognitionResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Structured output from a catalog matching operation.

    Attributes
    ----------
    matched_star : CatalogStar | None
        The best-matching catalog entry, or None if no confident match found.
    match_confidence : float
        Confidence of the match in [0.0, 1.0].
    is_confident : bool
        True if match_confidence exceeds the configured threshold.
    candidates : list[CatalogStar]
        Ordered list of candidate matches (best first) for inspection.
    """

    matched_star: CatalogStar | None = None
    match_confidence: float = 0.0
    is_confident: bool = False
    candidates: list[CatalogStar] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.candidates is None:
            self.candidates = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_pattern(
    recognition_result: RecognitionResult,
    catalog: StarCatalog,
    config: dict,
) -> MatchResult:
    """Match a recognition result against the star catalog.

    Parameters
    ----------
    recognition_result:
        Output of ``src.models.inference.run_inference``.
    catalog:
        Loaded StarCatalog instance.
    config:
        Evaluation configuration dict (``evaluation`` section of config.yaml).

    Returns
    -------
    MatchResult
        Best catalog match with confidence information.

    Raises
    ------
    NotImplementedError
        Until this function is implemented in Phase 4.
    """
    # TODO (Phase 4): implement pattern matching.
    #   Algorithm candidates:
    #     - Direct index lookup if the neural network outputs a catalog ID.
    #     - k-nearest-neighbour search in embedding space.
    #     - Geometric verification (RANSAC or similar) as a post-filter.
    raise NotImplementedError("match_pattern is not yet implemented.")
