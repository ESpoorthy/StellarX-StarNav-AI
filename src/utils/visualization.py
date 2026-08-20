"""
visualization.py
================
Shared plotting and visualization helpers used across the pipeline,
notebooks, and the Streamlit application.

Responsibility (planned)
------------------------
- Render a preprocessed star-field image with detected stars overlaid.
- Plot confidence score distributions.
- Visualize catalog match results.
- Display attitude estimation residuals.
- Produce processing-time and resource-usage charts.

Implementation note
-------------------
All functions should return Matplotlib Figure objects so that callers
(notebooks, Streamlit, tests) can control how figures are displayed or saved.
No hard-coded figure sizes or style settings — source from config where relevant.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_star_field(
    image: np.ndarray,
    title: str = "Star Field",
    cmap: str = "gray",
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Render a star-field image.

    Parameters
    ----------
    image:
        2-D float32 image array.
    title:
        Figure title string.
    cmap:
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    NotImplementedError
        Until implemented in Phase 2.
    """
    # TODO (Phase 2): implement using matplotlib.pyplot.imshow.
    raise NotImplementedError("plot_star_field is not yet implemented.")


def plot_detections(
    image: np.ndarray,
    star_positions: np.ndarray,
    title: str = "Detected Stars",
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Overlay detected star positions on a star-field image.

    Parameters
    ----------
    image:
        2-D float32 preprocessed image array.
    star_positions:
        Array of shape (N, 2) containing (x, y) centroid coordinates.
    title:
        Figure title string.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    NotImplementedError
        Until implemented in Phase 2.
    """
    # TODO (Phase 2): implement scatter overlay on the star-field image.
    raise NotImplementedError("plot_detections is not yet implemented.")


def plot_confidence_distribution(
    confidences: np.ndarray,
    title: str = "Confidence Distribution",
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Plot a histogram of recognition confidence scores.

    Parameters
    ----------
    confidences:
        1-D array of confidence values in [0.0, 1.0].
    title:
        Figure title string.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    NotImplementedError
        Until implemented in Phase 4.
    """
    # TODO (Phase 4): implement confidence histogram.
    raise NotImplementedError("plot_confidence_distribution is not yet implemented.")


def plot_attitude_residuals(
    residuals_deg: np.ndarray,
    title: str = "Attitude Estimation Residuals",
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Plot the distribution of attitude estimation angular residuals.

    Parameters
    ----------
    residuals_deg:
        1-D array of angular residual values in degrees.
    title:
        Figure title string.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    NotImplementedError
        Until implemented in Phase 5.
    """
    # TODO (Phase 5): implement residual distribution plot.
    raise NotImplementedError("plot_attitude_residuals is not yet implemented.")
