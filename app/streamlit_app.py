"""
streamlit_app.py
================
StellarX-StarNav-AI — Interactive Demonstration Interface.

This Streamlit application provides a user-facing interface for the
star pattern recognition and spacecraft navigation pipeline.

Planned features (Phase 7)
--------------------------
- Upload a star-field image
- Run the full preprocessing → detection → recognition → matching →
  navigation pipeline
- Display:
    - Preprocessed image with detected stars overlaid
    - Recognised star pattern identifier
    - Catalog match result and confidence score
    - Estimated spacecraft attitude (and position if applicable)
    - Per-stage processing time breakdown

Implementation note
-------------------
This file is a placeholder shell.  The UI layout and pipeline integration
will be implemented in Phase 7 once all upstream components are complete.
No mock predictions, dummy star catalogs, or fabricated results should be
added at any point.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StellarX — Star Pattern Recognition",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("StellarX — AI-Powered Star Pattern Recognition")
st.caption("Autonomous spacecraft navigation via neural-network-based star pattern recognition")

st.divider()


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------

st.info(
    "🚧  **Early development** — the pipeline is not yet implemented.  "
    "This interface will be activated in Phase 7 once all upstream "
    "components (preprocessing, detection, model, catalog, navigation) "
    "are complete.",
    icon="🚧",
)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    st.caption("Runtime settings will be loaded from config.yaml in Phase 7.")

    st.subheader("Pipeline stages")
    # TODO (Phase 7): add toggles / sliders for configurable parameters
    #   sourced from config.yaml (confidence threshold, max stars, etc.).
    st.write("_Not yet available._")

    st.divider()
    st.subheader("About")
    st.write(
        "**Team:** StellarX  \n"
        "**Version:** 0.1.0 (foundation)  \n"
        "**Status:** Phase 1 — foundation"
    )


# ---------------------------------------------------------------------------
# Main layout — placeholder columns
# ---------------------------------------------------------------------------

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Input")
    uploaded_file = st.file_uploader(
        "Upload a star-field image",
        type=["png", "jpg", "tiff", "fits"],
        disabled=True,
        help="Image upload will be enabled in Phase 7.",
    )
    st.caption("_Image upload is disabled until the pipeline is implemented._")

with col_output:
    st.subheader("Results")
    # TODO (Phase 7): display pipeline results here:
    #   - detected star overlay image
    #   - recognised pattern ID + confidence
    #   - catalog match details
    #   - attitude / position estimate
    #   - per-stage latency table
    st.write("_Results will appear here once the pipeline is implemented._")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "StellarX-StarNav-AI · Team StellarX · "
    "Implementation planned in subsequent development phases."
)
