"""
edge_config.py — Phase 6
=========================
Edge deployment configuration and requirements for StellarX.

Defines CPU-only inference configuration suitable for constrained hardware:
- Raspberry Pi 4 (4-core ARM Cortex-A72, 4 GB RAM)
- NVIDIA Jetson Nano (4-core ARM Cortex-A57, 4 GB LPDDR4)
- Low-power x86 (Intel NUC, i3/i5 class)

IMPORTANT: performance figures below are ESTIMATES based on measured
desktop performance scaled by known CPU speed ratios. They are NOT
measured on actual edge hardware. All measured values are labelled
MEASURED; estimates are labelled ESTIMATED.

To validate on actual edge hardware:
    python benchmark.py --n-images 20 --edge-mode
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Runtime dependencies for inference-only deployment
# ---------------------------------------------------------------------------

INFERENCE_DEPENDENCIES = [
    "numpy>=1.26",       # array ops
    "scipy>=1.11",       # KDTree, gaussian_filter, ndimage
    "pandas>=2.0",       # catalog CSV loading
    "scikit-learn>=1.3", # StarPatternClassifier (optional neural prior)
    "pyyaml>=6.0",       # config loading
    # NOT required for inference:
    #   matplotlib, pillow, jupyter, pytest
]

TRAINING_ONLY_DEPENDENCIES = [
    "matplotlib>=3.8",
    "pillow>=10.0",
    "pytest>=7.0",
    "jupyter",
]


# ---------------------------------------------------------------------------
# Edge hardware profiles
# ---------------------------------------------------------------------------


@dataclass
class EdgeProfile:
    """Hardware profile for edge deployment planning.

    Attributes
    ----------
    name : str  Hardware platform name.
    cpu_cores : int  Available CPU cores.
    ram_mb : int  Available RAM in megabytes.
    measured : bool  True if measured on actual hardware, False if estimated.
    expected_latency_ms : float  Expected per-image latency (ms).
    expected_fps : float  Expected frames per second.
    notes : str  Additional notes.
    """
    name: str = ""
    cpu_cores: int = 4
    ram_mb: int = 4096
    measured: bool = False
    expected_latency_ms: float = 0.0
    expected_fps: float = 0.0
    recommended_n_threads: int = 2
    notes: str = ""


# These are ESTIMATES — NOT measured on actual hardware.
# Desktop measured latency: ~2200 ms per image (50-star catalog).
# NOTE: latency is dominated by RANSAC (50 iterations × 10 stars × 49 catalog).
# With vectorized RANSAC (Phase 6), desktop latency drops to ~800 ms.
# Scaled estimates use CPU performance relative to a mid-range desktop i7.

EDGE_PROFILES = [
    EdgeProfile(
        name="Raspberry Pi 4 Model B (4 GB)",
        cpu_cores=4,
        ram_mb=4096,
        measured=False,
        expected_latency_ms=8000.0,   # ESTIMATED: ~4x slower than desktop
        expected_fps=0.125,
        recommended_n_threads=2,
        notes=(
            "ESTIMATED (not measured on device). "
            "ARM Cortex-A72 @ 1.8 GHz, ~4x slower than i7-class desktop. "
            "Sufficient for low-rate attitude updates (0.1 Hz). "
            "RAM footprint <50 MB — well within 4 GB."
        ),
    ),
    EdgeProfile(
        name="NVIDIA Jetson Nano (4 GB)",
        cpu_cores=4,
        ram_mb=4096,
        measured=False,
        expected_latency_ms=5000.0,   # ESTIMATED: ~2.5x slower
        expected_fps=0.2,
        recommended_n_threads=3,
        notes=(
            "ESTIMATED (not measured on device). "
            "ARM Cortex-A57 @ 1.43 GHz. CPU-only mode (no GPU for this pipeline). "
            "GPU acceleration possible for CNN if PyTorch added in Phase 7."
        ),
    ),
    EdgeProfile(
        name="Intel NUC (i5-class x86)",
        cpu_cores=4,
        ram_mb=8192,
        measured=False,
        expected_latency_ms=1200.0,   # ESTIMATED: similar to desktop
        expected_fps=0.8,
        recommended_n_threads=4,
        notes=(
            "ESTIMATED. Similar performance class to development machine. "
            "Should achieve near-desktop latency with all optimizations."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Edge configuration template
# ---------------------------------------------------------------------------


def get_edge_config(base_config: dict, profile: EdgeProfile) -> dict:
    """Return a config dict optimized for an edge hardware profile.

    Parameters
    ----------
    base_config : dict  Base project configuration dict.
    profile : EdgeProfile  Target hardware profile.

    Returns
    -------
    dict  Modified configuration suitable for edge deployment.
    """
    import copy
    cfg = copy.deepcopy(base_config)

    # Limit CPU threads to avoid thrashing
    cfg.setdefault("optimization", {})
    cfg["optimization"]["n_threads"] = profile.recommended_n_threads
    cfg["optimization"]["vectorize"] = True
    cfg["optimization"]["cache_catalog"] = True

    # Keep catalog small (only bright stars needed for initial attitude)
    cfg.setdefault("dataset", {})
    current_mag = cfg["dataset"].get("catalog_mag_limit", 6.5)
    cfg["dataset"]["catalog_mag_limit"] = min(current_mag, 4.0)  # brighter = fewer stars = faster

    return cfg


# ---------------------------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------------------------


def print_deployment_summary(measured_result=None) -> None:
    """Print edge deployment readiness summary.

    Parameters
    ----------
    measured_result : BenchmarkResult, optional
        Actual measured benchmark result from this machine.
        If provided, shows it alongside the edge estimates.
    """
    print("\n" + "="*70)
    print("  EDGE DEPLOYMENT READINESS SUMMARY")
    print("="*70)

    print("\n  Runtime dependencies (inference only):")
    for dep in INFERENCE_DEPENDENCIES:
        print(f"    {dep}")

    print("\n  Not required for inference (training-only):")
    for dep in TRAINING_ONLY_DEPENDENCIES:
        print(f"    {dep}")

    if measured_result is not None:
        print(f"\n  Measured desktop latency  : {measured_result.mean_latency_ms:.1f} ms MEASURED")
        print(f"  Measured desktop FPS      : {measured_result.fps:.2f} MEASURED")
        print(f"  Measured peak memory      : {measured_result.peak_memory_mb:.2f} MB MEASURED")

    print(f"\n  {'Platform':<35} {'Latency':>10} {'FPS':>8} {'Status':>12}")
    print("  " + "-"*68)
    for p in EDGE_PROFILES:
        status = "ESTIMATED" if not p.measured else "MEASURED"
        print(f"  {p.name:<35} {p.expected_latency_ms:>9.0f}ms {p.expected_fps:>8.3f} {status:>12}")

    print()
    print("  CPU-only inference: YES (no GPU required)")
    print("  Minimum RAM required: ~50 MB (catalog + pipeline)")
    print("  Model checkpoint: ~2 MB (.pkl sklearn RandomForest)")
    print()
    print("  IMPORTANT: Edge latency figures are ESTIMATES based on CPU")
    print("  performance scaling. They have NOT been measured on actual")
    print("  edge hardware. Run benchmark.py on target device to measure.")
    print("="*70)
