"""Defaults and run configuration — the single source of constants.

No logic lives here; just values (some marked TODO to confirm during implementation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Scenario source ---------------------------------------------------------
# Verified live on Hugging Face: one config per language ("en", "de", ...), the only
# split is "validation" (~400 turns/lang for big languages), audio is already 16 kHz,
# dataset license is CC-BY-4.0. See docs/metrics.md for the ground-truth mapping.
DATASET_ID = "livekit/eot-bench-data"
DEFAULT_LANGUAGE = "en"
DEFAULT_SPLIT = "validation"
DEFAULT_LIMIT = 100  # keep first-run download small for the 5-minute promise

# --- Detector / model --------------------------------------------------------
DEFAULT_DETECTOR = "silero-vad"
# Silero VAD v5 ONNX (~2.3 MB, MIT). Downloaded + sha-checked on first run, then
# cached offline. Verified I/O: input[N,N], state[2,N,128], sr(int64) -> output[N,1].
SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_ONNX_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

# --- Scoring -----------------------------------------------------------------
# Tolerance (s) around true_eot_s: firing earlier than this is a cutoff.
DEFAULT_TOLERANCE_S = 0.2  # TODO(step 2-3): tune against eot-bench span semantics

# --- Sweep -------------------------------------------------------------------
# Default Pareto sweep axis (the knob varied along the curve). The detector's
# param_space() supplies the lo/hi/step; this just names which knob.
DEFAULT_SWEEP_AXIS = "min_silence_s"

# --- Server ------------------------------------------------------------------
DEFAULT_PORT = 8000


def cache_dir() -> Path:
    """Where downloaded models, eot-bench data, materialized wavs, and signal npz live.

    Defaults to ./.turntune_cache (gitignored); override with $TURNTUNE_CACHE.
    """
    return Path(os.environ.get("TURNTUNE_CACHE", ".turntune_cache")).resolve()


@dataclass
class RunConfig:
    """Resolved configuration for a single turntune run."""

    detector: str = DEFAULT_DETECTOR
    loader: str = "eot-bench"
    language: str = DEFAULT_LANGUAGE
    limit: int = DEFAULT_LIMIT
    realtime: bool = False
    sweep_axis: str = DEFAULT_SWEEP_AXIS
    tolerance_s: float = DEFAULT_TOLERANCE_S
    port: int = DEFAULT_PORT
    open_browser: bool = False
    cache_root: Path = field(default_factory=cache_dir)
