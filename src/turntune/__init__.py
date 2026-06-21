"""turntune — tune turn-taking in voice agents.

See the latency-vs-cutoff tradeoff, find where the agent talks over people, and dial
in the endpointing policy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import (
    Axis,
    EotDecision,
    FrameSignal,
    PolicyParams,
    Scenario,
    ScenarioResult,
    SilenceSpan,
    SweepPoint,
)

__all__ = [
    "__version__",
    "Axis",
    "EotDecision",
    "FrameSignal",
    "PolicyParams",
    "Scenario",
    "ScenarioResult",
    "SilenceSpan",
    "SweepPoint",
]
