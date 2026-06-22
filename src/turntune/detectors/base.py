"""Pluggable seam #1 — the Detector protocol.

A detector is split into two halves on purpose:

  extract(frames, scenario) -> FrameSignal
      The EXPENSIVE pass. Runs the model over 20ms/16kHz frames exactly once and
      returns a per-frame signal. Must be CAUSAL (no future audio) and is cached.
      `scenario` carries clip context (e.g. the transcript words) for detectors that
      need more than the audio; audio-only detectors (Silero) ignore it.

  decide(signal, params) -> EotDecision
      The CHEAP, PURE pass. Turns the cached signal + tuning params into an
      end-of-turn decision. Replayed across the whole sweep and on every slider
      drag, so it must be fast and side-effect free.

The harness and metrics import ONLY this protocol, never a concrete detector.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np

from ..types import Axis, EotDecision, FrameSignal, PolicyParams, Scenario


@runtime_checkable
class Detector(Protocol):
    name: str
    version: str
    frame_ms: int

    def default_params(self) -> PolicyParams:
        """Sensible starting knob values (used to seed the UI sliders)."""
        ...

    def param_space(self) -> dict[str, Axis]:
        """The tunable knobs — drives both the sweep grid and the UI sliders."""
        ...

    def extract(
        self, frames: Iterable[np.ndarray], scenario: Scenario | None = None
    ) -> FrameSignal:
        """EXPENSIVE, causal, cached: stream frames (+ optional clip context) -> signal."""
        ...

    def decide(self, signal: FrameSignal, params: PolicyParams) -> EotDecision:
        """CHEAP, pure: cached signal + params -> end-of-turn decision."""
        ...
