"""Worked example: a custom turntune detector in ~40 lines.

Run conceptually:
    turntune serve --detector energy-vad

This toy detector marks a frame as speech when its short-time energy exceeds a
threshold, then reuses turntune's shared silence-hangover policy for the actual
end-of-turn decision. It exists to show the two-method contract; it is NOT meant to
be a good VAD.

The contract (see CONTRIBUTING.md):
  - extract(frames): EXPENSIVE, causal, cached -> a per-frame FrameSignal
  - decide(signal, params): CHEAP, pure -> an EotDecision
  - param_space(): the knobs (drive both the sweep and the UI sliders)
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from turntune.detectors.registry import register
from turntune.policy import silence_hangover
from turntune.types import Axis, EotDecision, FrameSignal, PolicyParams


@register("energy-vad")
class EnergyVadDetector:
    name = "energy-vad"
    version = "v1"
    frame_ms = 20

    def default_params(self) -> PolicyParams:
        return {k: ax.default for k, ax in self.param_space().items()}

    def param_space(self) -> dict[str, Axis]:
        return {
            # Here speech_threshold is interpreted as an RMS-energy cutoff, normalized
            # into [0, 1] in extract() so the same policy code applies.
            "speech_threshold": Axis(0.05, 0.9, 0.05, 0.3, "Energy threshold"),
            "min_silence_s": Axis(0.1, 1.5, 0.05, 0.6, "Silence before EOT"),
        }

    def extract(self, frames: Iterable[np.ndarray]) -> FrameSignal:
        # Per-frame RMS energy, squashed into [0, 1]. Causal: depends only on the
        # current frame. (A real model would carry state across frames.)
        probs = []
        for frame in frames:
            rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
            probs.append(min(1.0, rms * 4.0))
        return FrameSignal(
            "", self.name, self.version, self.frame_ms, np.asarray(probs, dtype=np.float32)
        )

    def decide(self, signal: FrameSignal, params: PolicyParams) -> EotDecision:
        return silence_hangover(signal, params)
