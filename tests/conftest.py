"""Shared test fixtures: a fake detector and synthetic-signal helpers.

These let the metrics/harness tests run with zero heavy deps and no network — they
prove the seams (Detector / ScenarioLoader contracts) on tiny, deterministic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from turntune.policy import silence_hangover
from turntune.types import Axis, EotDecision, FrameSignal, PolicyParams


class FakeDetector:
    """A detector whose `extract` just returns a pre-baked signal.

    Useful for testing the harness/metrics without running a real model.
    """

    name = "fake"
    version = "v1"
    frame_ms = 20

    def __init__(self, prob: np.ndarray | None = None):
        self._prob = prob

    def default_params(self) -> PolicyParams:
        return {k: ax.default for k, ax in self.param_space().items()}

    def param_space(self) -> dict[str, Axis]:
        return {
            "speech_threshold": Axis(0.1, 0.9, 0.05, 0.5, "Confidence"),
            "min_silence_s": Axis(0.1, 1.5, 0.05, 0.3, "Silence before EOT"),
            "timeout_s": Axis(0.0, 5.0, 0.5, 0.0, "Force EOT timeout (0=off)"),
        }

    def extract(self, frames) -> FrameSignal:
        prob = self._prob if self._prob is not None else np.zeros(0, np.float32)
        return FrameSignal("", self.name, self.version, self.frame_ms, prob)

    def decide(self, signal: FrameSignal, params: PolicyParams) -> EotDecision:
        return silence_hangover(signal, params)


def make_signal(speech_mask, frame_ms: int = 20, scenario_id: str = "s") -> FrameSignal:
    """Build a FrameSignal from a 0/1 (or bool) per-frame speech mask."""
    prob = np.asarray(speech_mask, dtype=np.float32)
    return FrameSignal(scenario_id, "fake", "v1", frame_ms, prob)


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
