"""Harness invariant: fast vs realtime pacing must yield identical decisions.

EOT timestamps derive from frame index, not wall clock, so realtime pacing (which
just sleeps between frames) cannot change the result.
"""

from __future__ import annotations

import numpy as np

from turntune.harness import run_detector
from turntune.policy import silence_hangover
from turntune.types import FrameSignal, Scenario


class MeanAbsDetector:
    """Tiny frame-consuming detector: per-frame mean-abs amplitude as the signal."""

    name = "meanabs"
    version = "v1"
    frame_ms = 20

    def default_params(self):
        return {"speech_threshold": 0.1, "min_silence_s": 0.2}

    def param_space(self):
        return {}

    def extract(self, frames, scenario=None) -> FrameSignal:
        vals = [float(np.mean(np.abs(np.asarray(f)))) for f in frames]
        return FrameSignal("", self.name, self.version, self.frame_ms, np.asarray(vals, np.float32))

    def decide(self, signal, params):
        return silence_hangover(signal, params)


def test_fast_equals_realtime():
    # Short clip so the realtime sleep is tiny: 0.2s speech-ish + 0.2s silence.
    rng = np.random.default_rng(0)
    audio = np.concatenate(
        [
            rng.normal(0, 0.3, 16000 // 5).astype(np.float32),  # 0.2s "speech"
            np.zeros(16000 // 5, np.float32),  # 0.2s silence
        ]
    )
    sc = Scenario(
        id="s", audio=audio, sample_rate=16000, spans=[], true_eot_s=0.2, wav_path="s.wav"
    )
    det = MeanAbsDetector()

    fast = run_detector([sc], det, cache=None, pace="fast")
    realtime = run_detector([sc], det, cache=None, pace="realtime")

    assert np.array_equal(fast["s"].speech_prob, realtime["s"].speech_prob)
