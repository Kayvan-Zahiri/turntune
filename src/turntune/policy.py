"""The endpointing policy — the cheap, pure function the sweep hammers.

Shared so that any signal-emitting detector (the shipped Silero VAD, or a future
semantic model) can reuse the same silence-hangover logic. Not a public registry seam
in v0, just an internal helper.
"""

from __future__ import annotations

import math

import numpy as np

from .types import EotDecision, FrameSignal, PolicyParams


def silence_hangover(signal: FrameSignal, params: PolicyParams) -> EotDecision:
    """Declare end-of-turn after enough trailing silence following speech.

    Vectorized O(n_frames):
      - a frame counts as speech when prob >= params["speech_threshold"]
        (eot-bench `threshold`)
      - fire at the first frame where the trailing run of silence since the last
        speech frame reaches params["min_silence_s"] (eot-bench `action_delay`)
      - params["timeout_s"] (eot-bench `timeout`; 0/None = off) is accepted for
        parity / future gated detectors; for a pure VAD it never fires earlier than
        action_delay, so it is inert here.

    Pure and side-effect free so it can be replayed thousands of times across the
    sweep and on every slider drag.
    """
    prob = signal.speech_prob
    n = len(prob)
    if n == 0:
        return EotDecision(False, None)

    dt = signal.frame_ms / 1000.0
    thr = float(params["speech_threshold"])
    k_min = max(1, math.ceil(float(params["min_silence_s"]) / dt))

    is_speech = prob >= thr
    idx = np.arange(n)
    # index of the most recent speech frame at or before i (-1 if none yet)
    last_speech_pos = np.maximum.accumulate(np.where(is_speech, idx, -1))
    # consecutive silence frames since the last speech frame
    run_frames = idx - last_speech_pos

    fired = (run_frames >= k_min) & (last_speech_pos >= 0)
    if not fired.any():
        return EotDecision(False, None)

    first = int(np.argmax(fired))  # argmax returns the first True
    return EotDecision(True, round((first + 1) * dt, 3))
