"""The endpointing policies — the cheap, pure functions the sweep hammers.

`silence_hangover` powers the silence-based Silero VAD; `content_gated_hangover` powers
the transcript-based semantic detector. Both are vectorized and side-effect free so the
sweep can replay them thousands of times. Not a public registry seam in v0.
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
      - params["timeout_s"] (eot-bench `timeout`; 0/None = off) force-fires once silence
        reaches it. For a pure VAD action_delay already fires by then, so timeout is
        usually inert here — but it is applied symmetrically with the semantic detector
        so both share the exact same three-knob policy.

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

    seen_speech = last_speech_pos >= 0
    fired = seen_speech & (run_frames >= k_min)
    timeout = float(params.get("timeout_s") or 0.0)
    if timeout > 0:
        fired = fired | (seen_speech & (run_frames >= max(1, math.ceil(timeout / dt))))
    if not fired.any():
        return EotDecision(False, None)

    first = int(np.argmax(fired))  # argmax returns the first True
    return EotDecision(True, round((first + 1) * dt, 3))


def content_gated_hangover(signal: FrameSignal, params: PolicyParams) -> EotDecision:
    """End-of-turn when there's enough trailing silence AND the transcript looks done.

    A transcript-based detector packs two things into one signal array: a frame is < 0
    while a word is being spoken (sentinel), and during a silence gap it holds
    P(end-of-turn) for the transcript seen so far. Fire at the first silence frame where
    the trailing silence run >= params["min_silence_s"] (action_delay) AND the EoT prob
    >= params["eot_threshold"]; or, if params["timeout_s"] > 0, force after that much
    silence regardless of confidence.

    Pure and vectorized, like silence_hangover, so the sweep can replay it cheaply.
    """
    s = signal.speech_prob
    n = len(s)
    if n == 0:
        return EotDecision(False, None)

    dt = signal.frame_ms / 1000.0
    thr = float(params["eot_threshold"])
    k_min = max(1, math.ceil(float(params["min_silence_s"]) / dt))

    idx = np.arange(n)
    is_speech = s < 0.0
    last_speech_pos = np.maximum.accumulate(np.where(is_speech, idx, -1))
    run_frames = idx - last_speech_pos  # consecutive silence frames since the last word
    seen_speech = last_speech_pos >= 0

    fired = seen_speech & (run_frames >= k_min) & (s >= thr)
    timeout = float(params.get("timeout_s") or 0.0)
    if timeout > 0:
        fired = fired | (seen_speech & (run_frames >= max(1, math.ceil(timeout / dt))))

    if not fired.any():
        return EotDecision(False, None)
    first = int(np.argmax(fired))
    return EotDecision(True, round((first + 1) * dt, 3))
