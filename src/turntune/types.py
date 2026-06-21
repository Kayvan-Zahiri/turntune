"""Shared vocabulary for turntune — behavior-free data types.

Every layer (loaders, detectors, harness, metrics, server) speaks in these types.
Defining them in one place keeps the audio timeline consistent end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np

# Audio is normalized to this grid everywhere downstream of the loader, so every
# timestamp (ground truth, frame index, EOT decision) is directly comparable.
SR: int = 16_000
FRAME_MS: int = 20
FRAME_SAMPLES: int = SR * FRAME_MS // 1000  # 320


@dataclass(frozen=True)
class SilenceSpan:
    """A stretch of silence in a scenario.

    `hold` = a mid-turn pause (the user is NOT done — firing here is a cutoff).
    The final span is labelled `eot`: its start is the true end of turn.
    """

    start_s: float
    end_s: float
    label: Literal["hold", "eot"]


@dataclass
class Scenario:
    """One normalized conversation clip with ground-truth turn boundaries."""

    id: str
    audio: np.ndarray  # float32 mono in [-1, 1] @ SR
    sample_rate: int
    spans: list[SilenceSpan]
    true_eot_s: float  # start of the `eot` span
    wav_path: str  # materialized 16-bit PCM wav for browser playback
    meta: dict = field(default_factory=dict)  # language, split, source id, transcript, ...


@dataclass(frozen=True)
class FrameSignal:
    """The cached, policy-independent output of a detector's expensive pass.

    `speech_prob[i]` is the model's signal for the i-th `frame_ms` frame. For a VAD
    this is P(speech); for a semantic model it could be P(end-of-turn). `decide()`
    interprets it together with the tuning params.
    """

    scenario_id: str
    detector_name: str
    detector_version: str
    frame_ms: int
    speech_prob: np.ndarray  # shape (n_frames,), values in [0, 1]

    def t_s(self, i: int) -> float:
        """End time (seconds) of frame i — decisions are reported at frame end."""
        return (i + 1) * self.frame_ms / 1000.0


class Axis(NamedTuple):
    """A tunable knob — drives BOTH the sweep grid and the auto-generated UI slider."""

    lo: float
    hi: float
    step: float
    default: float
    label: str
    help: str = ""


# A concrete, JSON-serializable set of tuning knob values, e.g.
# {"speech_threshold": 0.5, "min_silence_s": 0.6, "min_eot_delay_s": 0.0, "timeout_s": 0.0}
PolicyParams = dict


class EotDecision(NamedTuple):
    """The detector's end-of-turn decision for one scenario at one param setting."""

    fired: bool
    eot_s: float | None


class ScenarioResult(NamedTuple):
    """Per-scenario score against ground truth."""

    scenario_id: str
    klass: Literal["correct", "cutoff", "missed"]
    eot_s: float | None
    true_eot_s: float
    latency_s: float | None  # set only for `correct`


class SweepPoint(NamedTuple):
    """One point on the latency-vs-cutoff curve: aggregate metrics at given params."""

    params: PolicyParams
    cutoff_rate: float
    p50_latency_s: float
    p90_latency_s: float
    no_fire_rate: float
    n: int
