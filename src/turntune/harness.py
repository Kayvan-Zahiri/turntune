"""Streaming runner: stream each scenario's audio through a detector once, cache the
per-frame signal.

The expensive neural pass (`detector.extract`) runs at most once per
(scenario, detector); every subsequent tuning sweep replays only the cheap
`decide()` over the cached signal. EOT timestamps derive from frame index, not wall
clock, so the fast (sweep) path and the realtime (demo) path produce identical
decisions.
"""

from __future__ import annotations

from collections.abc import Iterable

from .audio import iter_frames
from .cache import SignalCache
from .detectors.base import Detector
from .types import FrameSignal, Scenario


def run_detector(
    scenarios: Iterable[Scenario],
    detector: Detector,
    cache: SignalCache | None = None,
    *,
    pace: str = "fast",
    progress: bool = False,
) -> dict[str, FrameSignal]:
    """Return {scenario_id: FrameSignal}, computing+caching on miss, reusing on hit."""
    scenarios = list(scenarios)
    items: Iterable[Scenario] = scenarios
    if progress:
        try:
            from tqdm import tqdm

            items = tqdm(scenarios, desc=f"extract [{detector.name}]", unit="clip")
        except ImportError:
            pass

    out: dict[str, FrameSignal] = {}
    for sc in items:
        key = (sc.id, detector.name, detector.version, detector.frame_ms)
        sig = cache.get(key) if cache is not None else None
        if sig is None:
            frames = iter_frames(sc.audio, sc.sample_rate, detector.frame_ms, pace=pace)
            raw = detector.extract(frames)
            # Stamp the scenario id (extract() returns it blank) — FrameSignal is frozen.
            sig = FrameSignal(
                sc.id, detector.name, detector.version, detector.frame_ms, raw.speech_prob
            )
            if cache is not None:
                cache.put(key, sig)
        out[sc.id] = sig
    return out
