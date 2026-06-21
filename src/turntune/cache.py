"""Persistent cache of per-frame detector signals (.npz).

The cache key is (scenario_id, detector_name, detector_version, frame_ms) — policy /
tuning params are deliberately EXCLUDED, because the whole point is that one cached
signal serves the entire parameter sweep. Bumping a detector's `version` cleanly
invalidates stale signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import FrameSignal


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


class SignalCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: tuple) -> Path:
        scenario_id, det_name, det_version, frame_ms = key
        return (
            self.root
            / "signals"
            / f"{_safe(det_name)}@{_safe(det_version)}"
            / f"{int(frame_ms)}ms"
            / f"{_safe(scenario_id)}.npz"
        )

    def get(self, key: tuple) -> FrameSignal | None:
        path = self._path(key)
        if not path.exists():
            return None
        scenario_id, det_name, det_version, frame_ms = key
        with np.load(path) as data:
            prob = data["speech_prob"]
        return FrameSignal(scenario_id, det_name, det_version, int(frame_ms), prob)

    def put(self, key: tuple, signal: FrameSignal) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, speech_prob=np.asarray(signal.speech_prob, dtype=np.float32))
