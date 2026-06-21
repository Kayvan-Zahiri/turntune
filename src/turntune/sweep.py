"""Build the parameter grid for a sweep from a detector's param_space().

Default: 1-D dense sweep over the chosen axis (e.g. min_silence_s) with the other
knobs pinned to the current slider values. Kept out of the metrics engine on purpose.
"""

from __future__ import annotations

import numpy as np

from .detectors.base import Detector
from .types import PolicyParams


def axis_values(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        return [round(float(lo), 4)]
    vals = np.arange(lo, hi + step / 2.0, step)
    return [round(float(v), 4) for v in vals]


def build_grid(
    detector: Detector,
    sweep_axis: str,
    pinned: PolicyParams | None = None,
    *,
    mode: str = "1d",
) -> list[PolicyParams]:
    """Return the list of PolicyParams to evaluate for the curve."""
    space = detector.param_space()
    if sweep_axis not in space:
        raise KeyError(f"unknown sweep axis {sweep_axis!r}; options: {sorted(space)}")
    pinned = pinned or {}
    base = {name: float(pinned.get(name, ax.default)) for name, ax in space.items()}

    ax = space[sweep_axis]
    grid: list[PolicyParams] = []
    for v in axis_values(ax.lo, ax.hi, ax.step):
        point = dict(base)
        point[sweep_axis] = v
        grid.append(point)
    return grid
