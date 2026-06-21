"""Pluggable seam #2 — the ScenarioLoader protocol.

A loader's whole job is to normalize some source (eot-bench, bundled fixtures, or your
own data later) into the common Scenario + SilenceSpan schema, so the rest of the
system stays dataset-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..types import Scenario


@runtime_checkable
class ScenarioLoader(Protocol):
    name: str

    def load(
        self,
        limit: int | None = None,
        language: str = "en",
        split: str = "validation",
    ) -> Iterable[Scenario]:
        """Yield normalized Scenario objects (audio @16k mono, hold/eot spans)."""
        ...
