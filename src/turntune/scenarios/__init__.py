"""Built-in scenario loaders self-register on import."""

from __future__ import annotations

from . import (
    eot_bench,  # noqa: F401  (registers "eot-bench")
    fixtures,  # noqa: F401  (registers "fixtures")
)
from .registry import create, names, register

__all__ = ["create", "names", "register"]
