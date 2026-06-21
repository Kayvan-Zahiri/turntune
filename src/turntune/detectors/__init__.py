"""Built-in detectors self-register on import."""

from __future__ import annotations

from . import silero_vad  # noqa: F401  (registers "silero-vad")
from .registry import create, names, register

__all__ = ["create", "names", "register"]
