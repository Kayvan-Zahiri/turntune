"""Built-in detectors self-register on import."""

from __future__ import annotations

from . import (
    semantic,  # noqa: F401  (registers "semantic-turn"; heavy deps load lazily)
    silero_vad,  # noqa: F401  (registers "silero-vad")
)
from .registry import create, names, register

__all__ = ["create", "names", "register"]
