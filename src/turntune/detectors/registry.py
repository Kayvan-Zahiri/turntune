"""Name -> Detector-factory registry.

Lets the CLI/server resolve a detector by string (e.g. "silero-vad") and lets new
adapters self-register on import, without the harness ever importing them directly.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Detector

_FACTORIES: dict[str, Callable[..., Detector]] = {}


def register(name: str) -> Callable[[Callable[..., Detector]], Callable[..., Detector]]:
    """Decorator: register a Detector factory (class or callable) under `name`."""

    def deco(factory: Callable[..., Detector]) -> Callable[..., Detector]:
        _FACTORIES[name] = factory
        return factory

    return deco


def create(name: str, **kwargs) -> Detector:
    if name not in _FACTORIES:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**kwargs)


def names() -> list[str]:
    return sorted(_FACTORIES)
