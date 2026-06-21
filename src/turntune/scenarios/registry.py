"""Name -> ScenarioLoader-factory registry (mirrors the detector registry)."""

from __future__ import annotations

from collections.abc import Callable

from .base import ScenarioLoader

_FACTORIES: dict[str, Callable[..., ScenarioLoader]] = {}


def register(name: str) -> Callable[[Callable[..., ScenarioLoader]], Callable[..., ScenarioLoader]]:
    def deco(factory: Callable[..., ScenarioLoader]) -> Callable[..., ScenarioLoader]:
        _FACTORIES[name] = factory
        return factory

    return deco


def create(name: str, **kwargs) -> ScenarioLoader:
    if name not in _FACTORIES:
        raise KeyError(f"unknown loader {name!r}; available: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**kwargs)


def names() -> list[str]:
    return sorted(_FACTORIES)
