from __future__ import annotations

from importlib import import_module

__all__ = ["ops"]


def __getattr__(name: str):
    if name == "ops":
        return import_module(".ops", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
