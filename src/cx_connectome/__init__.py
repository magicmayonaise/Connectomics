"""
cx_connectome: core package.

We export the 'ci' subpackage lazily to avoid heavy imports (e.g., torch) at module import time.
"""
from importlib import import_module

__all__ = ["ci"]

def __getattr__(name: str):
    if name == "ci":
        return import_module("cx_connectome.ci")
    raise AttributeError(f"module {__name__!s} has no attribute {name!r}")
