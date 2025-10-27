"""
cx_connectome.ci: Connectome Interpreter utilities.

Modules are imported lazily to minimize import-time overhead and dependency requirements.
"""
from importlib import import_module

__all__ = [
    "effective",
    "signed",
    "paths",
    "dynamics",
    "optimize",
    "metrics",
    "state_overlay",
]

def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!s} has no attribute {name!r}")
