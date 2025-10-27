"""Utilities for working with connectome-derived data."""

from importlib import metadata

__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("cx-connectome")
        except metadata.PackageNotFoundError:  # pragma: no cover - best effort
            return "0"
    raise AttributeError(name)
