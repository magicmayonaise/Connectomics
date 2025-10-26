"""Typed stubs for signed excitatory-inhibitory block logic."""

from __future__ import annotations

from typing import Literal

__all__ = ["SignedMode"]

SignedMode = Literal["net", "blocks"]
"""Allowed aggregation modes for signed connectivity."""
