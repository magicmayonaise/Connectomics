"""Configuration helpers for the Connectomics CLI.

The module centralises defaults to keep them consistent between the CLI, tests,
and downstream libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_DATASET = "flywire_fafb_production"
DEFAULT_MATERIALIZATION = 783
DEFAULT_CHUNK_SIZE = 5_000


@dataclass(frozen=True, slots=True)
class QueryContext:
    """Materialization parameters for a query.

    Parameters
    ----------
    dataset:
        The datastack name passed to :class:`caveclient.CAVEclient`.
    materialization:
        The integer materialization ID. ``None`` indicates a live query at a
        specific timestamp.
    timestamp:
        Optional timestamp for live queries. When provided it must be timezone
        aware.
    """

    dataset: str
    materialization: int | None
    timestamp: datetime | None

    def describe(self) -> str:
        """Return a human readable description.

        >>> QueryContext("ds", 10, None).describe()
        'dataset=ds materialization=10'
        >>> QueryContext("ds", None, datetime(2020, 1, 1, tzinfo=timezone.utc)).describe()
        'dataset=ds timestamp=2020-01-01T00:00:00+00:00'
        """

        if self.materialization is not None:
            return f"dataset={self.dataset} materialization={self.materialization}"
        if self.timestamp is None:
            return f"dataset={self.dataset} materialization=unspecified"
        return f"dataset={self.dataset} timestamp={self.timestamp.isoformat()}"


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO8601 timestamp string.

    Naive timestamps are interpreted as UTC for reproducibility.
    """

    if value is None or value.strip() == "":
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
