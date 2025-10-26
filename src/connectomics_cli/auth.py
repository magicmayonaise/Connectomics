"""Authentication helpers for talking to CAVE."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from caveclient import CAVEclient

TOKEN_FILENAMES = (
    "cave-secret.json",
    "secrets.json",
)
TOKEN_DIRS = (
    Path.home() / ".cloudvolume" / "secrets",
    Path.home() / ".secrets",
)


class CaveAuthenticationError(RuntimeError):
    """Raised when an OAuth2 token cannot be located."""


@dataclass(slots=True, frozen=True)
class TokenDiscoveryResult:
    """Outcome of token discovery."""

    token: str | None
    source: Path | str | None


def _candidate_paths() -> Iterable[Path]:
    token_file = os.getenv("CAVE_TOKEN_FILE")
    if token_file:
        yield Path(token_file)
    for directory in TOKEN_DIRS:
        for filename in TOKEN_FILENAMES:
            yield directory / filename


def _read_token(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf8").strip()
    if not text:
        return None
    if text.startswith("{"):
        data = json.loads(text)
        for key in ("token", "access_token", "oauth2_token"):
            candidate = data.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        return None
    return text


def discover_token() -> TokenDiscoveryResult:
    """Return a token string if one can be discovered.

    The function searches the ``CAVE_TOKEN`` environment variable first, then a
    configurable list of files that mirrors the behaviour of the reference
    ``caveclient`` implementation. The source is returned for diagnostics.
    """

    env_token = os.getenv("CAVE_TOKEN")
    if env_token:
        return TokenDiscoveryResult(env_token, "CAVE_TOKEN")
    for candidate in _candidate_paths():
        token = _read_token(candidate)
        if token:
            return TokenDiscoveryResult(token, candidate)
    return TokenDiscoveryResult(None, None)


def build_cave_client(dataset: str) -> CAVEclient:
    """Create a :class:`caveclient.CAVEclient` with helpful auth errors."""

    discovery = discover_token()
    token = discovery.token
    try:
        client = CAVEclient(dataset, auth_token=token)
    except Exception as exc:  # pragma: no cover - defensive, depends on client implementation
        raise CaveAuthenticationError(_format_auth_error(dataset, discovery.source)) from exc
    if token is None:
        # Ensure downstream code sees a descriptive failure if the client attempts
        # to lazily authenticate later.
        try:
            client.auth.get_token()
        except Exception as exc:  # pragma: no cover - depends on client implementation
            raise CaveAuthenticationError(_format_auth_error(dataset, discovery.source)) from exc
    return client


def _format_auth_error(dataset: str, source: Path | str | None) -> str:
    hint = (
        "Run 'caveclient login --datastack {dataset}' or store your OAuth token at "
        "~/.cloudvolume/secrets/cave-secret.json (or set CAVE_TOKEN/CAVE_TOKEN_FILE)."
    ).format(dataset=dataset)
    if source is None:
        return f"No CAVE OAuth2 token found for dataset '{dataset}'. {hint}"
    return f"Failed to use token from {source!s} for dataset '{dataset}'. {hint}"
