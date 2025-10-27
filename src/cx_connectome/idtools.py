"""Utilities for updating CAVE root identifiers.

CAVE's lineage graph captures how supervoxels merge and split over time,
enabling deterministic translation from historical IDs to the roots that were
current at any chosen point in time.  By pinning a materialization (and
optionally a timestamp), callers can obtain reproducible answers even as the
underlying segmentation continues to evolve.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def update_root_ids(
    client: Any,
    roots: list[int],
    materialization: int | None,
    at_ts: str | None,
) -> dict[int, int]:
    """Resolve possibly stale root IDs to the current roots.

    Parameters
    ----------
    client:
        A :class:`caveclient.CAVEclient` (or compatible) instance.  Only the
        ``chunkedgraph`` accessor with a ``get_roots`` method is required.
    roots:
        Iterable of root IDs that may refer to historical state.
    materialization:
        Materialization identifier to pin lineage queries.  ``None`` requests
        the dataset's default materialization.
    at_ts:
        ISO 8601 timestamp indicating when the mapping should be evaluated.

    Returns
    -------
    dict[int, int]
        Mapping from each requested root ID to the root that is current at the
        requested materialization/timestamp.
    """

    if not roots:
        logger.info("No roots provided; nothing to update.")
        return {}

    chunkedgraph = getattr(client, "chunkedgraph", None)
    if chunkedgraph is None:
        raise AttributeError("Client does not expose a 'chunkedgraph' accessor")

    try:
        get_roots = chunkedgraph.get_roots
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError("chunkedgraph accessor does not provide get_roots") from exc

    # Deduplicate while preserving order so that we do not request the same ID
    # more than once from the service.  This improves latency and reduces the
    # risk of server throttling when callers provide large lists of IDs.
    deduped_roots: list[int] = list(dict.fromkeys(int(r) for r in roots))
    if len(deduped_roots) != len(roots):
        logger.debug(
            "Deduplicated %d input roots down to %d unique roots before querying lineage.",
            len(roots),
            len(deduped_roots),
        )

    query_kwargs: dict[str, Any] = {}
    if materialization is not None:
        query_kwargs["materialization"] = materialization
    if at_ts is not None:
        query_kwargs["timestamp"] = at_ts

    logger.info(
        "Resolving %d unique roots using CAVE lineage (materialization=%s, timestamp=%s)",
        len(deduped_roots),
        materialization if materialization is not None else "default",
        at_ts if at_ts is not None else "current",
    )

    resolved = get_roots(deduped_roots, **query_kwargs)
    root_lookup = _normalise_roots_response(deduped_roots, resolved)

    missing = [root for root in deduped_roots if root not in root_lookup]
    if missing:
        raise KeyError(
            "Did not receive lineage results for the following roots: "
            + ", ".join(str(r) for r in missing)
        )

    mapping = {root: int(root_lookup[int(root)]) for root in deduped_roots}

    if any(src != dst for src, dst in mapping.items()):
        logger.info("Translated %d roots to their current IDs.", len(mapping))
    else:
        logger.info("All %d roots were already current.", len(mapping))

    return mapping


def _normalise_roots_response(
    deduped_roots: Sequence[int], result: Any
) -> dict[int, int]:
    """Convert ``chunkedgraph.get_roots`` responses into a dictionary."""

    if isinstance(result, Mapping):
        return {int(k): int(v) for k, v in result.items()}

    # Some caveclient versions return pandas objects; prefer converting via
    # ``to_dict`` when available.
    if hasattr(result, "to_dict"):
        converted = result.to_dict()
        if isinstance(converted, Mapping):
            return {int(k): int(v) for k, v in converted.items()}
        result = converted

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) != len(deduped_roots):
            raise ValueError(
                "Expected %d roots, received %d results" % (len(deduped_roots), len(result))
            )
        return {int(src): int(dst) for src, dst in zip(deduped_roots, result)}

    raise TypeError(
        "Unsupported response type from chunkedgraph.get_roots: %s" % type(result).__name__
    )


def _read_root_file(path: Path) -> list[int]:
    """Load root IDs from a text file."""

    roots: list[int] = []
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                roots.append(int(stripped))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid root ID '{stripped}' in {path}") from exc
    return roots


def _cli_update_ids(args: argparse.Namespace) -> int:
    """Entry point for the ``cx-cave update-ids`` sub-command."""

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise SystemExit(f"Unknown log level: {args.log_level}")
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    dataset = args.dataset or os.environ.get("CAVE_DATASET")
    if not dataset:
        raise SystemExit("A dataset must be supplied via --dataset or CAVE_DATASET")

    server = args.server or os.environ.get("CAVE_SERVER")

    try:
        from caveclient import CAVEclient
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit("caveclient must be installed to use this command") from exc

    client = CAVEclient(dataset, server_address=server)

    roots = _read_root_file(args.roots)
    if not roots:
        logger.warning("No roots were read from %s; nothing to do.", args.roots)
        return 0

    mapping = update_root_ids(client, roots, args.materialization, args.at_ts)

    lines = [f"{src}\t{dst}" for src, dst in mapping.items()]
    output = "\n".join(lines)

    if args.output:
        args.output.write_text(output + ("\n" if output else ""), encoding="utf8")
    else:
        if output:
            sys.stdout.write(output + "\n")
        else:
            logger.info("No mapping changes to report.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the root CLI parser used by ``main``."""

    parser = argparse.ArgumentParser(prog="cx-cave")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )

    subparsers = parser.add_subparsers(dest="command")
    update = subparsers.add_parser(
        "update-ids",
        help="Translate stale root IDs to the current IDs using CAVE lineage.",
    )
    update.add_argument(
        "--roots",
        type=Path,
        required=True,
        help="Path to a text file containing one root ID per line.",
    )
    update.add_argument(
        "--materialization",
        type=int,
        default=None,
        help="Materialization to query (defaults to dataset default).",
    )
    update.add_argument(
        "--at-ts",
        dest="at_ts",
        default=None,
        help="Optional ISO timestamp to evaluate lineage at.",
    )
    update.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (defaults to $CAVE_DATASET).",
    )
    update.add_argument(
        "--server",
        default=None,
        help="Override the CAVE server address (defaults to $CAVE_SERVER).",
    )
    update.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the mapping to (defaults to stdout).",
    )
    update.set_defaults(func=_cli_update_ids)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point exposing the ``cx-cave`` command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI use
    sys.exit(main())
