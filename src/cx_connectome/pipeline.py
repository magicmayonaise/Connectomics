"""Command line interface for running the synthetic connectome pipeline."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStep:
    """Represents a single stage in the pipeline."""

    index: int
    key: str
    label: str

    def output_path(self, run_dir: Path) -> Path:
        """Return the destination path for this step's cached output."""
        slug = slugify(self.key)
        return run_dir / f"{self.index:02d}_{slug}.json"


@dataclass
class StepResult:
    """Outcome of a pipeline step execution."""

    step: PipelineStep
    status: str
    path: Path
    duration: float
    cached: bool


@dataclass
class PipelineContext:
    """Shared state for the duration of the pipeline run."""

    dataset: str
    server: Optional[str]
    materialization: Optional[int]
    at_timestamp: Optional[dt.datetime]
    n1_roots_path: Path
    n1_roots: List[int]
    thresholds: List[float]
    table_overrides: Dict[str, str]
    base_dir: Path
    run_id: str
    run_dir: Path
    force: bool

    @property
    def identifier(self) -> str:
        return self.run_id


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------


PIPELINE_STEPS: Sequence[PipelineStep] = (
    PipelineStep(0, "bootstrap", "Bootstrap request"),
    PipelineStep(1, "fetch-metadata", "Fetch metadata"),
    PipelineStep(2, "stage-segments", "Stage segments"),
    PipelineStep(3, "build-contacts", "Build contacts"),
    PipelineStep(4, "score-contacts", "Score contacts"),
    PipelineStep(5, "threshold-contacts", "Threshold contacts"),
    PipelineStep(6, "join-annotations", "Join annotations"),
    PipelineStep(7, "emit-edges", "Emit edges"),
    PipelineStep(8, "compute-components", "Compute components"),
    PipelineStep(9, "derive-metrics", "Derive metrics"),
    PipelineStep(10, "summarize", "Summarize"),
    PipelineStep(11, "archive", "Archive"),
    PipelineStep(12, "finalize", "Finalize"),
)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cx-pipeline",
        description="Run the synthetic 0→12 connectome pipeline with caching.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Execute pipeline stages 0 through 12 in order."
    )

    run_parser.add_argument(
        "-dataset",
        "--dataset",
        required=True,
        choices=["flywire"],
        help="Dataset to operate on (currently only 'flywire' is supported).",
    )
    run_parser.add_argument(
        "-server",
        "--server",
        default=None,
        help="Optional CAVE server URL or alias.",
    )

    timing_group = run_parser.add_mutually_exclusive_group(required=True)
    timing_group.add_argument(
        "-materialization",
        "--materialization",
        type=int,
        help="Materialization identifier to query.",
    )
    timing_group.add_argument(
        "-at-timestamp",
        "--at-timestamp",
        type=parse_timestamp,
        help="Point-in-time timestamp (ISO-8601, e.g. 2025-07-21T00:00:00Z).",
    )

    run_parser.add_argument(
        "-n1-roots",
        "--n1-roots",
        type=Path,
        required=True,
        help="Path to a file containing newline or whitespace separated root IDs.",
    )
    run_parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=float,
        default=[],
        metavar="FLOAT",
        help="Threshold(s) to apply (may be repeated).",
    )
    run_parser.add_argument(
        "--table-override",
        dest="table_overrides",
        action="append",
        default=[],
        type=parse_table_override,
        metavar="KEY=VALUE",
        help="Override table sources using key=value pairs (may repeat).",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline_runs"),
        help="Directory where pipeline outputs should be stored.",
    )
    run_parser.add_argument(
        "-force",
        "--force",
        action="store_true",
        help="Recompute all steps even if cached outputs exist.",
    )

    return parser


def parse_timestamp(value: str) -> dt.datetime:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise argparse.ArgumentTypeError(
            "Expected ISO-8601 timestamp (e.g. 2025-07-21T00:00:00Z)."
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_table_override(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Override must look like key=value")
    key, override = value.split("=", 1)
    key = key.strip()
    override = override.strip()
    if not key:
        raise argparse.ArgumentTypeError("Override key cannot be empty")
    return key, override


# ---------------------------------------------------------------------------
# Core execution helpers
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":  # pragma: no cover - argparse enforces this
        parser.error("Unsupported command")

    try:
        context = build_context(args)
    except PipelineConfigurationError as exc:
        parser.error(str(exc))

    prepare_run_directory(context)
    results = run_pipeline(context)
    print_ledger(context, results)
    return 0


class PipelineConfigurationError(RuntimeError):
    """Raised when the provided CLI options are inconsistent."""


def build_context(args: argparse.Namespace) -> PipelineContext:
    dataset = args.dataset
    server = args.server
    materialization = args.materialization
    at_timestamp = args.at_timestamp

    n1_roots_path = args.n1_roots
    try:
        n1_roots = load_root_ids(n1_roots_path)
    except FileNotFoundError as exc:
        raise PipelineConfigurationError(f"Root file not found: {n1_roots_path}") from exc
    except ValueError as exc:
        raise PipelineConfigurationError(str(exc)) from exc

    thresholds = list(args.thresholds or [])
    table_overrides = consolidate_overrides(args.table_overrides or [])

    base_dir = args.output_dir.expanduser().resolve()
    run_id = build_run_id(dataset, server, materialization, at_timestamp)
    run_dir = base_dir / run_id

    return PipelineContext(
        dataset=dataset,
        server=server,
        materialization=materialization,
        at_timestamp=at_timestamp,
        n1_roots_path=n1_roots_path,
        n1_roots=n1_roots,
        thresholds=thresholds,
        table_overrides=table_overrides,
        base_dir=base_dir,
        run_id=run_id,
        run_dir=run_dir,
        force=args.force,
    )


def load_root_ids(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(path)

    roots: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in line.split():
                try:
                    roots.append(int(token))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid root ID '{token}' in {path}. Expected integers."
                    ) from exc

    if not roots:
        raise ValueError(f"No root IDs found in {path}.")

    return roots


def consolidate_overrides(pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for key, value in pairs:
        overrides[key] = value
    return overrides


def build_run_id(
    dataset: str,
    server: Optional[str],
    materialization: Optional[int],
    at_timestamp: Optional[dt.datetime],
) -> str:
    parts = [slugify(dataset)]
    if materialization is not None:
        parts.append(f"mat{materialization}")
    elif at_timestamp is not None:
        parts.append("ts" + at_timestamp.strftime("%Y%m%dT%H%M%SZ"))
    else:  # pragma: no cover - mutually exclusive group enforces this
        raise PipelineConfigurationError("Either materialization or timestamp required")

    if server:
        parts.append(slugify(server))

    return "-".join(filter(None, parts))


def prepare_run_directory(context: PipelineContext) -> None:
    context.base_dir.mkdir(parents=True, exist_ok=True)
    context.run_dir.mkdir(parents=True, exist_ok=True)

    config_path = context.run_dir / "run_config.json"
    config_payload = {
        "dataset": context.dataset,
        "server": context.server,
        "materialization": context.materialization,
        "at_timestamp": isoformat(context.at_timestamp),
        "n1_roots_file": str(context.n1_roots_path.resolve()),
        "n1_root_count": len(context.n1_roots),
        "thresholds": context.thresholds,
        "table_overrides": context.table_overrides,
        "run_id": context.run_id,
        "generated_at": isoformat(dt.datetime.now(dt.timezone.utc)),
    }

    if context.force or not config_path.exists():
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    roots_copy = context.run_dir / "n1_roots.txt"
    if context.force or not roots_copy.exists():
        with context.n1_roots_path.open("r", encoding="utf-8") as src:
            roots_copy.write_text(src.read(), encoding="utf-8")


def run_pipeline(context: PipelineContext) -> List[StepResult]:
    results: List[StepResult] = []
    for step in PIPELINE_STEPS:
        results.append(execute_step(step, context))
    return results


def execute_step(step: PipelineStep, context: PipelineContext) -> StepResult:
    output_path = step.output_path(context.run_dir)
    if output_path.exists() and not context.force:
        return StepResult(
            step=step,
            status="cached",
            path=output_path,
            duration=0.0,
            cached=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    payload = build_step_payload(step, context)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    duration = time.perf_counter() - start
    return StepResult(
        step=step,
        status="written",
        path=output_path,
        duration=duration,
        cached=False,
    )


def build_step_payload(step: PipelineStep, context: PipelineContext) -> Dict[str, object]:
    return {
        "step_index": step.index,
        "step_key": step.key,
        "step_label": step.label,
        "run_id": context.run_id,
        "dataset": context.dataset,
        "server": context.server,
        "materialization": context.materialization,
        "at_timestamp": isoformat(context.at_timestamp),
        "n1_roots": context.n1_roots,
        "thresholds": context.thresholds,
        "table_overrides": context.table_overrides,
        "generated_at": isoformat(dt.datetime.now(dt.timezone.utc)),
        "notes": f"Synthetic output artifact for step {step.label}",
    }


def print_ledger(context: PipelineContext, results: Sequence[StepResult]) -> None:
    executed = sum(1 for result in results if not result.cached)
    cached = sum(1 for result in results if result.cached)

    print(f"Run ID     : {context.run_id}")
    print(f"Dataset    : {context.dataset}")
    if context.server:
        print(f"Server     : {context.server}")
    if context.materialization is not None:
        print(f"Material   : {context.materialization}")
    elif context.at_timestamp is not None:
        print(f"Timestamp  : {isoformat(context.at_timestamp)}")
    print(f"Roots file : {context.n1_roots_path}")
    print(f"Root count : {len(context.n1_roots)}")
    if context.thresholds:
        print(f"Thresholds : {', '.join(str(t) for t in context.thresholds)}")
    if context.table_overrides:
        overrides = ", ".join(f"{k}={v}" for k, v in context.table_overrides.items())
        print(f"Overrides  : {overrides}")
    print(f"Output dir : {context.run_dir}")
    print()

    header = f"{'Step':<6}{'Stage':<24}{'Status':<10}{'Duration':>10}  Output"
    print(header)
    print("-" * len(header))
    for result in results:
        duration = f"{result.duration:.2f}s" if not result.cached else "-"
        print(
            f"{result.step.index:02d}    "
            f"{result.step.label:<24}"
            f"{result.status:<10}"
            f"{duration:>10}  "
            f"{result.path}"
        )

    print("-" * len(header))
    print(f"Executed {executed} step(s); {cached} cached.")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "run"


def isoformat(value: Optional[dt.datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
