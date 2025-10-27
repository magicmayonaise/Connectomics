<<<<<<< HEAD
# Connectomics CLI

Connectomics CLI provides reproducible workflows for querying the [CAVE](https://global.daf-apis.com/cave/doc) materialization service and summarising synaptic connectivity from public datasets such as FlyWire.

## Features

- OAuth2-aware authentication with actionable guidance when no token is present.
- Materialization-aware querying that respects CAVE row caps by chunking requests.
- Automatic table discovery for synapse and cell type annotation schemas.
- Streaming aggregation that yields tidy `pandas` summaries and NetworkX graphs.
- Deterministic output artefacts (CSV, Parquet, SVG, PNG) stored under `./out` with a JSON run summary for provenance.

## Installation

```bash
pip install -e .
```

Python 3.13 or newer is required. Installing in a virtual environment is recommended.

## Authentication

The CLI looks for an OAuth2 token using the same conventions as `caveclient`. If no token is discovered, a clear error message explains how to generate one:

1. Obtain credentials from https://global.daf-apis.com/cave.
2. Store the token at `~/.cloudvolume/secrets/cave-secret.json` **or** set `CAVE_TOKEN` / `CAVE_TOKEN_FILE`.

## Usage

List available commands:

```bash
connectomics --help
```

Generate a synapse summary:

```bash
connectomics synapse-report \
  --dataset flywire_fafb_production \
  --pre-root-ids 720575940614097912 720575940619443091 \
  --post-root-ids 720575940616053300 \
  --output-dir out/example
```

By default the command queries materialization `783`, the public FlyWire snapshot used by Schlegel et al. Override the snapshot or use live-time queries via `--materialization` or `--at-timestamp`.

Outputs include:

- `synapse_summary.csv` and `synapse_summary.parquet`
- `synapse_counts.png`
- `synapse_graph.svg`
- `run_summary.json`

## Development

Run static checks and tests:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

Unit tests rely on stub clients and golden CSV fixtures to remain deterministic.

=======
# Connectomics

## CI Upgrades
Effective connectivity augments structural adjacency with multi-hop weighting so the CI stack can focus on information flow instead of raw synapse counts (Fig. 1–2).
Signed excitatory-inhibitory blocks preserve direction-specific balance, allowing downstream solvers to mix net and block-resolved modes when summarizing motifs (Fig. 1–2).
Receptive field clustering (RFC) links pathway structure to measured responses, turning meso-scale dynamics into interpretable feature banks for decoding (Fig. 1–2).
The non-linear dynamics model integrates divisive normalization and excitability control to replay realistic transient states before readout, mirroring the cascades illustrated in Fig. 1–2.
>>>>>>> 8f0f588 (Add CI scaffolding and configuration)
