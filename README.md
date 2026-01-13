# Connectomics CLI

[![CI](https://github.com/magicmayonaise/Connectomics/actions/workflows/ci.yml/badge.svg)](https://github.com/magicmayonaise/Connectomics/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible workflow toolkit for querying the [CAVE](https://global.daf-apis.com/cave/doc) materialization service and analyzing synaptic connectivity from public datasets such as [FlyWire](https://codex.flywire.ai/).

## Features

- **OAuth2-aware authentication** with actionable guidance when no token is present
- **Materialization-aware querying** that respects CAVE row caps via intelligent chunking
- **Automatic table discovery** for synapse and cell type annotation schemas
- **Streaming aggregation** producing tidy pandas summaries and NetworkX graphs
- **Deterministic outputs** (CSV, Parquet, SVG, PNG) with JSON provenance tracking
- **Central Complex (CX) analysis** tools for Drosophila neural circuit research
- **Connectivity Interpreter (CI) stack** for effective connectivity and dynamics simulation

## Installation

```bash
# Clone the repository
git clone https://github.com/magicmayonaise/Connectomics.git
cd Connectomics

# Install in development mode
pip install -e ".[dev]"
```

**Requirements:** Python 3.11 or newer. A virtual environment is recommended.

### Optional Dependencies

```bash
# For dynamics simulation (requires PyTorch)
pip install -e ".[dyn]"

# For documentation building
pip install -e ".[docs]"

# Install everything
pip install -e ".[all]"
```

## Authentication

The CLI uses the same authentication conventions as `caveclient`. If no token is discovered, clear error messages explain how to obtain one:

1. Obtain credentials from https://global.daf-apis.com/cave
2. Store the token at `~/.cloudvolume/secrets/cave-secret.json`
   **OR** set the `CAVE_TOKEN` / `CAVE_TOKEN_FILE` environment variable

## Quick Start

### CLI Usage

List available commands:

```bash
connectomics --help
```

Generate a synapse summary report:

```bash
connectomics synapse-report \
  --dataset flywire_fafb_production \
  --pre-root-ids 720575940614097912 720575940619443091 \
  --post-root-ids 720575940616053300 \
  --output-dir out/example
```

By default, queries use materialization `783` (public FlyWire snapshot). Override with `--materialization` or use live queries via `--at-timestamp`.

**Outputs include:**
- `synapse_summary.csv` and `synapse_summary.parquet` - Aggregated connectivity data
- `synapse_counts.png` - Visualization of synapse counts
- `synapse_graph.svg` - Network graph visualization
- `run_summary.json` - Provenance and parameter tracking

### Python API Usage

```python
from cx_connectome import (
    build_connectivity_graph,
    DEFAULT_DATASET,
    DEFAULT_MATERIALIZATION,
)

# Build a multi-hop connectivity graph
graph = build_connectivity_graph(
    root_ids=[720575940614097912],
    dataset=DEFAULT_DATASET,
    materialization=DEFAULT_MATERIALIZATION,
)

# Access graph properties
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

```python
from cx_connectome.ci.effective import compute_effective_connectivity
from scipy.sparse import csr_matrix
import numpy as np

# Compute effective connectivity for multi-hop analysis
adjacency = csr_matrix(np.random.rand(100, 100))
results = compute_effective_connectivity(
    adjacency,
    k_hops=[1, 2, 3],
    normalize="post_total",
)
```

## Project Structure

```
Connectomics/
├── src/
│   ├── connectomics_cli/     # CLI entry point and synapse analysis
│   │   ├── cli.py            # Typer CLI commands
│   │   ├── config.py         # Configuration and QueryContext
│   │   ├── auth.py           # CAVE OAuth2 authentication
│   │   └── synapse_analysis.py
│   └── cx_connectome/        # Core connectomics library
│       ├── constants.py      # Centralized configuration
│       ├── adjacency.py      # Connectivity table building
│       ├── annotations.py    # CAVE annotation utilities
│       ├── cx_network.py     # N1→N2→N3 graph construction
│       ├── functional_roles.py # Neuron role annotation
│       ├── motifs.py         # Network motif discovery
│       ├── topology.py       # Connectivity metrics
│       ├── ci/               # Connectivity Interpreter stack
│       │   ├── effective.py  # Effective connectivity
│       │   ├── dynamics.py   # Neural dynamics simulation
│       │   ├── metrics.py    # EI-ratio, lateral bias
│       │   └── signed.py     # Excitatory/inhibitory blocks
│       └── legacy/           # Backward-compatible modules
├── tests/                    # Test suite
├── configs/                  # YAML configuration files
└── .github/workflows/        # CI/CD pipelines
```

## Core Modules

### cx_connectome

| Module | Description |
|--------|-------------|
| `adjacency` | Build weighted edge lists from CAVE synaptic data |
| `annotations` | Fetch hierarchical cell-type annotations |
| `cx_network` | Construct multi-hop connectivity graphs (N1→N2→N3) |
| `functional_roles` | Annotate neurons with functional roles (Navigation, Sleep-Promoting) |
| `motifs` | Discover canonical network motifs (feedback, lateral, recurrent) |
| `topology` | Compute overlap, fan-in/fan-out metrics |

### cx_connectome.ci (Connectivity Interpreter)

| Module | Description |
|--------|-------------|
| `effective` | Multi-hop effective connectivity via matrix powers |
| `dynamics` | Non-linear circuit simulation with divisive normalization |
| `metrics` | EI-ratio, lateral bias, robustness curves |
| `signed` | Excitatory/inhibitory block decomposition |
| `rfc` | Receptive field clustering |

## Configuration

Default parameters are centralized in `cx_connectome.constants`:

```python
from cx_connectome.constants import (
    DEFAULT_DATASET,           # "flywire_fafb_production"
    DEFAULT_MATERIALIZATION,   # 783
    DEFAULT_CHUNK_SIZE,        # 5000
    DEFAULT_SYNAPSE_TABLE,     # "synapses_nt_v1"
    DEFAULT_MIN_SYNAPSES,      # 10
)
```

Pipeline configuration can also be specified via YAML:

```yaml
# configs/ci.yaml
materialization: 783
k_hops: [1, 2, 3, 4, 5]
normalize: post_total
signed_mode: net
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_constants.py -v
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Lint**: Ruff linting and formatting checks
- **Type Check**: MyPy strict type checking
- **Test**: pytest on Python 3.11 and 3.12
- **Build**: Package building and validation
- **Security**: Bandit security scanning

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest && ruff check .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{connectomics_cli,
  title = {Connectomics CLI: Reproducible CAVE Connectomics Analysis},
  url = {https://github.com/magicmayonaise/Connectomics},
  year = {2024}
}
```

## Acknowledgments

- [FlyWire](https://codex.flywire.ai/) for the public connectome dataset
- [CAVE](https://github.com/seung-lab/CAVEclient) for the materialization API
- The neuroscience community for their contributions to open connectomics data
