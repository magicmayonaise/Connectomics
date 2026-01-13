# Contributing to Connectomics CLI

Thank you for your interest in contributing to Connectomics CLI! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Prerequisites

- Python 3.11 or newer
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Connectomics.git
   cd Connectomics
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-motif-analysis` - New features
- `fix/synapse-counting-bug` - Bug fixes
- `docs/improve-readme` - Documentation
- `refactor/simplify-config` - Code refactoring

### Making Changes

1. **Write your code** following the project's style guidelines
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run the test suite** before committing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific tests
pytest tests/test_constants.py -v

# Run tests matching a pattern
pytest -k "test_effective"
```

### Code Quality Checks

```bash
# Lint your code
ruff check src/ tests/

# Format your code
ruff format src/ tests/

# Type checking
mypy src/
```

### Committing Changes

Write clear, concise commit messages:

```bash
git add .
git commit -m "Add functional role annotation for ExR neurons"
```

Follow these conventions:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Fix bug" not "Fixes bug")
- Keep the first line under 72 characters
- Reference issues when applicable ("Fix #123: Handle empty synapse tables")

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Related issues
   - Testing done
   - Screenshots (if applicable)

4. **Address review feedback** promptly

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use double quotes for strings

```python
def compute_connectivity(
    root_ids: list[int],
    materialization: int = 783,
) -> pd.DataFrame:
    """Compute connectivity for the given root IDs.

    Parameters
    ----------
    root_ids
        List of neuron root IDs to analyze.
    materialization
        CAVE materialization version.

    Returns
    -------
    pd.DataFrame
        Connectivity table with synapse counts.
    """
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Short description of the function.

    Longer description if needed, explaining the function's
    behavior in more detail.

    Parameters
    ----------
    param1
        Description of param1.
    param2
        Description of param2.

    Returns
    -------
    bool
        Description of the return value.

    Raises
    ------
    ValueError
        When param1 is negative.

    Examples
    --------
    >>> function_name(1, "test")
    True
    """
```

### Import Order

Imports should be sorted (ruff handles this automatically):

```python
# Standard library
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import pandas as pd

# Local
from cx_connectome.constants import DEFAULT_MATERIALIZATION
from cx_connectome.adjacency import build_connectivity
```

## Testing Guidelines

### Test Organization

- Place tests in the `tests/` directory
- Name test files `test_<module_name>.py`
- Name test functions `test_<what_is_being_tested>`

### Test Structure

```python
class TestClassName:
    """Test class description."""

    def test_method_does_expected_thing(self) -> None:
        """Test that method produces expected output."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_value
```

### Using Fixtures

Define reusable test data in `conftest.py`:

```python
@pytest.fixture
def sample_synapse_data() -> pd.DataFrame:
    """Create sample synapse data for testing."""
    return pd.DataFrame({
        "pre_pt_root_id": [1001, 1001, 1002],
        "post_pt_root_id": [2001, 2002, 2001],
        "synapse_count": [5, 3, 2],
    })
```

## Documentation Guidelines

### README Updates

Update the README when:
- Adding new features
- Changing CLI commands
- Modifying installation steps
- Adding new modules

### Module Documentation

Each module should have a docstring explaining:
- Purpose of the module
- Key functions/classes
- Usage examples

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions considered

## Questions?

Feel free to:
- Open a GitHub issue
- Start a discussion
- Reach out to maintainers

Thank you for contributing!
