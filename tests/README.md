# Synthetic Data Kit Tests

This directory contains tests for the Synthetic Data Kit.

## Test Structure

The tests are organized into three categories:

- **Unit Tests** (`unit/`): Test individual components in isolation
- **Integration Tests** (`integration/`): Test interactions between components
- **Functional Tests** (`functional/`): Test end-to-end workflows

## Running Tests

Run all tests:

```bash
pytest
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/unit

# Integration tests only
pytest tests/integration

# Functional tests only
pytest tests/functional
```

Run with coverage:

```bash
pytest --cov=synthetic_data_kit
```

## Test Data

The `data/` directory contains sample files used for testing:

- `sample.txt`: Sample text file for testing parsers
- `sample_qa_pairs.json`: Sample QA pairs for testing generation

## Writing New Tests

When adding new features or fixing bugs, please add appropriate tests. Follow these guidelines:

1. Place tests in the appropriate category directory
2. Use descriptive test names (`test_feature_name.py`)
3. Follow the existing test patterns
4. Use fixtures from `conftest.py` when possible
5. Add test markers (`@pytest.mark.unit`, etc.)
6. Mock external dependencies and API calls