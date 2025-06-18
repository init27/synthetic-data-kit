"""Common pytest fixtures for synthetic-data-kit."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_data_path():
    """Fixture providing path to the sample data directory."""
    base_dir = Path(__file__).parent
    return str(base_dir / "data")


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_client = MagicMock()
    mock_client.generate.return_value = json.dumps([
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data."
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns."
        }
    ])
    
    return mock_client


@pytest.fixture
def test_env():
    """Set test environment variables."""
    original_env = os.environ.copy()
    os.environ["PROJECT_TEST_ENV"] = "1"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)