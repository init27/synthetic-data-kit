"""Common pytest fixtures for synthetic-data-kit."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_data_path():
    """Fixture providing path to the sample data directory."""
    base_dir = Path(__file__).parent
    return str(base_dir / "data")


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        f.write("This is sample text content for testing Synthetic Data Kit.")
        file_path = f.name
        
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def sample_qa_pairs():
    """Return sample QA pairs for testing."""
    return [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data."
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns."
        }
    ]


@pytest.fixture
def sample_qa_pairs_file():
    """Create a temporary file with sample QA pairs for testing."""
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data."
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns."
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(qa_pairs, f)
        file_path = f.name
        
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = json.dumps([
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data."
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns."
        }
    ])
    
    mock_client.batch_completion.return_value = [
        json.dumps([
            {
                "question": "What is synthetic data?",
                "answer": "Synthetic data is artificially generated data that mimics real data."
            }
        ]),
        json.dumps([
            {
                "question": "Why use synthetic data for fine-tuning?",
                "answer": "Synthetic data can help overcome data scarcity and privacy concerns."
            }
        ])
    ]
    
    return mock_client


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "llm": {"provider": "api-endpoint"},
        "api-endpoint": {
            "api_base": "https://api.together.xyz/v1",
            "api_key": "mock-key",
            "model": "Llama-3-70B-Instruct",
            "max_retries": 3,
            "retry_delay": 1
        },
        "vllm": {
            "api_base": "http://localhost:8000/v1",
            "model": "Llama-3-70B-Instruct",
            "max_retries": 3,
            "retry_delay": 1
        },
        "generation": {
            "temperature": 0.7,
            "chunk_size": 4000,
            "overlap": 200,
            "num_pairs": 10,
            "batch_size": 8
        },
        "curate": {
            "threshold": 7.0,
            "batch_size": 8,
            "temperature": 0.1
        },
        "format": {
            "default": "jsonl"
        },
        "paths": {
            "output": "data/output",
            "generated": "data/generated",
            "cleaned": "data/cleaned",
            "final": "data/final"
        },
        "prompts": {
            "summary": "Summarize the following text concisely.",
            "qa_generation": "Generate {num_pairs} high-quality question-answer pairs based on the following text about: {summary}\n\nText:\n{text}",
            "qa_rating": "Rate each question-answer pair on a scale of 1-10 based on quality, accuracy, and relevance.\n\nPairs:\n{pairs}",
            "cot_generation": "Generate Chain of Thought reasoning examples from the following text about: {summary}\n\nText:\n{text}"
        }
    }


@pytest.fixture
def test_env():
    """Set test environment variables."""
    original_env = os.environ.copy()
    os.environ["PROJECT_TEST_ENV"] = "1"
    # Only use API_ENDPOINT_KEY for consistency with the code
    os.environ["API_ENDPOINT_KEY"] = "mock-api-key-for-testing"
    os.environ["SDK_VERBOSE"] = "false"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def patch_config():
    """Patch the config loader to return a mock configuration."""
    with patch("synthetic_data_kit.utils.config.load_config") as mock_load_config:
        mock_load_config.return_value = {
            "llm": {"provider": "api-endpoint"},
            "api-endpoint": {
                "api_base": "https://api.together.xyz/v1",
                "api_key": "mock-key",
                "model": "Llama-3-70B-Instruct",
                "max_retries": 3,
                "retry_delay": 1
            },
            "vllm": {
                "api_base": "http://localhost:8000/v1",
                "model": "Llama-3-70B-Instruct",
                "max_retries": 3,
                "retry_delay": 1
            },
            "generation": {
                "temperature": 0.7,
                "chunk_size": 4000,
                "overlap": 200,
                "num_pairs": 10,
                "batch_size": 8
            },
            "curate": {
                "threshold": 7.0,
                "batch_size": 8,
                "temperature": 0.1
            },
            "format": {
                "default": "jsonl"
            },
            "paths": {
                "output": "data/output",
                "generated": "data/generated",
                "cleaned": "data/cleaned",
                "final": "data/final"
            },
            "prompts": {
                "summary": "Summarize the following text concisely.",
                "qa_generation": "Generate {num_pairs} high-quality question-answer pairs based on the following text about: {summary}\n\nText:\n{text}",
                "qa_rating": "Rate each question-answer pair on a scale of 1-10 based on quality, accuracy, and relevance.\n\nPairs:\n{pairs}",
                "cot_generation": "Generate Chain of Thought reasoning examples from the following text about: {summary}\n\nText:\n{text}"
            }
        }
        yield mock_load_config