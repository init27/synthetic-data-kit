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
            "answer": "Synthetic data is artificially generated data that mimics real data.",
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns.",
        },
    ]


@pytest.fixture
def sample_qa_pairs_file():
    """Create a temporary file with sample QA pairs for testing."""
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data.",
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns.",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(qa_pairs, f)
        file_path = f.name

    yield file_path

    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


# Mock factories for reusable test components

class MockLLMClientFactory:
    """Factory for creating mock LLM clients with different configurations."""

    @staticmethod
    def create_qa_client(qa_pairs=None):
        """Create a mock client for QA generation."""
        if qa_pairs is None:
            qa_pairs = [
                {
                    "question": "What is synthetic data?",
                    "answer": "Synthetic data is artificially generated data that mimics real data.",
                },
                {
                    "question": "Why use synthetic data for fine-tuning?",
                    "answer": "Synthetic data can help overcome data scarcity and privacy concerns.",
                },
            ]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps(qa_pairs)
        mock_client.batch_completion.return_value = [
            json.dumps([pair]) for pair in qa_pairs
        ]
        return mock_client

    @staticmethod
    def create_cot_client(cot_examples=None):
        """Create a mock client for Chain of Thought generation."""
        if cot_examples is None:
            cot_examples = [
                {
                    "reasoning": "Let me think step by step...",
                    "answer": "Based on my analysis, the answer is..."
                }
            ]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps(cot_examples)
        mock_client.batch_completion.return_value = [
            json.dumps([example]) for example in cot_examples
        ]
        return mock_client

    @staticmethod
    def create_summary_client(summary_text="This is a test summary."):
        """Create a mock client for summary generation."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = summary_text
        return mock_client

    @staticmethod
    def create_rating_client(ratings=None):
        """Create a mock client for content rating."""
        if ratings is None:
            ratings = [8, 7, 9]  # Default ratings

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps(ratings)
        mock_client.batch_completion.return_value = [
            json.dumps([rating]) for rating in ratings
        ]
        return mock_client


@pytest.fixture
def llm_client_factory():
    """Factory fixture for creating various mock LLM clients."""
    return MockLLMClientFactory


@pytest.fixture
def mock_llm_client(llm_client_factory):
    """Default mock LLM client for backward compatibility."""
    return llm_client_factory.create_qa_client()


class MockConfigFactory:
    """Factory for creating various mock configurations."""

    @staticmethod
    def create_api_config(provider="api-endpoint", api_key="mock-key", model="mock-model"):
        """Create a mock API endpoint configuration."""
        return {
            "llm": {"provider": provider},
            "api-endpoint": {
                "api_base": "https://api.together.xyz/v1",
                "api_key": api_key,
                "model": model,
                "max_retries": 3,
                "retry_delay": 1,
            },
            "generation": {
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.95,
                "batch_size": 32,
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
            },
        }

    @staticmethod
    def create_vllm_config(model="mock-vllm-model"):
        """Create a mock vLLM configuration."""
        return {
            "llm": {"provider": "vllm"},
            "vllm": {
                "api_base": "http://localhost:8000",
                "model": model,
                "max_retries": 3,
                "retry_delay": 1,
            },
            "generation": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 0.95,
                "batch_size": 16,
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
            },
        }


@pytest.fixture
def config_factory():
    """Factory fixture for creating various mock configurations."""
    return MockConfigFactory


@pytest.fixture
def mock_config(config_factory):
    """Default mock configuration for backward compatibility."""
    return config_factory.create_api_config()


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
def patch_config(config_factory):
    """Patch the config loader to return a mock configuration."""
    with patch("synthetic_data_kit.utils.config.load_config") as mock_load_config:
        mock_load_config.return_value = config_factory.create_api_config()
        yield mock_load_config


@pytest.fixture
def patch_vllm_config(config_factory):
    """Patch the config loader to return a vLLM configuration."""
    with patch("synthetic_data_kit.utils.config.load_config") as mock_load_config:
        mock_load_config.return_value = config_factory.create_vllm_config()
        yield mock_load_config


# Additional utility fixtures for common test patterns

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_file_operations():
    """Mock common file operations for testing."""
    with patch('builtins.open'), \
         patch('os.makedirs'), \
         patch('os.path.exists', return_value=True), \
         patch('pathlib.Path.exists', return_value=True):
        yield


@pytest.fixture
def sample_cot_data():
    """Sample Chain of Thought data for testing."""
    return [
        {
            "query": "What is 2 + 2?",
            "reasoning": "Let me solve this step by step. First, I need to add 2 and 2. This is a basic arithmetic operation.",
            "answer": "2 + 2 = 4"
        },
        {
            "query": "Explain photosynthesis",
            "reasoning": "To explain photosynthesis, I need to break it down into its key components and process.",
            "answer": "Photosynthesis is the process by which plants convert sunlight into energy."
        }
    ]


@pytest.fixture
def sample_conversations():
    """Sample conversation data for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "How do I bake a cake?"},
                {"role": "assistant", "content": "To bake a cake, you need flour, eggs, sugar, and butter."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to current weather data."}
            ]
        }
    ]
