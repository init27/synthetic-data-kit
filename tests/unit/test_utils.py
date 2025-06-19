"""Unit tests for utility functions."""

import json
import pytest
from pathlib import Path

from synthetic_data_kit.utils import text, config


@pytest.mark.unit
def test_split_into_chunks():
    """Test splitting text into chunks."""
    # Create multi-paragraph text
    paragraphs = ["Paragraph one." * 5, "Paragraph two." * 5, "Paragraph three." * 5]
    text_content = "\n\n".join(paragraphs)
    
    # Using a small chunk size to ensure splitting
    chunks = text.split_into_chunks(text_content, chunk_size=50, overlap=10)
    
    # Check that chunks were created
    assert len(chunks) > 0
    
    # For this specific test case, we should have at least 2 chunks
    assert len(chunks) >= 2
    
    # Check that the total content is preserved (allow for some difference due to overlap)
    combined_length = sum(len(chunk) for chunk in chunks)
    # The combined length should be at least the original length
    assert combined_length >= len(text_content)


@pytest.mark.unit
def test_extract_json_from_text():
    """Test extracting JSON from text."""
    json_text = """
    Some random text before the JSON
    ```json
    {
        "question": "What is synthetic data?",
        "answer": "Synthetic data is artificially generated data."
    }
    ```
    Some random text after the JSON
    """
    
    result = text.extract_json_from_text(json_text)
    
    assert isinstance(result, dict)
    assert "question" in result
    assert result["question"] == "What is synthetic data?"
    assert result["answer"] == "Synthetic data is artificially generated data."


@pytest.mark.unit
def test_extract_json_list_from_text():
    """Test extracting JSON list from text."""
    json_text = """
    Some random text before the JSON
    ```json
    [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data."
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples."
        }
    ]
    ```
    Some random text after the JSON
    """
    
    result = text.extract_json_from_text(json_text)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["question"] == "What is synthetic data?"
    assert result[1]["question"] == "Why use synthetic data?"


@pytest.mark.unit
def test_load_config(tmpdir):
    """Test loading config from file."""
    # Create a temporary config file
    config_data = {
        "llm": {"provider": "test-provider"},
        "test-provider": {
            "api_base": "http://test-api.com",
            "model": "test-model"
        }
    }
    
    config_path = Path(tmpdir) / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write("llm:\n  provider: test-provider\ntest-provider:\n  api_base: http://test-api.com\n  model: test-model")
    
    # Load the config
    loaded_config = config.load_config(config_path)
    
    # Check that the config was loaded correctly
    assert loaded_config["llm"]["provider"] == "test-provider"
    assert loaded_config["test-provider"]["api_base"] == "http://test-api.com"
    assert loaded_config["test-provider"]["model"] == "test-model"


@pytest.mark.unit
def test_get_llm_provider(mock_config):
    """Test getting the LLM provider from config."""
    provider = config.get_llm_provider(mock_config)
    assert provider == "api-endpoint"
    
    # Test with empty config
    empty_config = {}
    default_provider = config.get_llm_provider(empty_config)
    assert default_provider == "vllm"  # Should return the default provider


@pytest.mark.unit
def test_get_path_config(mock_config):
    """Test getting path configuration."""
    output_path = config.get_path_config(mock_config, "output", "default")
    assert output_path == "data/output"
    
    # Test with missing path in config
    missing_path = config.get_path_config(mock_config, "nonexistent", "default")
    assert missing_path == "data/nonexistent"
    
    # Test with empty config
    empty_config = {}
    default_path = config.get_path_config(empty_config, "output", "default")
    assert default_path == "data/output"