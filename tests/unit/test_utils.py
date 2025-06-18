"""Unit tests for utility functions."""

import pytest

from synthetic_data_kit.utils import text


@pytest.mark.unit
def test_split_into_chunks():
    """Test splitting text into chunks."""
    text_content = "This is a sample text. " * 50
    chunks = text.split_into_chunks(text_content, chunk_size=100)
    
    # Check that all chunks are created correctly
    assert all(len(chunk) <= 100 for chunk in chunks)
    
    # Check that we have the expected number of chunks
    expected_chunks = (len(text_content) + 99) // 100
    assert len(chunks) >= expected_chunks - 1  # Allow for some flexibility due to overlap


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