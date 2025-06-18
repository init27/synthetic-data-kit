"""Unit tests for utility functions."""

import pytest

from synthetic_data_kit.utils import text


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