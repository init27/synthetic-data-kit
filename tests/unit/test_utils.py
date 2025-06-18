"""Unit tests for utility functions."""

import pytest

from synthetic_data_kit.utils import text


@pytest.mark.unit
def test_chunk_text():
    """Test chunking of text."""
    text_content = "This is a sample text. " * 50
    chunks = text.chunk_text(text_content, chunk_size=100)
    
    # Check that all chunks are created correctly
    assert all(len(chunk) <= 100 for chunk in chunks)
    
    # Check that all text is preserved
    combined = " ".join(chunks)
    assert combined.replace(" ", "") == text_content.replace(" ", "")


@pytest.mark.unit
def test_truncate_text():
    """Test text truncation."""
    text_content = "This is a sample text that should be truncated."
    truncated = text.truncate_text(text_content, max_length=20)
    
    assert len(truncated) <= 20
    assert truncated.startswith("This is a sample")
    assert truncated.endswith("...")