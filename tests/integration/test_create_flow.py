"""Integration tests for the create workflow."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from synthetic_data_kit.core import create


@pytest.mark.integration
def test_create_qa_pairs(mock_llm_client):
    """Test creation of QA pairs from text."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("This is sample text content for testing QA pair generation.")
        input_path = f.name
    
    output_path = None
    
    try:
        with patch("synthetic_data_kit.models.llm_client.LLMClient", return_value=mock_llm_client):
            # Run the create function
            output_path = create.create_from_file(
                file_path=input_path,
                content_type="qa",
                num_pairs=2,
                verbose=True
            )
            
            # Verify output exists
            assert os.path.exists(output_path)
            
            # Verify content
            with open(output_path, "r") as f:
                content = json.load(f)
                assert len(content) == 2
                assert "question" in content[0]
                assert "answer" in content[0]
    
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)