"""Integration tests for the create workflow."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from pathlib import Path

from synthetic_data_kit.core import create


@pytest.mark.integration
def test_process_file(mock_llm_client):
    """Test processing a file to generate QA pairs."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("This is sample text content for testing QA pair generation.")
        input_path = f.name
    
    output_dir = tempfile.mkdtemp()
    output_path = None
    
    try:
        with patch("synthetic_data_kit.models.llm_client.LLMClient", return_value=mock_llm_client):
            # Mock QAGenerator
            mock_generator = MagicMock()
            mock_generator.generate_qa_pairs.return_value = [
                {"question": "What is this?", "answer": "This is sample text."},
                {"question": "What is it for?", "answer": "For testing QA generation."}
            ]
            
            with patch("synthetic_data_kit.generators.qa_generator.QAGenerator", return_value=mock_generator):
                # Run the process_file function
                output_path = create.process_file(
                    file_path=input_path,
                    output_dir=output_dir,
                    config_path=None,
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
            os.remove(output_path)
        os.rmdir(output_dir)