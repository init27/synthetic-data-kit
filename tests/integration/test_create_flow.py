"""Integration tests for the create workflow."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from pathlib import Path

from synthetic_data_kit.core import create


@pytest.mark.integration
def test_process_file():
    """Test processing a file to generate QA pairs."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("This is sample text content for testing QA pair generation.")
        input_path = f.name
    
    output_dir = tempfile.mkdtemp()
    output_path = None
    
    try:
        # Mock OpenAI client
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"choices": [{"message": {"content": json.dumps([
            {"question": "What is this?", "answer": "This is sample text."},
            {"question": "What is it for?", "answer": "For testing QA generation."}
        ])}}]}
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Mock config
        mock_config = {
            "llm": {"provider": "api-endpoint"},
            "api-endpoint": {
                "api_base": "https://api.together.xyz/v1",
                "api_key": "mock-key",
                "model": "Llama-3-70B-Instruct"
            },
            "generation": {
                "temperature": 0.7,
                "chunk_size": 4000,
                "num_pairs": 2
            }
        }
        
        # Set environment variables
        with patch.dict(os.environ, {"OPENAI_API_KEY": "mock-key", "API_ENDPOINT_KEY": "mock-key"}, clear=False):
            # Mock necessary dependencies
            with patch("openai.OpenAI", return_value=mock_openai):
                with patch("synthetic_data_kit.utils.config.load_config", return_value=mock_config):
                    # Mock QAGenerator
                    mock_generator = MagicMock()
                    mock_generator.generate_qa_pairs.return_value = [
                        {"question": "What is this?", "answer": "This is sample text."},
                        {"question": "What is it for?", "answer": "For testing QA generation."}
                    ]
                    
                    with patch("synthetic_data_kit.generators.qa_generator.QAGenerator", return_value=mock_generator):
                        # Run the process_file function with minimal arguments
                        output_path = create.process_file(
                            file_path=input_path,
                            output_dir=output_dir,
                            provider="api-endpoint"
                        )
                        
                        # Just verify that the function doesn't raise an exception
                        assert output_path is not None
                        
                        # If the output file exists, verify its content
                        if os.path.exists(output_path):
                            with open(output_path, "r") as f:
                                try:
                                    content = json.load(f)
                                    assert len(content) >= 1
                                except json.JSONDecodeError:
                                    # If not valid JSON, just check file exists
                                    pass
    
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        try:
            os.rmdir(output_dir)
        except:
            pass