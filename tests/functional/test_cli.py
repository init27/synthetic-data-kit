"""Functional tests for the CLI interface."""

import os
import tempfile
import requests
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
def test_system_check_command():
    """Test the system-check command with vLLM provider."""
    runner = CliRunner()
    
    # Mock the requests.get to simulate a vLLM server response
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["Llama-3-70B-Instruct"]
        mock_get.return_value = mock_response
        
        # Mock the config to use vLLM
        with patch("synthetic_data_kit.utils.config.get_llm_provider") as mock_provider:
            mock_provider.return_value = "vllm"
            
            with patch("synthetic_data_kit.utils.config.get_vllm_config") as mock_vllm_config:
                mock_vllm_config.return_value = {
                    "api_base": "http://localhost:8000/v1",
                    "model": "Llama-3-70B-Instruct"
                }
                
                result = runner.invoke(app, ["system-check"])
                
                assert result.exit_code == 0
                assert "vLLM server is running" in result.stdout
                mock_get.assert_called_once_with("http://localhost:8000/v1/models", timeout=2)


@pytest.mark.functional
def test_ingest_command():
    """Test the ingest command with a text file."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
        f.write("Sample text content for testing.")
        input_path = f.name
    
    try:
        # Create a mock for process_file
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            # Set up the mock to return a valid output path
            output_path = os.path.join(os.path.dirname(input_path), "output_test.txt")
            mock_process.return_value = output_path
            
            # Run the ingest command
            result = runner.invoke(app, ["ingest", input_path])
            
            # Verify the command executed successfully
            assert result.exit_code == 0
            assert "Text successfully extracted" in result.stdout
            
            # Verify the process_file function was called with correct arguments
            mock_process.assert_called_once()
            # Check that the first argument (file_path) matches
            assert mock_process.call_args[0][0] == input_path
    
    finally:
        # Clean up the temporary file
        if os.path.exists(input_path):
            os.unlink(input_path)