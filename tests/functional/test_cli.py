"""Functional tests for the CLI interface."""

import os
import tempfile
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
def test_system_check_command():
    """Test the system-check command."""
    runner = CliRunner()
    
    with patch("synthetic_data_kit.core.context.get_llm_client") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_client.system_check.return_value = True
        
        result = runner.invoke(app, ["system-check"])
        
        assert result.exit_code == 0
        assert "System check" in result.stdout
        mock_client.system_check.assert_called_once()


@pytest.mark.functional
def test_ingest_command():
    """Test the ingest command with a text file."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
        f.write("Sample text content for testing.")
        input_path = f.name
    
    try:
        with patch("synthetic_data_kit.core.ingest.ingest_file") as mock_ingest:
            # Set up the mock to return a valid output path
            output_path = os.path.join(os.path.dirname(input_path), "output_test.txt")
            mock_ingest.return_value = output_path
            
            # Run the ingest command
            result = runner.invoke(app, ["ingest", input_path])
            
            # Verify the command executed successfully
            assert result.exit_code == 0
            assert "Successfully ingested" in result.stdout
            
            # Verify the ingest function was called correctly
            mock_ingest.assert_called_once_with(
                file_path=input_path,
                verbose=False
            )
    
    finally:
        # Clean up the temporary file
        if os.path.exists(input_path):
            os.unlink(input_path)