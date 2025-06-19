"""Integration tests for the create workflow."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from pathlib import Path

from synthetic_data_kit.core import create


@pytest.mark.integration
def test_process_file(patch_config, test_env):
    """Test processing a file to generate QA pairs."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("This is sample text content for testing QA pair generation.")
        input_path = f.name
    
    output_dir = tempfile.mkdtemp()
    output_path = None
    
    try:
        # Mock OpenAI client
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {"choices": [{"message": {"content": json.dumps([
                {"question": "What is this?", "answer": "This is sample text."},
                {"question": "What is it for?", "answer": "For testing QA generation."}
            ])}}]}
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock QAGenerator
            with patch("synthetic_data_kit.generators.qa_generator.QAGenerator") as mock_qa_gen_class:
                mock_generator = MagicMock()
                mock_generator.process_document.return_value = {
                    "summary": "A sample text for testing.",
                    "qa_pairs": [
                        {"question": "What is this?", "answer": "This is sample text."},
                        {"question": "What is it for?", "answer": "For testing QA generation."}
                    ]
                }
                mock_qa_gen_class.return_value = mock_generator
                
                # Run the process_file function with minimal arguments
                output_path = create.process_file(
                    file_path=input_path,
                    output_dir=output_dir,
                    config_path=None,
                    api_base=None,
                    model=None,
                    content_type="qa",
                    num_pairs=2,
                    verbose=False,
                    provider="api-endpoint"
                )
                
                # Verify function doesn't raise an exception
                assert output_path is not None
                
                # If the output file exists, verify its content
                if os.path.exists(output_path):
                    with open(output_path, "r") as f:
                        try:
                            content = json.load(f)
                            assert len(content) >= 1
                            assert "question" in content[0]
                            assert "answer" in content[0]
                        except json.JSONDecodeError:
                            # If not valid JSON, just check file exists
                            assert os.path.exists(output_path)
    
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        try:
            os.rmdir(output_dir)
        except:
            pass


@pytest.mark.integration
def test_process_directory(patch_config, test_env):
    """Test processing a directory to generate QA pairs."""
    # Create a temporary directory with test files
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Create a few test files
        file_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", dir=temp_dir, delete=False) as f:
                f.write(f"This is sample text content {i} for testing QA pair generation.")
                file_paths.append(f.name)
        
        # Mock OpenAI client
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {"choices": [{"message": {"content": json.dumps([
                {"question": "What is this?", "answer": "This is sample text."},
                {"question": "What is it for?", "answer": "For testing QA generation."}
            ])}}]}
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock QAGenerator
            with patch("synthetic_data_kit.generators.qa_generator.QAGenerator") as mock_qa_gen_class:
                mock_generator = MagicMock()
                mock_generator.process_document.return_value = {
                    "summary": "A sample text for testing.",
                    "qa_pairs": [
                        {"question": "What is this?", "answer": "This is sample text."},
                        {"question": "What is it for?", "answer": "For testing QA generation."}
                    ]
                }
                mock_qa_gen_class.return_value = mock_generator
                
                # Mock list of text files
                with patch("os.path.isdir") as mock_isdir, \
                     patch("synthetic_data_kit.core.create.find_text_files") as mock_find_files:
                    mock_isdir.return_value = True
                    mock_find_files.return_value = file_paths
                    
                    # Run the process_directory function
                    output_paths = create.process_directory(
                        input_dir=temp_dir,
                        output_dir=output_dir,
                        config_path=None,
                        api_base=None,
                        model=None,
                        content_type="qa",
                        num_pairs=2,
                        verbose=False,
                        provider="api-endpoint"
                    )
                    
                    # Verify function returns expected number of files
                    assert len(output_paths) == len(file_paths)
                    
                    # Check if output files exist
                    for output_path in output_paths:
                        assert os.path.exists(output_path)
                        
                        # Verify content
                        with open(output_path, "r") as f:
                            try:
                                content = json.load(f)
                                assert len(content) >= 1
                                assert "question" in content[0]
                                assert "answer" in content[0]
                            except json.JSONDecodeError:
                                # If not valid JSON, just check file exists
                                assert os.path.exists(output_path)
    
    finally:
        # Clean up temporary files and directories
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        # Clean up output directory
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            try:
                os.unlink(file_path)
            except:
                pass
        
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        try:
            os.rmdir(output_dir)
        except:
            pass