"""Unit tests for LLM client."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from synthetic_data_kit.models.llm_client import LLMClient


@pytest.mark.unit
def test_llm_client_initialization(patch_config, test_env):
    """Test LLM client initialization with API endpoint provider."""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Initialize client
        client = LLMClient(provider="api-endpoint")
        
        # Check that the client was initialized correctly
        assert client.provider == "api-endpoint"
        assert client.api_base is not None
        assert client.model is not None
        # Check that OpenAI client was initialized
        assert mock_openai.called


@pytest.mark.unit
def test_llm_client_vllm_initialization(patch_config, test_env):
    """Test LLM client initialization with vLLM provider."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["mock-model"]
        mock_get.return_value = mock_response
        
        # Initialize client
        client = LLMClient(provider="vllm")
        
        # Check that the client was initialized correctly
        assert client.provider == "vllm"
        assert client.api_base is not None
        assert client.model is not None
        # Check that vLLM server was checked
        assert mock_get.called


@pytest.mark.unit
def test_llm_client_chat_completion(patch_config, test_env):
    """Test LLM client chat completion with API endpoint provider."""
    with patch("openai.OpenAI") as mock_openai:
        # Mock OpenAI client response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Initialize client
        client = LLMClient(provider="api-endpoint")
        
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"}
        ]
        
        response = client.chat_completion(messages, temperature=0.7)
        
        # Check that the response is correct
        assert response == "This is a test response"
        # Check that OpenAI client was called
        assert mock_client.chat.completions.create.called


@pytest.mark.unit
def test_llm_client_vllm_chat_completion(patch_config, test_env):
    """Test LLM client chat completion with vLLM provider."""
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        # Mock vLLM server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = ["mock-model"]
        mock_get.return_value = mock_check_response
        
        # Mock vLLM API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Initialize client
        client = LLMClient(provider="vllm")
        
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"}
        ]
        
        response = client.chat_completion(messages, temperature=0.7)
        
        # Check that the response is correct
        assert response == "This is a test response"
        # Check that vLLM API was called
        assert mock_post.called