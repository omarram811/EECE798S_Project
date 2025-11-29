# tests/test_providers.py
"""
Unit tests for app/providers.py

Tests cover:
- API key validation functions (OpenAI and Gemini)
- Provider classes (OpenAIProvider, GeminiProvider)
- Embedding generation
- Chat completion
"""

import pytest
from unittest.mock import MagicMock, patch


class TestAPIKeyValidation:
    """Tests for API key validation functions."""

    def test_validate_api_key_with_provider_unknown_provider(self):
        """Test that unknown provider returns error."""
        from app.providers import validate_api_key_with_provider
        
        is_valid, error = validate_api_key_with_provider("unknown_provider", "some_key", "some_model")
        
        assert is_valid is False
        assert "Unknown provider" in error

    @patch('app.providers.OpenAI')
    def test_validate_openai_api_key_success(self, mock_openai):
        """Test successful OpenAI API key validation."""
        from app.providers import validate_openai_api_key
        
        # Setup mock
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        client_instance.embeddings.create.return_value = embedding_response
        
        is_valid, error = validate_openai_api_key("sk-test123456789", "text-embedding-3-small")
        
        assert is_valid is True
        assert error is None
        client_instance.embeddings.create.assert_called_once()

    @patch('app.providers.OpenAI')
    def test_validate_openai_api_key_invalid(self, mock_openai):
        """Test OpenAI API key validation with invalid key."""
        from app.providers import validate_openai_api_key
        
        # Setup mock to raise exception
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.embeddings.create.side_effect = Exception("invalid api key provided")
        
        is_valid, error = validate_openai_api_key("sk-invalid", "text-embedding-3-small")
        
        assert is_valid is False
        assert "Invalid API Key" in error

    @patch('app.providers.OpenAI')
    def test_validate_openai_api_key_rate_limit(self, mock_openai):
        """Test OpenAI API key validation with rate limit error."""
        from app.providers import validate_openai_api_key
        
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.embeddings.create.side_effect = Exception("429 rate limit exceeded")
        
        is_valid, error = validate_openai_api_key("sk-test123", "text-embedding-3-small")
        
        assert is_valid is False
        assert "Rate Limit" in error

    @patch('app.providers.OpenAI')
    def test_validate_openai_api_key_quota_exceeded(self, mock_openai):
        """Test OpenAI API key validation with quota exceeded error."""
        from app.providers import validate_openai_api_key
        
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.embeddings.create.side_effect = Exception("quota exceeded for this billing cycle")
        
        is_valid, error = validate_openai_api_key("sk-test123", "text-embedding-3-small")
        
        assert is_valid is False
        assert "Quota Exceeded" in error

    @patch('app.providers.genai')
    def test_validate_gemini_api_key_success(self, mock_genai):
        """Test successful Gemini API key validation."""
        from app.providers import validate_gemini_api_key
        
        mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}
        
        is_valid, error = validate_gemini_api_key("AIza_test_key_123", "models/text-embedding-004")
        
        assert is_valid is True
        assert error is None
        mock_genai.configure.assert_called_once()

    @patch('app.providers.genai')
    def test_validate_gemini_api_key_invalid(self, mock_genai):
        """Test Gemini API key validation with invalid key."""
        from app.providers import validate_gemini_api_key
        
        mock_genai.embed_content.side_effect = Exception("api key not valid")
        
        is_valid, error = validate_gemini_api_key("AIza_invalid", "models/text-embedding-004")
        
        assert is_valid is False
        assert "Invalid API Key" in error

    @patch('app.providers.genai')
    def test_validate_gemini_api_key_permission_denied(self, mock_genai):
        """Test Gemini API key validation with permission denied error."""
        from app.providers import validate_gemini_api_key
        
        mock_genai.embed_content.side_effect = Exception("403 permission denied")
        
        is_valid, error = validate_gemini_api_key("AIza_test", "models/text-embedding-004")
        
        assert is_valid is False
        assert "Authentication Error" in error


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    @patch('app.providers.OpenAI')
    def test_openai_provider_initialization(self, mock_openai):
        """Test OpenAI provider initialization."""
        from app.providers import OpenAIProvider
        
        provider = OpenAIProvider(
            model="gpt-4o-mini",
            embed_model="text-embedding-3-small",
            api_key="sk-test123"
        )
        
        assert provider.model == "gpt-4o-mini"
        assert provider.embed_model == "text-embedding-3-small"
        mock_openai.assert_called_once_with(api_key="sk-test123")

    @patch('app.providers.OpenAI')
    def test_openai_provider_embed(self, mock_openai):
        """Test OpenAI provider embedding generation."""
        from app.providers import OpenAIProvider
        
        # Setup mock
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        
        embedding_response = MagicMock()
        embedding_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6])
        ]
        client_instance.embeddings.create.return_value = embedding_response
        
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        embeddings = provider.embed(["text1", "text2"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @patch('app.providers.OpenAI')
    def test_openai_provider_complete(self, mock_openai):
        """Test OpenAI provider non-streaming completion."""
        from app.providers import OpenAIProvider
        
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        
        chat_response = MagicMock()
        chat_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        client_instance.chat.completions.create.return_value = chat_response
        
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        result = provider.complete([{"role": "user", "content": "Hello"}])
        
        assert result == "Test response"

    @patch('app.providers.OpenAI')
    def test_openai_provider_stream_chat(self, mock_openai):
        """Test OpenAI provider streaming chat."""
        from app.providers import OpenAIProvider
        
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        
        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" World"))]
        
        client_instance.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])
        
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        tokens = list(provider.stream_chat([{"role": "user", "content": "Hi"}]))
        
        assert tokens == ["Hello", " World"]


class TestGeminiProvider:
    """Tests for GeminiProvider class."""

    @patch('app.providers.genai')
    def test_gemini_provider_initialization(self, mock_genai):
        """Test Gemini provider initialization."""
        from app.providers import GeminiProvider
        
        provider = GeminiProvider(
            model="gemini-2.5-flash",
            embed_model="models/text-embedding-004",
            api_key="AIza_test"
        )
        
        assert provider.model == "gemini-2.5-flash"
        assert provider.embed_model == "models/text-embedding-004"
        mock_genai.configure.assert_called_once_with(api_key="AIza_test")

    @patch('app.providers.genai')
    def test_gemini_provider_embed(self, mock_genai):
        """Test Gemini provider embedding generation."""
        from app.providers import GeminiProvider
        
        mock_genai.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        provider = GeminiProvider(model="gemini-2.5-flash", api_key="AIza_test")
        embeddings = provider.embed(["test text"])
        
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

    @patch('app.providers.genai')
    def test_gemini_provider_complete(self, mock_genai):
        """Test Gemini provider non-streaming completion."""
        from app.providers import GeminiProvider
        
        model_instance = MagicMock()
        model_instance.generate_content.return_value = MagicMock(text="Gemini response")
        mock_genai.GenerativeModel.return_value = model_instance
        
        provider = GeminiProvider(model="gemini-2.5-flash", api_key="AIza_test")
        result = provider.complete([{"role": "user", "content": "Hello"}])
        
        assert result == "Gemini response"

    @patch('app.providers.genai')
    def test_gemini_convert_messages_to_prompt(self, mock_genai):
        """Test message conversion for Gemini."""
        from app.providers import GeminiProvider
        
        provider = GeminiProvider(model="gemini-2.5-flash", api_key="AIza_test")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        prompt = provider._convert_messages_to_prompt(messages)
        
        assert "SYSTEM:" in prompt
        assert "USER:" in prompt
        assert "You are a helpful assistant" in prompt
        assert "Hello" in prompt


class TestProviderBase:
    """Tests for ProviderBase class."""

    def test_provider_base_not_implemented(self):
        """Test that ProviderBase methods raise NotImplementedError."""
        from app.providers import ProviderBase
        
        provider = ProviderBase(model="test", api_key="key")
        
        with pytest.raises(NotImplementedError):
            list(provider.stream_chat([]))
        
        with pytest.raises(NotImplementedError):
            provider.embed([])
