# tests/test_agents.py
"""
Unit tests for app/agents.py

Tests cover:
- Agent validation functions (API key format, drive folder)
- Supported models configuration
- Default model selection
"""

import pytest
from unittest.mock import MagicMock, patch


class TestSupportedModels:
    """Tests for supported models configuration."""

    def test_supported_models_structure(self):
        """Test that SUPPORTED_MODELS has correct structure."""
        from app.agents import SUPPORTED_MODELS
        
        assert "openai" in SUPPORTED_MODELS
        assert "gemini" in SUPPORTED_MODELS
        
        for provider in ["openai", "gemini"]:
            assert "chat" in SUPPORTED_MODELS[provider]
            assert "embed" in SUPPORTED_MODELS[provider]
            assert "default_chat" in SUPPORTED_MODELS[provider]
            assert "default_embed" in SUPPORTED_MODELS[provider]
            
            assert isinstance(SUPPORTED_MODELS[provider]["chat"], list)
            assert isinstance(SUPPORTED_MODELS[provider]["embed"], list)
            assert len(SUPPORTED_MODELS[provider]["chat"]) > 0
            assert len(SUPPORTED_MODELS[provider]["embed"]) > 0

    def test_openai_models_list(self):
        """Test OpenAI models are correctly defined."""
        from app.agents import SUPPORTED_MODELS
        
        openai_chat = SUPPORTED_MODELS["openai"]["chat"]
        openai_embed = SUPPORTED_MODELS["openai"]["embed"]
        
        assert "gpt-4o-mini" in openai_chat
        assert "text-embedding-3-small" in openai_embed
        assert SUPPORTED_MODELS["openai"]["default_chat"] in openai_chat
        assert SUPPORTED_MODELS["openai"]["default_embed"] in openai_embed

    def test_gemini_models_list(self):
        """Test Gemini models are correctly defined."""
        from app.agents import SUPPORTED_MODELS
        
        gemini_chat = SUPPORTED_MODELS["gemini"]["chat"]
        gemini_embed = SUPPORTED_MODELS["gemini"]["embed"]
        
        assert "gemini-2.5-flash" in gemini_chat
        assert "models/text-embedding-004" in gemini_embed
        assert SUPPORTED_MODELS["gemini"]["default_chat"] in gemini_chat
        assert SUPPORTED_MODELS["gemini"]["default_embed"] in gemini_embed


class TestDefaultModelSelection:
    """Tests for default model selection functions."""

    def test_get_default_model_openai(self):
        """Test getting default chat model for OpenAI."""
        from app.agents import get_default_model, SUPPORTED_MODELS
        
        default = get_default_model("openai")
        assert default == SUPPORTED_MODELS["openai"]["default_chat"]

    def test_get_default_model_gemini(self):
        """Test getting default chat model for Gemini."""
        from app.agents import get_default_model, SUPPORTED_MODELS
        
        default = get_default_model("gemini")
        assert default == SUPPORTED_MODELS["gemini"]["default_chat"]

    def test_get_default_model_unknown_provider(self):
        """Test getting default model for unknown provider falls back to OpenAI."""
        from app.agents import get_default_model, SUPPORTED_MODELS
        
        default = get_default_model("unknown_provider")
        assert default == SUPPORTED_MODELS["openai"]["default_chat"]

    def test_get_default_embed_model_openai(self):
        """Test getting default embedding model for OpenAI."""
        from app.agents import get_default_embed_model, SUPPORTED_MODELS
        
        default = get_default_embed_model("openai")
        assert default == SUPPORTED_MODELS["openai"]["default_embed"]

    def test_get_default_embed_model_gemini(self):
        """Test getting default embedding model for Gemini."""
        from app.agents import get_default_embed_model, SUPPORTED_MODELS
        
        default = get_default_embed_model("gemini")
        assert default == SUPPORTED_MODELS["gemini"]["default_embed"]


class TestAPIKeyFormatValidation:
    """Tests for API key format validation (not actual API calls)."""

    def test_validate_api_key_openai_valid(self):
        """Test valid OpenAI API key format."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("openai", "sk-1234567890abcdefghijklmnop")
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_openai_wrong_prefix(self):
        """Test OpenAI API key with wrong prefix."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("openai", "abc-1234567890")
        assert is_valid is False
        assert "sk-" in error

    def test_validate_api_key_openai_too_short(self):
        """Test OpenAI API key that's too short."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("openai", "sk-123")
        assert is_valid is False
        assert "short" in error.lower()

    def test_validate_api_key_gemini_valid(self):
        """Test valid Gemini API key format."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("gemini", "AIza_12345678901234567890123456789012")
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_gemini_wrong_prefix(self):
        """Test Gemini API key with wrong prefix."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("gemini", "sk-1234567890")
        assert is_valid is False
        assert "AI" in error

    def test_validate_api_key_gemini_too_short(self):
        """Test Gemini API key that's too short."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("gemini", "AIza_123")
        assert is_valid is False
        assert "short" in error.lower()

    def test_validate_api_key_empty(self):
        """Test empty API key."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("openai", "")
        assert is_valid is False
        assert "required" in error.lower()

    def test_validate_api_key_whitespace_only(self):
        """Test API key with only whitespace."""
        from app.agents import validate_api_key
        
        is_valid, error = validate_api_key("openai", "   ")
        assert is_valid is False
        assert "required" in error.lower()


class TestDriveFolderValidation:
    """Tests for Drive folder validation."""

    def test_validate_drive_folder_valid(self):
        """Test valid drive folder URL."""
        from app.agents import validate_drive_folder
        
        is_valid, error = validate_drive_folder("https://drive.google.com/drive/folders/1ABC123")
        assert is_valid is True
        assert error == ""

    def test_validate_drive_folder_empty(self):
        """Test empty drive folder."""
        from app.agents import validate_drive_folder
        
        is_valid, error = validate_drive_folder("")
        assert is_valid is False
        assert "required" in error.lower()

    def test_validate_drive_folder_whitespace(self):
        """Test drive folder with only whitespace."""
        from app.agents import validate_drive_folder
        
        is_valid, error = validate_drive_folder("   ")
        assert is_valid is False
        assert "required" in error.lower()


class TestAgentConfigValidation:
    """Tests for full agent configuration validation."""

    def test_validate_agent_config_all_valid(self):
        """Test fully valid agent configuration."""
        from app.agents import validate_agent_config
        
        is_valid, error = validate_agent_config(
            provider="openai",
            api_key="sk-1234567890abcdefghijklmnop",
            drive_folder="https://drive.google.com/drive/folders/1ABC",
            require_api_key=True,
            require_drive=True
        )
        assert is_valid is True
        assert error == ""

    def test_validate_agent_config_missing_api_key(self):
        """Test configuration with missing API key when required."""
        from app.agents import validate_agent_config
        
        is_valid, error = validate_agent_config(
            provider="openai",
            api_key="",
            drive_folder="https://drive.google.com/drive/folders/1ABC",
            require_api_key=True,
            require_drive=True
        )
        assert is_valid is False
        assert "required" in error.lower()

    def test_validate_agent_config_api_key_optional(self):
        """Test configuration with optional API key."""
        from app.agents import validate_agent_config
        
        is_valid, error = validate_agent_config(
            provider="openai",
            api_key="",
            drive_folder="https://drive.google.com/drive/folders/1ABC",
            require_api_key=False,
            require_drive=True
        )
        assert is_valid is True
        assert error == ""

    def test_validate_agent_config_missing_drive(self):
        """Test configuration with missing drive folder when required."""
        from app.agents import validate_agent_config
        
        is_valid, error = validate_agent_config(
            provider="openai",
            api_key="sk-1234567890abcdefghijklmnop",
            drive_folder="",
            require_api_key=True,
            require_drive=True
        )
        assert is_valid is False
        assert "drive" in error.lower() or "required" in error.lower()

    def test_validate_agent_config_invalid_api_key_format(self):
        """Test configuration with invalid API key format."""
        from app.agents import validate_agent_config
        
        is_valid, error = validate_agent_config(
            provider="openai",
            api_key="invalid-key-format",
            drive_folder="https://drive.google.com/drive/folders/1ABC",
            require_api_key=True,
            require_drive=True
        )
        assert is_valid is False
        assert "sk-" in error
