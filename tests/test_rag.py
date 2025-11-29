# tests/test_rag.py
"""
Unit tests for app/rag.py

Tests cover:
- File ID extraction from Google Drive URLs
- Drive folder validation
- Collection naming
- Document retrieval (mocked)
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFileIdFromUrl:
    """Tests for file_id_from_url_or_id function."""

    def test_extract_from_folder_url(self):
        """Test extracting folder ID from Google Drive folder URL."""
        from app.rag import file_id_from_url_or_id
        
        url = "https://drive.google.com/drive/folders/1ABC123XYZ?usp=sharing"
        result = file_id_from_url_or_id(url)
        
        assert result == "1ABC123XYZ"

    def test_extract_from_folder_url_no_params(self):
        """Test extracting folder ID from URL without query params."""
        from app.rag import file_id_from_url_or_id
        
        url = "https://drive.google.com/drive/folders/1ABC123XYZ"
        result = file_id_from_url_or_id(url)
        
        assert result == "1ABC123XYZ"

    def test_extract_from_open_url(self):
        """Test extracting file ID from Google Drive open URL."""
        from app.rag import file_id_from_url_or_id
        
        url = "https://drive.google.com/open?id=1ABC123XYZ&other=param"
        result = file_id_from_url_or_id(url)
        
        assert result == "1ABC123XYZ"

    def test_extract_plain_id(self):
        """Test that plain ID is returned as-is."""
        from app.rag import file_id_from_url_or_id
        
        folder_id = "1ABC123XYZ"
        result = file_id_from_url_or_id(folder_id)
        
        assert result == "1ABC123XYZ"

    def test_empty_string(self):
        """Test empty string returns empty."""
        from app.rag import file_id_from_url_or_id
        
        result = file_id_from_url_or_id("")
        
        assert result == ""

    def test_none_value(self):
        """Test None-like value returns empty."""
        from app.rag import file_id_from_url_or_id
        
        result = file_id_from_url_or_id(None)
        
        assert result == ""

    def test_trailing_slashes_removed(self):
        """Test that trailing slashes are removed from folder ID."""
        from app.rag import file_id_from_url_or_id
        
        url = "https://drive.google.com/drive/folders/1ABC123XYZ/"
        result = file_id_from_url_or_id(url)
        
        assert result == "1ABC123XYZ"


class TestCollectionNaming:
    """Tests for collection naming function."""

    def test_collection_name_format(self):
        """Test collection name format for an agent."""
        from app.rag import _collection_name
        
        result = _collection_name(42)
        
        assert result == "agent_42"

    def test_collection_name_different_ids(self):
        """Test collection names are unique per agent ID."""
        from app.rag import _collection_name
        
        name1 = _collection_name(1)
        name2 = _collection_name(2)
        
        assert name1 != name2
        assert "1" in name1
        assert "2" in name2


class TestDriveFolderValidation:
    """Tests for Google Drive folder validation."""

    @patch('app.rag._user_token_path')
    def test_validate_drive_folder_no_credentials(self, mock_token_path):
        """Test validation fails when no credentials file exists."""
        from app.rag import validate_drive_folder
        from pathlib import Path
        
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_token_path.return_value = mock_path
        
        is_valid, error = validate_drive_folder("1ABC123", user_id=1)
        
        assert is_valid is False
        assert "not connected" in error.lower()

    def test_validate_drive_folder_empty_url(self):
        """Test validation fails for empty folder URL."""
        from app.rag import validate_drive_folder
        
        is_valid, error = validate_drive_folder("", user_id=1)
        
        assert is_valid is False
        assert "Invalid" in error


class TestMakeDocId:
    """Tests for document ID generation."""

    def test_make_doc_id_with_source(self):
        """Test document ID generation with source metadata."""
        from app.rag import _make_doc_id
        
        metadata = {"source": "https://drive.google.com/file/123"}
        result = _make_doc_id(agent_id=1, md=metadata, idx=0)
        
        assert "https://drive.google.com/file/123" in result

    def test_make_doc_id_with_page(self):
        """Test document ID includes page number."""
        from app.rag import _make_doc_id
        
        metadata = {"source": "doc123", "page": 5}
        result = _make_doc_id(agent_id=1, md=metadata, idx=0)
        
        assert "p5" in result

    def test_make_doc_id_with_slide(self):
        """Test document ID includes slide number."""
        from app.rag import _make_doc_id
        
        metadata = {"source": "presentation", "slide": 3}
        result = _make_doc_id(agent_id=1, md=metadata, idx=0)
        
        assert "s3" in result

    def test_make_doc_id_fallback_to_agent(self):
        """Test document ID falls back to agent ID."""
        from app.rag import _make_doc_id
        
        metadata = {}
        result = _make_doc_id(agent_id=42, md=metadata, idx=5)
        
        assert "agent-42" in result
        assert "#i5" in result


class TestRetrieve:
    """Tests for document retrieval."""

    @patch('app.rag.get_vector_client')
    @patch('app.rag.ensure_collection')
    def test_retrieve_empty_collection(self, mock_ensure_col, mock_get_client):
        """Test retrieval from empty collection."""
        from app.rag import retrieve
        
        # Setup mocks
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_ensure_col.return_value = mock_collection
        
        mock_agent = MagicMock()
        mock_agent.id = 1
        
        result = retrieve(mock_agent, "test query", k=5)
        
        assert result == []

    @patch('app.rag.get_vector_client')
    @patch('app.rag.ensure_collection')
    def test_retrieve_with_results(self, mock_ensure_col, mock_get_client):
        """Test retrieval with results."""
        from app.rag import retrieve
        
        # Setup mocks
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1 text", "Document 2 text"]],
            "metadatas": [[{"title": "Doc1"}, {"title": "Doc2"}]]
        }
        mock_ensure_col.return_value = mock_collection
        
        mock_agent = MagicMock()
        mock_agent.id = 1
        
        result = retrieve(mock_agent, "test query", k=5)
        
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["text"] == "Document 1 text"
        assert result[0]["metadata"]["title"] == "Doc1"

    @patch('app.rag.get_vector_client')
    @patch('app.rag.ensure_collection')
    def test_retrieve_limits_k_to_collection_size(self, mock_ensure_col, mock_get_client):
        """Test that k is limited to collection size."""
        from app.rag import retrieve
        
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Text 1", "Text 2", "Text 3"]],
            "metadatas": [[{}, {}, {}]]
        }
        mock_ensure_col.return_value = mock_collection
        
        mock_agent = MagicMock()
        mock_agent.id = 1
        
        retrieve(mock_agent, "test", k=100)  # k > collection size
        
        # Verify query was called with limited k
        call_args = mock_collection.query.call_args
        assert call_args[1]["n_results"] <= 3


class TestProviderFrom:
    """Tests for provider_from function."""

    def test_provider_from_openai(self):
        """Test creating OpenAI provider from agent."""
        from app.rag import provider_from
        
        mock_agent = MagicMock()
        mock_agent.provider = "openai"
        mock_agent.model = "gpt-4o-mini"
        mock_agent.embed_model = "text-embedding-3-small"
        mock_agent.api_key = "sk-test123"
        
        with patch('app.rag.OpenAIProvider') as mock_provider:
            provider_from(mock_agent)
            # Provider is called with positional args
            mock_provider.assert_called_once_with(
                "gpt-4o-mini",
                "text-embedding-3-small",
                "sk-test123"
            )

    def test_provider_from_gemini(self):
        """Test creating Gemini provider from agent."""
        from app.rag import provider_from
        
        mock_agent = MagicMock()
        mock_agent.provider = "gemini"
        mock_agent.model = "gemini-2.5-flash"
        mock_agent.embed_model = "models/text-embedding-004"
        mock_agent.api_key = "AIza_test"
        
        with patch('app.rag.GeminiProvider') as mock_provider:
            provider_from(mock_agent)
            # Provider is called with positional args
            mock_provider.assert_called_once_with(
                "gemini-2.5-flash",
                "models/text-embedding-004",
                "AIza_test"
            )

    def test_provider_from_unknown_defaults_to_openai(self):
        """Test that unknown provider defaults to OpenAI."""
        from app.rag import provider_from
        
        mock_agent = MagicMock()
        mock_agent.provider = "unknown"
        mock_agent.model = "some-model"
        mock_agent.embed_model = "some-embed"
        mock_agent.api_key = "some-key"
        
        with patch('app.rag.OpenAIProvider') as mock_provider:
            provider_from(mock_agent)
            mock_provider.assert_called_once()
