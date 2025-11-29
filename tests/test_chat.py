# tests/test_chat.py
"""
Unit tests for app/chat.py

Tests cover:
- Conversation management
- Message handling
- Summarization logic
"""

import pytest
from unittest.mock import MagicMock, patch


class TestEnsureConversation:
    """Tests for _ensure_conversation function."""

    def test_ensure_conversation_creates_new(self, test_db, sample_agent, sample_user):
        """Test that a new conversation is created when none exists."""
        from app.chat import _ensure_conversation
        
        conv = _ensure_conversation(test_db, sample_agent.id, sample_user.id)
        
        assert conv is not None
        assert conv.agent_id == sample_agent.id
        assert conv.user_id == sample_user.id
        assert conv.title == "Conversation"

    def test_ensure_conversation_returns_existing(self, test_db, sample_agent, sample_user, sample_conversation):
        """Test that existing conversation is returned."""
        from app.chat import _ensure_conversation
        
        conv = _ensure_conversation(test_db, sample_agent.id, sample_user.id)
        
        assert conv.id == sample_conversation.id


class TestGetLastMessages:
    """Tests for _get_last_messages function."""

    def test_get_last_messages_empty(self, test_db, sample_conversation):
        """Test getting messages from empty conversation."""
        from app.chat import _get_last_messages
        
        last_msgs, all_msgs = _get_last_messages(test_db, sample_conversation, k=4)
        
        assert len(last_msgs) == 0
        assert len(all_msgs) == 0

    def test_get_last_messages_returns_active(self, test_db, sample_conversation, sample_messages):
        """Test that only active (non-summarized) messages are returned."""
        from app.chat import _get_last_messages
        
        last_msgs, all_msgs = _get_last_messages(test_db, sample_conversation, k=4)
        
        # All sample messages should be active (is_summarized=False by default)
        assert len(last_msgs) == 2
        assert len(all_msgs) == 2

    def test_get_last_messages_respects_limit(self, test_db, sample_conversation, sample_user):
        """Test that k limits the number of messages returned."""
        from app.models import Message
        from app.chat import _get_last_messages
        
        # Create more messages
        for i in range(10):
            msg = Message(
                conversation_id=sample_conversation.id,
                user_id=sample_user.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
            test_db.add(msg)
        test_db.commit()
        
        last_msgs, all_msgs = _get_last_messages(test_db, sample_conversation, k=4)
        
        assert len(last_msgs) == 4
        assert len(all_msgs) == 10


class TestSummarizeMemory:
    """Tests for _summarize_memory function."""

    def test_summarize_memory_with_messages(self):
        """Test summarization with messages."""
        from app.chat import _summarize_memory
        
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "This is a summary of the conversation."
        
        mock_messages = [
            MagicMock(role="user", content="Hello"),
            MagicMock(role="assistant", content="Hi there!")
        ]
        
        result = _summarize_memory(mock_provider, "", mock_messages)
        
        assert result == "This is a summary of the conversation."
        mock_provider.complete.assert_called_once()

    def test_summarize_memory_with_existing_summary(self):
        """Test summarization preserves existing summary context."""
        from app.chat import _summarize_memory
        
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "Updated summary."
        
        mock_messages = [MagicMock(role="user", content="New message")]
        
        result = _summarize_memory(mock_provider, "Previous summary here", mock_messages)
        
        # Verify the prompt includes previous summary
        call_args = mock_provider.complete.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "Previous summary" in user_content

    def test_summarize_memory_handles_error(self):
        """Test summarization handles provider errors."""
        from app.chat import _summarize_memory
        
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = Exception("API Error")
        
        mock_messages = [MagicMock(role="user", content="Test")]
        
        result = _summarize_memory(mock_provider, "existing", mock_messages)
        
        # Should return existing summary on error
        assert result == "existing"

    def test_summarize_memory_empty_result(self):
        """Test summarization handles empty result."""
        from app.chat import _summarize_memory
        
        mock_provider = MagicMock()
        mock_provider.complete.return_value = ""
        
        mock_messages = [MagicMock(role="user", content="Test")]
        
        result = _summarize_memory(mock_provider, "existing", mock_messages)
        
        assert result == "existing"


class TestGeneralRecommendations:
    """Tests for general_recommendations endpoint."""

    @patch('app.chat.retrieve')
    @patch('app.chat.provider_from')
    def test_recommendations_with_docs(self, mock_provider_from, mock_retrieve, test_db, sample_agent):
        """Test recommendations generation with documents."""
        from app.chat import general_recommendations
        
        # Mock retrieve to return some docs
        mock_retrieve.return_value = [
            {"text": "Lecture 1 content about algorithms", "metadata": {"title": "Lecture 1"}},
            {"text": "Homework 1 details", "metadata": {"title": "HW1"}}
        ]
        
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "What are the key concepts in Lecture 1?\nWhen is Homework 1 due?\nHow do algorithms work?"
        mock_provider_from.return_value = mock_provider
        
        result = general_recommendations(sample_agent.id, test_db)
        
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    @patch('app.chat.retrieve')
    def test_recommendations_empty_collection(self, mock_retrieve, test_db, sample_agent):
        """Test recommendations with empty collection."""
        from app.chat import general_recommendations
        
        mock_retrieve.return_value = []
        
        result = general_recommendations(sample_agent.id, test_db)
        
        assert result == {"suggestions": []}

    def test_recommendations_agent_not_found(self, test_db):
        """Test recommendations with non-existent agent."""
        from app.chat import general_recommendations
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            general_recommendations(99999, test_db)
        
        assert exc_info.value.status_code == 404
