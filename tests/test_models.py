# tests/test_models.py
"""
Unit tests for app/models.py

Tests cover:
- SQLAlchemy model definitions
- Model relationships
- Default values and constraints
"""

import pytest
from datetime import datetime


class TestUserModel:
    """Tests for User model."""

    def test_user_creation(self, test_db):
        """Test creating a user."""
        from app.models import User
        from app.security import hash_password
        
        user = User(
            email="newuser@example.com",
            password_hash=hash_password("password123")
        )
        test_db.add(user)
        test_db.commit()
        
        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.created_at is not None

    def test_user_email_unique(self, test_db, sample_user):
        """Test that user emails must be unique."""
        from app.models import User
        from sqlalchemy.exc import IntegrityError
        
        duplicate_user = User(
            email=sample_user.email,  # Same email
            password_hash="hash"
        )
        test_db.add(duplicate_user)
        
        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_user_agents_relationship(self, test_db, sample_user, sample_agent):
        """Test User -> Agents relationship."""
        test_db.refresh(sample_user)
        
        assert len(sample_user.agents) == 1
        assert sample_user.agents[0].id == sample_agent.id


class TestAgentModel:
    """Tests for Agent model."""

    def test_agent_creation(self, test_db, sample_user):
        """Test creating an agent."""
        from app.models import Agent
        
        agent = Agent(
            owner_id=sample_user.id,
            name="Test Agent",
            slug="test-slug-123",
            drive_folder_id="folder123",
            persona="You are a helpful TA",
            provider="gemini",
            model="gemini-2.5-flash",
            embed_model="models/text-embedding-004"
        )
        test_db.add(agent)
        test_db.commit()
        
        assert agent.id is not None
        assert agent.name == "Test Agent"
        assert agent.created_at is not None
        assert agent.updated_at is not None

    def test_agent_default_values(self, test_db, sample_user):
        """Test agent default values."""
        from app.models import Agent
        
        agent = Agent(
            owner_id=sample_user.id,
            name="Minimal Agent",
            slug="minimal-slug"
        )
        test_db.add(agent)
        test_db.commit()
        
        assert agent.provider == "openai"  # Default
        assert agent.model == "gpt-4o-mini"  # Default
        assert agent.persona == ""  # Default

    def test_agent_slug_unique(self, test_db, sample_user, sample_agent):
        """Test that agent slugs must be unique."""
        from app.models import Agent
        from sqlalchemy.exc import IntegrityError
        
        duplicate_agent = Agent(
            owner_id=sample_user.id,
            name="Another Agent",
            slug=sample_agent.slug  # Same slug
        )
        test_db.add(duplicate_agent)
        
        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_agent_owner_relationship(self, test_db, sample_agent, sample_user):
        """Test Agent -> Owner relationship."""
        test_db.refresh(sample_agent)
        
        assert sample_agent.owner is not None
        assert sample_agent.owner.id == sample_user.id

    def test_agent_conversations_relationship(self, test_db, sample_agent, sample_conversation):
        """Test Agent -> Conversations relationship."""
        test_db.refresh(sample_agent)
        
        assert len(sample_agent.conversations) == 1
        assert sample_agent.conversations[0].id == sample_conversation.id


class TestConversationModel:
    """Tests for Conversation model."""

    def test_conversation_creation(self, test_db, sample_agent, sample_user):
        """Test creating a conversation."""
        from app.models import Conversation
        
        conv = Conversation(
            agent_id=sample_agent.id,
            user_id=sample_user.id,
            title="Test Conversation"
        )
        test_db.add(conv)
        test_db.commit()
        
        assert conv.id is not None
        assert conv.title == "Test Conversation"
        assert conv.created_at is not None

    def test_conversation_default_title(self, test_db, sample_agent, sample_user):
        """Test conversation default title."""
        from app.models import Conversation
        
        conv = Conversation(
            agent_id=sample_agent.id,
            user_id=sample_user.id
        )
        test_db.add(conv)
        test_db.commit()
        
        assert conv.title == "New chat"

    def test_conversation_messages_relationship(self, test_db, sample_conversation, sample_messages):
        """Test Conversation -> Messages relationship."""
        test_db.refresh(sample_conversation)
        
        assert len(sample_conversation.messages) == 2


class TestMessageModel:
    """Tests for Message model."""

    def test_message_creation(self, test_db, sample_conversation, sample_user):
        """Test creating a message."""
        from app.models import Message
        
        msg = Message(
            conversation_id=sample_conversation.id,
            user_id=sample_user.id,
            role="user",
            content="Hello, I have a question"
        )
        test_db.add(msg)
        test_db.commit()
        
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "Hello, I have a question"
        assert msg.created_at is not None

    def test_message_is_summarized_default(self, test_db, sample_conversation, sample_user):
        """Test message is_summarized default value."""
        from app.models import Message
        
        msg = Message(
            conversation_id=sample_conversation.id,
            user_id=sample_user.id,
            role="assistant",
            content="Here's your answer"
        )
        test_db.add(msg)
        test_db.commit()
        
        assert msg.is_summarized is False

    def test_message_conversation_relationship(self, test_db, sample_messages, sample_conversation):
        """Test Message -> Conversation relationship."""
        msg = sample_messages[0]
        test_db.refresh(msg)
        
        assert msg.conversation is not None
        assert msg.conversation.id == sample_conversation.id


class TestGoogleTokenModel:
    """Tests for GoogleToken model."""

    def test_google_token_creation(self, test_db, sample_user):
        """Test creating a Google token."""
        from app.models import GoogleToken
        
        token = GoogleToken(
            user_id=sample_user.id,
            token_json='{"access_token": "test", "refresh_token": "test_refresh"}'
        )
        test_db.add(token)
        test_db.commit()
        
        assert token.id is not None
        assert token.user_id == sample_user.id

    def test_google_token_user_unique(self, test_db, sample_user, sample_google_token):
        """Test that each user can have only one Google token."""
        from app.models import GoogleToken
        from sqlalchemy.exc import IntegrityError
        
        duplicate_token = GoogleToken(
            user_id=sample_user.id,  # Same user
            token_json='{"another": "token"}'
        )
        test_db.add(duplicate_token)
        
        with pytest.raises(IntegrityError):
            test_db.commit()


class TestQueryLogModel:
    """Tests for QueryLog model."""

    def test_query_log_creation(self, test_db, sample_agent):
        """Test creating a query log entry."""
        from app.models import QueryLog
        
        log = QueryLog(
            agent_id=sample_agent.id,
            query="What is the homework deadline?"
        )
        test_db.add(log)
        test_db.commit()
        
        assert log.id is not None
        assert log.timestamp is not None
        assert log.query == "What is the homework deadline?"

    def test_query_log_agent_relationship(self, test_db, sample_agent):
        """Test QueryLog -> Agent relationship."""
        from app.models import QueryLog
        
        log = QueryLog(
            agent_id=sample_agent.id,
            query="Test query"
        )
        test_db.add(log)
        test_db.commit()
        test_db.refresh(log)
        
        assert log.agent is not None
        assert log.agent.id == sample_agent.id


class TestAgentFileModel:
    """Tests for AgentFile model."""

    def test_agent_file_creation(self, test_db, sample_agent):
        """Test creating an agent file record."""
        from app.models import AgentFile
        
        file = AgentFile(
            agent_id=sample_agent.id,
            file_id="file123",
            title="Lecture 1.pdf",
            source="https://drive.google.com/...",
            page=1,
            text="Lecture content here..."
        )
        test_db.add(file)
        test_db.commit()
        
        assert file.id is not None
        assert file.title == "Lecture 1.pdf"

    def test_agent_file_unique_constraint(self, test_db, sample_agent):
        """Test AgentFile unique constraint on agent_id, file_id, page, slide."""
        from app.models import AgentFile
        
        file1 = AgentFile(
            agent_id=sample_agent.id,
            file_id="file123",
            page=1,
            slide=None
        )
        test_db.add(file1)
        test_db.commit()
        
        # Adding another file with same file_id is allowed
        # (model doesn't enforce unique constraint on file_id alone)
        file2 = AgentFile(
            agent_id=sample_agent.id,
            file_id="file123",
            page=2,  # Different page
            slide=None
        )
        test_db.add(file2)
        test_db.commit()
        
        # Both files should exist
        files = test_db.query(AgentFile).filter_by(agent_id=sample_agent.id).all()
        assert len(files) == 2
