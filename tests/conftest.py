# tests/conftest.py
"""
Pytest configuration and fixtures for Course TA Agent Studio tests.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing database dependencies
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import Base
    from app.models import User, Agent, Conversation, Message, GoogleToken, QueryLog, AgentFile
    HAS_DB_DEPS = True
except ImportError:
    HAS_DB_DEPS = False


# ============ DATABASE FIXTURES ============

@pytest.fixture(scope="function")
def test_engine():
    """Create an in-memory SQLite database engine for testing."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create a database session for testing."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============ MODEL FIXTURES ============

@pytest.fixture
def sample_user(test_db):
    """Create a sample user for testing."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    from app.security import hash_password
    user = User(
        email="test@example.com",
        password_hash=hash_password("testpassword123")
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def sample_agent(test_db, sample_user):
    """Create a sample agent for testing."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    agent = Agent(
        owner_id=sample_user.id,
        name="Test TA Agent",
        slug="test-agent-slug",
        drive_folder_id="1ABC123XYZ",
        persona="You are a helpful teaching assistant for a computer science course.",
        provider="gemini",
        model="gemini-2.5-flash",
        embed_model="models/text-embedding-004",
        api_key="AIza_test_api_key_12345"
    )
    test_db.add(agent)
    test_db.commit()
    test_db.refresh(agent)
    return agent


@pytest.fixture
def sample_conversation(test_db, sample_agent, sample_user):
    """Create a sample conversation for testing."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    conversation = Conversation(
        agent_id=sample_agent.id,
        user_id=sample_user.id,
        title="Test Conversation"
    )
    test_db.add(conversation)
    test_db.commit()
    test_db.refresh(conversation)
    return conversation


@pytest.fixture
def sample_messages(test_db, sample_conversation, sample_user):
    """Create sample messages in a conversation."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    messages = [
        Message(
            conversation_id=sample_conversation.id,
            user_id=sample_user.id,
            role="user",
            content="What is the homework deadline?"
        ),
        Message(
            conversation_id=sample_conversation.id,
            user_id=sample_user.id,
            role="assistant",
            content="The homework is due next Friday at 11:59 PM."
        ),
    ]
    for msg in messages:
        test_db.add(msg)
    test_db.commit()
    for msg in messages:
        test_db.refresh(msg)
    return messages


@pytest.fixture
def sample_google_token(test_db, sample_user):
    """Create a sample Google token for testing."""
    if not HAS_DB_DEPS:
        pytest.skip("SQLAlchemy not installed")
    token = GoogleToken(
        user_id=sample_user.id,
        token_json='{"access_token": "test_token", "refresh_token": "test_refresh"}'
    )
    test_db.add(token)
    test_db.commit()
    test_db.refresh(token)
    return token


# ============ MOCK FIXTURES ============

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    with patch('app.providers.OpenAI') as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        
        # Mock embeddings
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        client_instance.embeddings.create.return_value = embedding_response
        
        # Mock chat completions
        chat_response = MagicMock()
        chat_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        client_instance.chat.completions.create.return_value = chat_response
        
        yield mock


@pytest.fixture
def mock_gemini():
    """Mock Gemini client for testing without API calls."""
    with patch('app.providers.genai') as mock:
        # Mock configure
        mock.configure = MagicMock()
        
        # Mock embed_content
        mock.embed_content.return_value = {"embedding": [0.1] * 768}
        
        # Mock GenerativeModel
        model_instance = MagicMock()
        model_instance.generate_content.return_value = MagicMock(text="Test response")
        mock.GenerativeModel.return_value = model_instance
        
        yield mock


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request object."""
    request = MagicMock()
    request.cookies = {}
    request.session = {}
    return request


# ============ ENVIRONMENT FIXTURES ============

@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for testing."""
    os.environ["SECRET_KEY"] = "test_secret_key_for_testing"
    os.environ["TESTING"] = "1"
    yield
    # Cleanup
    if "SECRET_KEY" in os.environ:
        del os.environ["SECRET_KEY"]
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
