from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base
import uuid

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    agents = relationship("Agent", back_populates="owner")
    messages = relationship("Message", back_populates="user")

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    drive_folder_id = Column(String, nullable=True)
    persona = Column(Text, default="")
    provider = Column(String, default="openai")
    model = Column(String, default="gpt-4o-mini")
    embed_model = Column(String, default="openai:text-embedding-3-small")
    api_key = Column(String, nullable=True)  # Store API key per agent
    last_indexed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    query_logs = relationship("QueryLog", back_populates="agent", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan")

    # 🚀 Add this line
    agent_files = relationship("AgentFile", backref="agent", cascade="all, delete-orphan")

    owner = relationship("User", back_populates="agents" )


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    title = Column(String, default="New chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    agent = relationship("Agent", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role = Column(String)  # 'user' or 'assistant' or 'system'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="messages")

class GoogleToken(Base):
    __tablename__ = "google_tokens"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    token_json = Column(Text)  # serialized credentials

    __table_args__ = (UniqueConstraint('user_id', name='uq_google_token_user'),)

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)

    #agent = relationship("Agent")
    agent = relationship("Agent", back_populates="query_logs")


class AgentFile(Base):
    __tablename__ = "agent_files"

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), index=True, nullable=False)

    file_id = Column(String, nullable=False)   # the document ID
    title = Column(String, nullable=True)
    source = Column(String, nullable=True)

    page = Column(Integer, nullable=True)      # <-- REQUIRED
    slide = Column(Integer, nullable=True)     # <-- REQUIRED
    text = Column(Text, nullable=True) 
    last_modified = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("agent_id", "file_id", "page", "slide",
                         name="uq_agentfile_filepage"),
    )

