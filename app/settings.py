# app/settings.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    BASE_URL: str = Field(..., description="e.g., http://127.0.0.1:8000")
    SECRET_KEY: str = "dev"

    # Google OAuth
    GOOGLE_API_CLIENT_ID: str | None = None
    GOOGLE_API_CLIENT_SECRET: str | None = None

    # Providers (optional)
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # RAG
    EMBED_MODEL: str = "openai:text-embedding-3-small"
    SCHED_CRON: str = "0 3 * * *"

    # Storage root for DB / vector stores / tokens; relative to project root if not absolute
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
