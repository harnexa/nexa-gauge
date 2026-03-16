import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Central config — all values sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Runtime environment
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM / Judge
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Web search
    TAVILY_API_KEY: Optional[str] = None
    WEB_SEARCH_ENABLED: bool = False

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector storage
    LANCEDB_PATH: str = "./.lancedb"
    LANCEDB_MCP_URI: Optional[str] = None

    # Evidence routing
    EVIDENCE_THRESHOLD: float = 0.75

    # Job execution
    MAX_CONCURRENT_JOBS: int = 4
    BUDGET_CAP_USD: Optional[float] = None

    # Composite score weights (must sum to 1.0)
    SCORE_WEIGHT_FAITHFULNESS: float = 0.4
    SCORE_WEIGHT_HALLUCINATION: float = 0.3
    SCORE_WEIGHT_RUBRIC: float = 0.2
    SCORE_WEIGHT_SAFETY: float = 0.1


config = Config()
