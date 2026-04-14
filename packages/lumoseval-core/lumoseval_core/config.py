from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from lumoseval_core.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LANCEDB_PATH,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_CONCURRENT_JOBS,
    EVIDENCE_VERDICT_SUPPORTED_THRESHOLD,
)


class Config(BaseSettings):
    """Central config — all values sourced from environment variables.

    Defaults come from constants.py so there is a single source of truth.
    Set any of these via a .env file or shell environment to override.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # treat empty string env vars as unset
    )

    # Runtime environment
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM / Judge
    LLM_PROVIDER: str = DEFAULT_LLM_PROVIDER
    LLM_MODEL: str = DEFAULT_JUDGE_MODEL
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Web search
    TAVILY_API_KEY: Optional[str] = None
    WEB_SEARCH_ENABLED: bool = False

    # Embeddings
    EMBEDDING_MODEL: str = DEFAULT_EMBEDDING_MODEL

    # Vector storage
    LANCEDB_PATH: str = DEFAULT_LANCEDB_PATH
    LANCEDB_MCP_URI: Optional[str] = None

    # Evidence routing
    EVIDENCE_THRESHOLD: float = EVIDENCE_VERDICT_SUPPORTED_THRESHOLD

    # Job execution
    MAX_CONCURRENT_JOBS: int = DEFAULT_MAX_CONCURRENT_JOBS
    BUDGET_CAP_USD: Optional[float] = None


config = Config()
