from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from ng_core.constants import (
    DEFAULT_CLAIMS_MAX_WORKERS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEVAL_STEPS_MAX_WORKERS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_REDTEAM_MAX_WORKERS,
    DEFAULT_REFERENCE_ALIGN_MAX_WORKERS,
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

    # Embeddings
    EMBEDDING_MODEL: str = DEFAULT_EMBEDDING_MODEL

    # Evidence routing
    EVIDENCE_THRESHOLD: float = EVIDENCE_VERDICT_SUPPORTED_THRESHOLD

    # Job execution
    CLAIMS_MAX_WORKERS: int = DEFAULT_CLAIMS_MAX_WORKERS
    GEVAL_STEPS_MAX_WORKERS: int = DEFAULT_GEVAL_STEPS_MAX_WORKERS
    REDTEAM_MAX_WORKERS: int = DEFAULT_REDTEAM_MAX_WORKERS
    REFERENCE_ALIGN_MAX_WORKERS: int = DEFAULT_REFERENCE_ALIGN_MAX_WORKERS
    BUDGET_CAP_USD: Optional[float] = None


config = Config()
