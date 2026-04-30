"""
NexaGauge pipeline constants.

Single source of truth for every default value and magic number in the pipeline.
Values here are compile-time constants — things that do not change at runtime.
For values that are user-configurable via environment variables, see config.py,
which uses these constants as its defaults.

Sections
────────
  TOKENIZATION      Encoding and chunk sizing
  COST_ESTIMATION   Pricing heuristics
  MMR               Chunk refinement thresholds
  EVIDENCE          Retrieval limits and verdict thresholds
  METRICS           Per-metric pass/fail thresholds and composite score weights
  LLM               Model defaults and gateway settings
  STORAGE           Database and cache paths
  DATASET           Dataset adapter defaults
"""

import os
import sys
from pathlib import Path

# ── Tokenization & Chunking ──────────────────────────────────────────────────

# tiktoken encoding used for token counting across scanner and chunker.
# cl100k_base is compatible with GPT-4 and gives a good approximation for
# Anthropic and other providers.
TIKTOKEN_ENCODING: str = "cl100k_base"

# Target chunk size (in tokens) for the text splitter and the scanner's
# heuristic chunk count estimate.
GENERATION_CHUNK_SIZE_TOKENS: int = 100

# Minimum token count required before a chunk is eligible for splitting.
CHUNK_MIN_TOKENS_FOR_SPLIT: int = 100

# Heuristic: average number of verifiable claims expected per chunk.
# Used by the scanner to estimate claim_count before any LLM calls.
CLAIMS_PER_CHUNK: int = 1

# ── Cost Estimation ──────────────────────────────────────────────────────────

# Average token count used to simulate a single judge LLM call for pricing.
# This covers the claim-verification prompt + expected response length.
COST_AVG_JUDGE_TOKENS: int = 250

# Conservative per-call cost fallback used when tokencost cannot price the
# chosen model (e.g. a private deployment).
COST_FALLBACK_PER_CALL_USD: float = 0.0003

# ── Per-Node Token Estimation Heuristics ─────────────────────────────────────
# These constants drive the per-node cost_estimate() methods. They model the
# dynamic content that varies at runtime (claims, context, questions, outputs)
# without actually running any LLM calls.

# Average tokens in a single extracted claim text.
AVG_CLAIM_INPUT_TOKENS: int = 25 + 12  # 25 claims token, and 12 structured LLM output tokens
AVG_CLAIMS_PER_CHUNK: int = 2  # Average number of claims extracted per chunk

# Average output tokens for a single boolean verdict (grounding): "true"/"false".
AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT: int = 7

# DeepEval BiasMetric / ToxicityMetric each make internal LLM calls whose
# prompts are not directly accessible. These constants approximate that overhead.
AVG_DEEPEVAL_PROMPT_TOKENS: int = 100
AVG_DEEPEVAL_OUTPUT_REASONING_TOKENS: int = 50
AVG_DEEPEVAL_OUTPUT_VERDICT: int = 14


# DeepEval GEval constructs a multi-step evaluation prompt per metric.
AVG_DEEPEVAL_GEVAL_CRITERIA_STEPS: int = 3
AVG_DEEPEVAL_GEVAL_CRITERIA_STEP_TOKENS: int = 40

# ── Chunk/Refiner Strategies ─────────────────────────────────────────────────

# Default chunking strategy identifier.
DEFAULT_CHUNKER_STRATEGY: str = "semchunk"

# Default refiner strategy identifier.
DEFAULT_REFINER_STRATEGY: str = "mmr"

# ── MMR Refiner ───────────────────────────────────────────────────────────────

# Cosine similarity threshold above which a candidate claim is considered a
# duplicate of an already-selected claim and is discarded.
MMR_SIMILARITY_THRESHOLD: float = 0.7

# MMR λ (lambda) — weight balancing relevance (λ) vs. diversity (1-λ).
# 0.5 = equal weight; increase toward 1.0 to keep higher-confidence claims
# even if they are similar; decrease toward 0.0 to maximise diversity.
MMR_LAMBDA: float = 0.5

# Maximum number of chunks to keep after MMR refinement.
REFINER_TOP_K: int = 3

# ── Evidence Retrieval ───────────────────────────────────────────────────────

# Retrieval score (0–1) at or above which a claim is labelled SUPPORTED.
# Also used as the default evidence_threshold in EvalJobConfig and Config.
EVIDENCE_VERDICT_SUPPORTED_THRESHOLD: float = 0.75

# ── Metrics ──────────────────────────────────────────────────────────────────

# Score at or above which a metric is considered "passed".
# Applied uniformly to hallucination, GEval, and bias metrics.
GROUNDING_METRIC_PASS_THRESHOLD: float = 0.6  # Atleast 60% of claims should be supported
RELEVANCE_METRIC_PASS_THRESHOLD: float = 0.6  # Atleast 60% of claims should be supported
GEVAL_METRIC_PASS_THRESHOLD: float = 0.6
REDTEAM_METRIC_PASS_THRESHOLD: float = 0.6


# ── LLM ──────────────────────────────────────────────────────────────────────

# Default LiteLLM model used as judge when no override is provided.
DEFAULT_JUDGE_MODEL: str = "gpt-4o-mini"

# Default LLM provider prefix used for routing in LiteLLM.
DEFAULT_LLM_PROVIDER: str = "openai"

# Fully-qualified default primary model used by the CLI (``provider/model``).
DEFAULT_PRIMARY_LLM: str = f"{DEFAULT_LLM_PROVIDER}/{DEFAULT_JUDGE_MODEL}"

# Fully-qualified default fallback model used when the primary errors.
DEFAULT_FALLBACK_LLM: str = f"{DEFAULT_LLM_PROVIDER}/gpt-4o"

# Default sentence-transformer model used for local embeddings.
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Wall-clock timeout in seconds for a single litellm.completion() call.
LLM_CALL_TIMEOUT_SECONDS: int = 60

# ── Storage ───────────────────────────────────────────────────────────────────


def default_cache_dir() -> Path:
    """Return the per-user cache root for nexagauge.

    Follows the XDG Base Directory spec on macOS and Linux — the convention
    used by uv, pip, huggingface, and other Python CLI tools — so cached
    artifacts live in one place per user regardless of which directory the
    ``nexagauge`` CLI is invoked from.

    Resolution:
      - macOS / Linux : ``$XDG_CACHE_HOME/nexagauge`` or ``~/.cache/nexagauge``
      - Windows       : ``%LOCALAPPDATA%\\nexagauge``

    Callers can override this via ``NEXAGAUGE_CACHE_DIR`` (env var) or the
    ``--cache-dir`` CLI flag — see :class:`ng_core.cache.CacheStore`.
    """
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "nexagauge"


# ── Dataset Adapter ───────────────────────────────────────────────────────────

# Default dataset name assigned to cases that don't carry an explicit name.
DEFAULT_DATASET_NAME: str = "user_dataset"

# Default dataset split used by adapters when no split is specified.
DEFAULT_SPLIT: str = "train"

# ── Job Execution ─────────────────────────────────────────────────────────────
# Node-local internal fan-out caps (within a single case execution).
DEFAULT_CLAIMS_MAX_WORKERS: int = 8
DEFAULT_GEVAL_STEPS_MAX_WORKERS: int = 8
DEFAULT_REDTEAM_MAX_WORKERS: int = 8
