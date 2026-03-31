"""
LumisEval pipeline constants.

Single source of truth for every default value and magic number in the pipeline.
Values here are compile-time constants — things that do not change at runtime.
For values that are user-configurable via environment variables, see config.py,
which uses these constants as its defaults.

Sections
────────
  TOKENIZATION      Encoding and chunk sizing
  COST_ESTIMATION   Pricing heuristics and confidence band
  MMR               Claim deduplication thresholds
  EVIDENCE          Retrieval limits and verdict thresholds
  METRICS           Per-metric pass/fail thresholds and composite score weights
  LLM               Model defaults and gateway settings
  STORAGE           Database and cache paths
  ADVERSARIAL       Default Giskard probe categories
  JOB               Job execution defaults
  DATASET           Dataset adapter defaults
"""

# ── Tokenization & Chunking ──────────────────────────────────────────────────

# tiktoken encoding used for token counting across scanner and chunker.
# cl100k_base is compatible with GPT-4 and gives a good approximation for
# Anthropic and other providers.
TIKTOKEN_ENCODING: str = "cl100k_base"

# Target chunk size (in tokens) for the text splitter and the scanner's
# heuristic chunk count estimate.
CHUNK_SIZE_TOKENS: int = 54

# Minimum token count before attempting semantic splitting.
# Short texts below this threshold are returned as a single chunk.
CHUNK_MIN_TOKENS_FOR_SPLIT: int = 100

# Heuristic: average number of verifiable claims expected per chunk.
# Used by the scanner to estimate claim_count before any LLM calls.
CLAIMS_PER_CHUNK: int = 3

# ── Cost Estimation ──────────────────────────────────────────────────────────

# Tavily web search price per API call (approximate, 2024 free tier).
COST_TAVILY_PER_CALL_USD: float = 0.001

# Average token count used to simulate a single judge LLM call for pricing.
# This covers the claim-verification prompt + expected response length.
COST_AVG_JUDGE_TOKENS: int = 250

# Fraction of claims expected to require a web-search fallback when
# WEB_SEARCH_ENABLED=true.  Used to estimate Tavily call count upfront.
COST_WEB_SEARCH_CLAIM_FRACTION: float = 0.4

# Average tokens per embedding call (one call per chunk).
# Sentence-transformer embeddings are local so this drives no API cost today,
# but is tracked for future cloud-embedding support.
COST_AVG_EMBEDDING_TOKENS: int = 400

# Confidence band multipliers applied to the central cost estimate to produce
# the low / high bounds shown in the pre-run cost table (±20%).
COST_ESTIMATE_BAND_LOW: float = 0.8
COST_ESTIMATE_BAND_HIGH: float = 1.2

# Conservative per-call cost fallback used when tokencost cannot price the
# chosen model (e.g. a private deployment).
COST_FALLBACK_PER_CALL_USD: float = 0.0003

# ── Per-Node Token Estimation Heuristics ─────────────────────────────────────
# These constants drive the per-node cost_estimate() methods. They model the
# dynamic content that varies at runtime (claims, context, questions, outputs)
# without actually running any LLM calls.

# Average tokens in a single extracted claim text.
COST_AVG_CLAIM_TOKENS: int = 15

# Average tokens of retrieved context passages passed to the grounding node
# per record (roughly one CHUNK_SIZE_TOKENS chunk from the evidence retriever).
COST_AVG_CONTEXT_TOKENS: int = 512

# Average tokens in a user question / query string.
COST_AVG_QUESTION_TOKENS: int = 25

# Average output tokens for a single boolean verdict (grounding): "true"/"false".
COST_AVG_OUTPUT_TOKENS_BOOLEAN_VERDICT: int = 5

# Average output tokens for a single JSON relevance verdict:
# {"verdict": "relevant"} is ~10 tokens.
COST_AVG_OUTPUT_TOKENS_JSON_VERDICT: int = 10

# DeepEval BiasMetric / ToxicityMetric each make internal LLM calls whose
# prompts are not directly accessible. These constants approximate that overhead.
COST_DEEPEVAL_INPUT_OVERHEAD_TOKENS: int = 350
COST_DEEPEVAL_OUTPUT_OVERHEAD_TOKENS: int = 50

# DeepEval GEval (rubric) constructs a multi-step evaluation prompt per rule.
COST_GEVAL_INPUT_OVERHEAD_TOKENS: int = 400
COST_GEVAL_OUTPUT_OVERHEAD_TOKENS: int = 60

# ── MMR Claim Deduplication ──────────────────────────────────────────────────

# Cosine similarity threshold above which a candidate claim is considered a
# duplicate of an already-selected claim and is discarded.
MMR_SIMILARITY_THRESHOLD: float = 0.9

# MMR λ (lambda) — weight balancing relevance (λ) vs. diversity (1-λ).
# 0.5 = equal weight; increase toward 1.0 to keep higher-confidence claims
# even if they are similar; decrease toward 0.0 to maximise diversity.
MMR_LAMBDA: float = 0.5

# ── Evidence Retrieval ───────────────────────────────────────────────────────

# Maximum number of passages returned per LanceDB vector query.
EVIDENCE_RETRIEVAL_TOP_K: int = 5

# Maximum number of results requested from Tavily per web-search query.
EVIDENCE_TAVILY_MAX_RESULTS: int = 5

# Retrieval score (0–1) at or above which a claim is labelled SUPPORTED.
# Also used as the default evidence_threshold in EvalJobConfig and Config.
EVIDENCE_VERDICT_SUPPORTED_THRESHOLD: float = 0.75

# Retrieval score at or above which a claim is labelled UNVERIFIABLE
# (insufficient evidence).  Below this it is labelled CONTRADICTED.
EVIDENCE_VERDICT_UNVERIFIABLE_THRESHOLD: float = 0.4

# ── Metrics ──────────────────────────────────────────────────────────────────

# Score at or above which a metric is considered "passed".
# Applied uniformly to hallucination, rubric, and bias metrics.
METRIC_PASS_THRESHOLD: float = 0.5

# Composite score weights — must sum to 1.0.
# Used as defaults in EvalJobConfig.score_weights.
SCORE_WEIGHT_FAITHFULNESS: float = 0.25
SCORE_WEIGHT_ANSWER_RELEVANCY: float = 0.20
SCORE_WEIGHT_HALLUCINATION: float = 0.25
SCORE_WEIGHT_RUBRIC: float = 0.15
SCORE_WEIGHT_SAFETY: float = 0.10
SCORE_WEIGHT_EVIDENCE_SUPPORT_RATE: float = 0.05

# ── LLM ──────────────────────────────────────────────────────────────────────

# Default LiteLLM model used as judge when no override is provided.
DEFAULT_JUDGE_MODEL: str = "gpt-4o-mini"

# Default LLM provider prefix used for routing in LiteLLM.
DEFAULT_LLM_PROVIDER: str = "openai"

# Default sentence-transformer model used for local embeddings.
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Wall-clock timeout in seconds for a single litellm.completion() call.
LLM_CALL_TIMEOUT_SECONDS: int = 60

# ── Storage ───────────────────────────────────────────────────────────────────

# Default local path for the LanceDB vector store.
DEFAULT_LANCEDB_PATH: str = "./.lancedb"

# Default directory for node-level execution cache.
# Override with the LUMISEVAL_CACHE_DIR environment variable.
CACHE_DIR: str = ".lumiseval_cache"

# ── Adversarial Probes ───────────────────────────────────────────────────────

# Default Giskard vulnerability categories probed by the adversarial node.
# Extend or override this list per-job via the job config when needed.
ADVERSARIAL_DEFAULT_PROBE_CATEGORIES: list[str] = [
    "prompt_injection",
    "pii_leakage",
    "jailbreak",
    "stereotype",
]

# ── Job Execution ─────────────────────────────────────────────────────────────

# Maximum number of evaluation jobs that can run concurrently.
DEFAULT_MAX_CONCURRENT_JOBS: int = 4

# ── Dataset Adapter ───────────────────────────────────────────────────────────

# Default dataset name assigned to cases that don't carry an explicit name.
DEFAULT_DATASET_NAME: str = "user_dataset"

# Default dataset split used by adapters when no split is specified.
DEFAULT_SPLIT: str = "train"
