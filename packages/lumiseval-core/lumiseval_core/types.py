"""Shared domain types for lumis-eval."""

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from lumiseval_core.constants import (
    DEFAULT_DATASET_NAME,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SPLIT,
    EVIDENCE_VERDICT_SUPPORTED_THRESHOLD,
    SCORE_WEIGHT_ANSWER_RELEVANCY,
    SCORE_WEIGHT_EVIDENCE_SUPPORT_RATE,
    SCORE_WEIGHT_FAITHFULNESS,
    SCORE_WEIGHT_HALLUCINATION,
    SCORE_WEIGHT_RUBRIC,
    SCORE_WEIGHT_SAFETY,
)

# ── Enums ──────────────────────────────────────────────────────────────────


class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class EvidenceSource(str, Enum):
    LOCAL = "local"
    MCP = "mcp"
    WEB = "web"
    NONE = "none"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MetricCategory(str, Enum):
    RETRIEVAL = "retrieval"  # was evidence retrieval correct/complete?
    ANSWER = "answer"  # is the answer correct, relevant, and safe?


# ── Core domain models ─────────────────────────────────────────────────────


class Chunk(BaseModel):
    index: int
    text: str
    char_start: int
    char_end: int
    sha256: str


class Claim(BaseModel):
    text: str
    source_chunk_index: int
    confidence: float = 1.0
    extraction_failed: bool = False


class RubricRule(BaseModel):
    id: str
    statement: str
    pass_condition: str
    low_confidence_extraction: bool = False


class EvidencePassage(BaseModel):
    text: str
    source_doc_id: str
    retrieval_score: float
    source: EvidenceSource


class InputMetadata(BaseModel):
    record_count: int
    total_tokens: int
    total_chars: int
    estimated_chunk_count: int
    estimated_claim_count: int
    per_record: list[dict[str, Any]] = Field(default_factory=list)
    eligible_record_count: dict[str, int] = Field(default_factory=dict)
    eligible_chunk_count: dict[str, int] = Field(default_factory=dict)
    eligible_claim_count: dict[str, int] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    estimated_judge_calls: int
    estimated_embedding_calls: int
    estimated_tavily_calls: int
    judge_cost_usd: float
    embedding_cost_usd: float
    tavily_cost_usd: float
    total_estimated_usd: float
    low_usd: float
    high_usd: float
    approximate: bool = False
    approximate_warning: Optional[str] = None


# ── Unified metric result types ─────────────────────────────────────────────


class Faithfulness(Claim):
    """Per-claim verdict produced by the faithfulness evaluator.

    Inherits all Claim fields (text, source_chunk_index, confidence,
    extraction_failed) and adds the faithfulness verdict.
    """

    verdict: str  # "ACCEPTED" or "REJECTED"


class Relevancy(Claim):
    verdict: str


class MetricResult(BaseModel):
    """Result for a single evaluation metric. score is always 0.0–1.0 where 1.0 is best."""

    name: str
    category: MetricCategory
    score: Optional[float] = None
    passed: Optional[bool] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    result: list[Union[Faithfulness, Relevancy]] = Field(default_factory=list)


class QualityScore(BaseModel):
    """Composite score for one evaluation dimension (retrieval or answer)."""

    score: Optional[float] = None
    metrics: list[MetricResult] = Field(default_factory=list)


class EvalCase(BaseModel):
    """Canonical dataset row used by adapters and dataset runners."""

    case_id: str
    generation: str
    dataset: str = DEFAULT_DATASET_NAME
    split: str = DEFAULT_SPLIT
    question: Optional[str] = None
    ground_truth: Optional[str] = None
    context: list[str] = Field(default_factory=list)
    reference_files: list[str] = Field(default_factory=list)
    rubric_rules: list[RubricRule] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Configuration ───────────────────────────────────────────────────────────


class EvalJobConfig(BaseModel):
    job_id: str
    judge_model: str = DEFAULT_JUDGE_MODEL
    enable_hallucination: bool = True
    enable_faithfulness: bool = True
    enable_answer_relevancy: bool = True
    enable_adversarial: bool = False
    enable_rubric: bool = False
    web_search: bool = False
    evidence_threshold: float = EVIDENCE_VERDICT_SUPPORTED_THRESHOLD
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": SCORE_WEIGHT_FAITHFULNESS,
            "answer_relevancy": SCORE_WEIGHT_ANSWER_RELEVANCY,
            "hallucination": SCORE_WEIGHT_HALLUCINATION,
            "rubric": SCORE_WEIGHT_RUBRIC,
            "safety": SCORE_WEIGHT_SAFETY,
            "evidence_support_rate": SCORE_WEIGHT_EVIDENCE_SUPPORT_RATE,
        }
    )
    budget_cap_usd: Optional[float] = None


# ── Report ──────────────────────────────────────────────────────────────────


class EvalReport(BaseModel):
    job_id: str
    composite_score: Optional[float] = None
    confidence_band: Optional[float] = None
    retrieval_score: Optional[QualityScore] = None
    answer_score: Optional[QualityScore] = None
    cost_estimate: Optional[CostEstimate] = None
    cost_actual_usd: float = 0.0
    evaluation_incomplete: bool = False
    warnings: list[str] = Field(default_factory=list)
