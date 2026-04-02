"""Shared domain types for lumis-eval."""

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from lumiseval_core.constants import (
    AVG_CLAIM_TOKENS,
    AVG_DEEPEVAL_INPUT_OVERHEAD_TOKENS,
    AVG_DEEPEVAL_OUTPUT_OVERHEAD_TOKENS,
    AVG_GEVAL_INPUT_OVERHEAD_TOKENS,
    AVG_GEVAL_OUTPUT_OVERHEAD_TOKENS,
    AVG_OUTPUT_TOKENS_BOOLEAN_VERDICT,
    AVG_OUTPUT_TOKENS_JSON_VERDICT,
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


class Rubric(BaseModel):
    id: str
    statement: str
    pass_condition: str
    low_confidence_extraction: bool = False


class EvidencePassage(BaseModel):
    text: str
    source_doc_id: str
    retrieval_score: float
    source: EvidenceSource


class EvidenceResult(BaseModel):
    """Evidence retrieved for a single claim, including verdict and source passages."""

    claim_text: str
    source: EvidenceSource
    passages: list[EvidencePassage] = Field(default_factory=list)
    verdict: ClaimVerdict = ClaimVerdict.UNVERIFIABLE
    no_evidence_found: bool = False


class RecordMeta(BaseModel):
    context_token_count: int
    generation_chunk_count: int
    generation_token_count: int
    rubric_token_count: int
    total_token_count: int
    estimated_claim_count: int
    has_context: bool
    has_rubric: bool
    has_reference: bool = False
    eligible_nodes: list[str]


class Record(BaseModel):
    case_id: str
    record_index: int
    question: Optional[str]
    context: Optional[list[str]]
    generation: Optional[str]
    generation_chunks: Optional[list[str]]
    rubric: Optional[list[str]]
    record_metadata: RecordMeta


class ClaimCostMeta(BaseModel):
    eligible_records: int
    avg_generation_chunks: float
    avg_generation_tokens: float
    avg_claim_tokens: float = AVG_CLAIM_TOKENS
    avg_output_token: float = AVG_OUTPUT_TOKENS_JSON_VERDICT


class GorundingCostMeta(BaseModel):
    eligible_records: int
    avg_claims_per_record: float
    avg_context_tokens: float
    avg_claim_tokens: float = AVG_CLAIM_TOKENS
    avg_output_token: float = AVG_OUTPUT_TOKENS_BOOLEAN_VERDICT


class RelevanceCostMeta(BaseModel):
    eligible_records: int
    avg_claims_per_record: float
    avg_question_tokens: float
    avg_claim_tokens: float = AVG_CLAIM_TOKENS
    avg_output_token: float = AVG_OUTPUT_TOKENS_JSON_VERDICT


class RubricCostMeta(BaseModel):
    eligible_records: int
    rule_count: int
    unique_rule_count: int
    rule_tokens: float
    unique_rule_tokens: float
    avg_input_tokens: float = AVG_GEVAL_INPUT_OVERHEAD_TOKENS
    avg_output_tokens: float = AVG_GEVAL_OUTPUT_OVERHEAD_TOKENS


class RedTeamCostMeta(BaseModel):
    eligible_records: int
    avg_input_tokens: float = AVG_DEEPEVAL_INPUT_OVERHEAD_TOKENS
    avg_output_tokens: float = AVG_DEEPEVAL_OUTPUT_OVERHEAD_TOKENS


class ReferenceCostMeta(BaseModel):
    """Cost metadata for generation metrics (ROUGE/BLEU/METEOR). No LLM calls — always $0."""

    eligible_records: int


class CostMetadata(BaseModel):
    grounding: GorundingCostMeta
    relevance: RelevanceCostMeta
    rubric: RubricCostMeta
    readteam: RedTeamCostMeta
    reference: ReferenceCostMeta


class InputMetadata(BaseModel):
    record_count: int
    total_tokens: int  # generation_tokens + context_tokens + rubric_tokens

    generation_tokens: int = 0  # tokens across all generation fields
    context_tokens: int = 0  # tokens across all context passages
    rubric_tokens: int = 0  # tokens across all rubric rule statements
    unique_rubric_tokens: int = 0  # tokens across all unique rubric rules

    rubric_rule_count: int = 0  # Total counts of rubrics
    unique_rubric_rule_count: int = 0  # Total counts of unique rubrics
    generation_chunk_count: int = 0  # chunks produced by chunking generation text

    cost_meta: CostMetadata
    # Each Record and its respective metadata
    records: list[Record] = Field(default_factory=list)


class NodeCostBreakdown(BaseModel):
    model_calls: int = 0
    cost_usd: float = 0.0


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
    reference: Optional[str] = None
    context: list[str] = Field(default_factory=list)
    reference_files: list[str] = Field(default_factory=list)
    rubric: list[Rubric] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Configuration ───────────────────────────────────────────────────────────


class EvalJobConfig(BaseModel):
    job_id: str
    judge_model: str = DEFAULT_JUDGE_MODEL
    enable_grounding: bool = True
    enable_relevance: bool = True
    enable_redteam: bool = False
    enable_rubric: bool = False
    enable_reference: bool = True
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


# ── Cost estimate ────────────────────────────────────────────────────────────


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
    node_breakdown: dict[str, NodeCostBreakdown] = Field(default_factory=dict)


# ── Report ───────────────────────────────────────────────────────────────────


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
