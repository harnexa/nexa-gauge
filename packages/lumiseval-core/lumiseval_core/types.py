"""Shared domain types for lumis-eval."""

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    GENERATION_CHUNK_SIZE_TOKENS,
    SCORE_WEIGHT_ANSWER_RELEVANCY,
    SCORE_WEIGHT_EVIDENCE_SUPPORT_RATE,
    SCORE_WEIGHT_FAITHFULNESS,
    SCORE_WEIGHT_GEVAL,
    SCORE_WEIGHT_HALLUCINATION,
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


_ALLOWED_GEVAL_RECORD_FIELDS = {"question", "generation", "reference", "context"}


# ──────────────────────────────────────────────────────────────────────────────────
# GEVAL Types
# ──────────────────────────────────────────────────────────────────────────────────
class GevalMetricSpec(BaseModel):
    """Public GEval metric contract for one custom judge metric.

    # TODO: handle the case where
    "evaluation_steps": [
        "Check whether the facts in `reference` contradicts any facts in `generation`"
    ] make sure to change reference to GEVAL.EXPECTED_OUTPUT, and generation GEVAL.ACTUAL_OUTPUT
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    record_fields: list[str] = Field(default_factory=list)
    criteria: Optional[str] = None
    evaluation_steps: Optional[list[str]] = None

    @model_validator(mode="after")
    def _validate_one_of_and_fields(self) -> "GevalMetricSpec":
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("GEval metric 'name' must be non-empty.")

        has_criteria = bool(self.criteria and self.criteria.strip())
        has_steps = bool(self.evaluation_steps)
        if has_criteria == has_steps:
            raise ValueError("Exactly one of 'criteria' or 'evaluation_steps' must be provided.")

        if self.criteria is not None:
            self.criteria = self.criteria.strip()

        if self.evaluation_steps is not None:
            cleaned_steps = [step.strip() for step in self.evaluation_steps if step and step.strip()]
            if not cleaned_steps:
                raise ValueError("'evaluation_steps' must contain at least one non-empty step.")
            self.evaluation_steps = cleaned_steps

        normalized_fields: list[str] = []
        seen: set[str] = set()
        for field_name in self.record_fields:
            if field_name not in _ALLOWED_GEVAL_RECORD_FIELDS:
                allowed = ", ".join(sorted(_ALLOWED_GEVAL_RECORD_FIELDS))
                raise ValueError(
                    f"Unknown GEval record field '{field_name}'. Allowed: {allowed}."
                )
            if field_name in seen:
                continue
            normalized_fields.append(field_name)
            seen.add(field_name)

        # GEval scoring always needs ACTUAL_OUTPUT; auto-include generation.
        if "generation" not in seen:
            normalized_fields.append("generation")

        self.record_fields = normalized_fields
        return self


class GevalConfig(BaseModel):
    """Top-level GEval payload attached to one EvalCase."""
    model_config = ConfigDict(extra="forbid")
    metrics: list[GevalMetricSpec] = Field(default_factory=list)


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
    generation_chunk_count: int
    context_token_count: int
    question_token_count: int
    generation_token_count: int
    geval_token_count: int
    total_token_count: int
    estimated_claim_count: int
    has_question: bool
    has_context: bool
    has_geval: bool = False
    has_reference: bool = False
    eligible_nodes: list[str]


class RecordGeval(BaseModel):
    """Record-level GEval projection used by scanner/debug surfaces.

    Notes:
        - Keys are deterministic per-record metric keys (for example ``factuality`` or
          ``factuality#2`` when duplicate metric names exist).
        - ``criteria`` and ``evaluation_steps`` are both present for every key.
          Criteria-only metrics use empty steps; steps-only metrics use ``None``
          criteria.
    """

    criteria: dict[str, str | None] = Field(default_factory=dict)
    evaluation_steps: dict[str, list[str]] = Field(default_factory=dict)


class Record(BaseModel):
    case_id: str
    record_index: int
    question: Optional[str]
    context: Optional[list[str]]
    generation: Optional[str]
    geval: Optional[RecordGeval] = None
    generation_chunks: Optional[list[str]]
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

class GevalStepsCostMeta(BaseModel):
    eligible_records: int
    criteria_count: int
    unique_criteria_count: int
    criteria_tokens: float
    unique_criteria_tokens: float
    avg_output_tokens: float = AVG_GEVAL_OUTPUT_OVERHEAD_TOKENS


class GevalCostMeta(BaseModel):
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
    claim: ClaimCostMeta
    grounding: GorundingCostMeta
    relevance: RelevanceCostMeta
    geval_steps: GevalStepsCostMeta
    geval: GevalCostMeta
    readteam: RedTeamCostMeta
    reference: ReferenceCostMeta


class InputMetadata(BaseModel):
    record_count: int
    total_tokens: int  # question_tokens + generation_tokens + context_tokens + geval_tokens

    question_tokens: int = 0  # tokens across all question fields
    generation_tokens: int = 0  # tokens across all generation fields
    context_tokens: int = 0  # tokens across all context passages
    geval_tokens: int = 0  # tokens across all GEval instructions (criteria/evaluation_steps)
    unique_geval_tokens: int = 0  # tokens across all unique GEval instructions

    geval_metric_count: int = 0  # total GEval metric count across records
    unique_geval_metric_count: int = 0  # total unique GEval metric signatures
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
    geval: Optional[GevalConfig] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Configuration ───────────────────────────────────────────────────────────


class EvalJobConfig(BaseModel):
    job_id: str
    judge_model: str = DEFAULT_JUDGE_MODEL
    enable_grounding: bool = True
    enable_relevance: bool = True
    enable_redteam: bool = False
    enable_geval: bool = False
    enable_reference: bool = True
    web_search: bool = False
    evidence_threshold: float = EVIDENCE_VERDICT_SUPPORTED_THRESHOLD
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": SCORE_WEIGHT_FAITHFULNESS,
            "answer_relevancy": SCORE_WEIGHT_ANSWER_RELEVANCY,
            "hallucination": SCORE_WEIGHT_HALLUCINATION,
            "geval": SCORE_WEIGHT_GEVAL,
            "safety": SCORE_WEIGHT_SAFETY,
            "evidence_support_rate": SCORE_WEIGHT_EVIDENCE_SUPPORT_RATE,
        }
    )
    chunk_size: int = GENERATION_CHUNK_SIZE_TOKENS
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
