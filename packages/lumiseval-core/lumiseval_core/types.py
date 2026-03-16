"""Shared domain types for lumis-eval."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class RuleCompliance(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


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


class EvidenceResult(BaseModel):
    claim_text: str
    source: EvidenceSource
    passages: list[EvidencePassage] = Field(default_factory=list)
    verdict: ClaimVerdict = ClaimVerdict.UNVERIFIABLE
    no_evidence_found: bool = False
    knowledge_gap: bool = False


class InputMetadata(BaseModel):
    record_count: int
    total_tokens: int
    total_chars: int
    estimated_chunk_count: int
    estimated_claim_count: int
    per_record: list[dict[str, Any]] = Field(default_factory=list)


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


class EvalJobConfig(BaseModel):
    job_id: str
    judge_model: str = "gpt-4o-mini"
    enable_ragas: bool = True
    enable_deepeval: bool = True
    enable_giskard: bool = False
    enable_rubric_eval: bool = False
    web_search: bool = False
    adversarial: bool = False
    evidence_threshold: float = 0.75
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": 0.4,
            "hallucination": 0.3,
            "rubric": 0.2,
            "safety": 0.1,
        }
    )
    budget_cap_usd: Optional[float] = None


class RAGASMetricResult(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    error: Optional[str] = None


class DeepEvalMetricResult(BaseModel):
    hallucination_score: Optional[float] = None
    g_eval_score: Optional[float] = None
    privacy_score: Optional[float] = None
    bias_score: Optional[float] = None
    error: Optional[str] = None


class GiskardVulnerability(BaseModel):
    probe_type: str
    severity: Severity
    description: str
    reproduction_details: str
    false_positive: bool = False


class GiskardScanResult(BaseModel):
    vulnerabilities: list[GiskardVulnerability] = Field(default_factory=list)
    giskard_available: bool = True
    error: Optional[str] = None


class RubricRuleResult(BaseModel):
    rule_id: str
    compliance: RuleCompliance
    reasoning: str
    confidence: float
    violated_claims: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class RubricEvalResult(BaseModel):
    rule_results: list[RubricRuleResult] = Field(default_factory=list)
    compliance_rate: float = 0.0
    composite_adherence_score: float = 0.0


class EvalReport(BaseModel):
    job_id: str
    composite_score: Optional[float] = None
    confidence_band: Optional[float] = None
    claim_verdicts: list[EvidenceResult] = Field(default_factory=list)
    ragas: Optional[RAGASMetricResult] = None
    deepeval: Optional[DeepEvalMetricResult] = None
    giskard: Optional[GiskardScanResult] = None
    rubric: Optional[RubricEvalResult] = None
    cost_estimate: Optional[CostEstimate] = None
    cost_actual_usd: float = 0.0
    evaluation_incomplete: bool = False
    warnings: list[str] = Field(default_factory=list)
