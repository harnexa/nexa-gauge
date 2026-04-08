"""Shared domain types for lumis-eval."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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


_ALLOWED_GEVAL_Item_FIELDS = {"question", "generation", "reference", "context"}




# ------------------------------------------------------------
# New Nodes
# ------------------------------------------------------------


GevalItemField = Literal["question", "generation", "reference", "context"]

class Item(BaseModel):
    id: str = ""
    text: str
    tokens: float
    confidence: float = 1.0
    cached: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

class GevalMetricInput(BaseModel):
    """Input carried by a metric per Item"""
    name: str
    item_fields: list[GevalItemField] = Field(default_factory=lambda: ["generation"])
    criteria: Item | None = None
    evaluation_steps: list[Item] 

class Geval(BaseModel):
    """Input contract carried by Item input payload."""
    metrics: list[GevalMetricInput] = Field(default_factory=list)

class Chunk(BaseModel):
    index: int
    item: Item
    char_start: int
    char_end: int
    sha256: str

class Claim(BaseModel):
    item: Item
    source_chunk_index: Optional[int] = None
    confidence: float = 1.0
    extraction_failed: bool = False


class MetricResult(BaseModel):
    name: str
    category: MetricCategory
    score: float | None = None
    result: list[Any] | None = None
    error: str | None = None


class Faithfulness(Claim):
    verdict: Literal["ACCEPTED", "REJECTED"]


class Relevancy(Claim):
    verdict: Literal["relevant", "irrelevant", "idk"]

class Inputs(BaseModel):
    generation: Item
    question: Optional[Item] = None
    reference: Optional[Item] = None
    context: Optional[Item] = None
    geval: Optional[Geval] = None

    has_generation: bool = False
    has_question: bool = False
    has_reference: bool = False
    has_context: bool = False
    has_geval: bool = False


class CostEstimate(BaseModel):
    cost: float
    input_tokens: Optional[float] = None
    output_tokens: Optional[float] = None

# -----
# NODES Essentials
#
class ChunkArtifacts(BaseModel):
    chunks: list[Chunk]
    cost:  CostEstimate

class ClaimArtifacts(BaseModel):
    claims: list[Claim]
    cost: list[CostEstimate]

class DedupArtifacts(BaseModel):
    items: list[Item]
    dropped: int
    dedup_map: dict[int, int]
    cost: CostEstimate

class GroundingMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

class RelevanceMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

class RedteamMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate


class GevalStepsResolved(BaseModel):
    """This is per metric per Item."""
    key: str
    name: str
    item_fields: list[GevalItemField]
    evaluation_steps: list[Item]
    steps_source: Literal["provided", "generated", "cache_used"]
    signature: str | None = None



class GevalStepsArtifacts(BaseModel):
    resolved_steps: list[GevalStepsResolved] = Field(default_factory=list)
    cost: CostEstimate | None = None


class GevalMetrics(BaseModel):
    metrics: list[MetricResult] = Field(default_factory=list)
    cost: CostEstimate | None = None

class ReferenceMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

class EvalPayload(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate
