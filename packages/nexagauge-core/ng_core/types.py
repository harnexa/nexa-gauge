"""Shared domain types for nexa-gauge."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator

ExecutionMode = Literal["run", "estimate"]

# ── Enums ──────────────────────────────────────────────────────────────────


class ClaimVerdict(str, Enum):
    """Verdict from the grounding/faithfulness judge for a single claim."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class EvidenceSource(str, Enum):
    """Where evidence for a claim was retrieved from."""

    LOCAL = "local"
    MCP = "mcp"
    WEB = "web"
    NONE = "none"


class MetricCategory(str, Enum):
    """High-level grouping used by reports to organise metric results."""

    RETRIEVAL = "retrieval"  # was evidence retrieval correct/complete?
    ANSWER = "answer"  # is the answer correct, relevant, and safe?


# ------------------------------------------------------------
# New Nodes
# ------------------------------------------------------------
GevalItemField = Literal["question", "generation", "reference", "context"]
RedteamItemField = Literal["question", "generation", "reference", "context"]


class GevalMetricSpec(BaseModel):
    """Legacy GEval metric shape used by cache/tests/adapters."""

    name: str
    item_fields: list[GevalItemField] = Field(default_factory=lambda: ["generation"])
    criteria: str | None = None
    evaluation_steps: list[str] = Field(default_factory=list)


class GevalConfig(BaseModel):
    """Legacy GEval config shape used by cache/tests/adapters."""

    metrics: list[GevalMetricSpec] = Field(default_factory=list)


class Item(BaseModel):
    """Smallest unit of text in the pipeline (a chunk, claim, step, …).

    ``id`` defaults to a short SHA-256 of ``text`` so equal text yields a
    stable identity across runs without the caller having to mint one.
    """

    id: str = ""
    text: str
    tokens: float
    confidence: float = 1.0
    cached: bool = False

    def model_post_init(self, __context: Any) -> None:
        del __context
        if not self.id:
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


class GevalMetricInput(BaseModel):
    """One GEval metric as the scanner emits it onto an ``Inputs`` payload.

    ``evaluation_steps`` may be empty — in that case ``GevalStepsNode``
    generates them from ``criteria`` and caches by signature.
    """

    name: str
    item_fields: list[GevalItemField] = Field(default_factory=lambda: ["generation"])
    criteria: Item | None = None
    evaluation_steps: list[Item]


class Geval(BaseModel):
    """Input contract carried by Item input payload."""

    metrics: list[GevalMetricInput] = Field(default_factory=list)


class RedteamRubric(BaseModel):
    """Structured rubric for one redteam metric."""

    goal: str
    violations: list[str]
    non_violations: list[str] = Field(default_factory=list)


class RedteamMetricInput(BaseModel):
    """Input carried by one redteam metric per Item."""

    name: str
    rubric: RedteamRubric
    item_fields: list[RedteamItemField] = Field(default_factory=lambda: ["generation"])


class Redteam(BaseModel):
    """Input contract for redteam metric configuration."""

    metrics: list[RedteamMetricInput] = Field(default_factory=list)


class Chunk(BaseModel):
    """One span of the generation after chunking, with offsets back to source."""

    index: int
    item: Item
    char_start: int
    char_end: int
    sha256: str


class Claim(BaseModel):
    """Atomic factual claim extracted from a chunk; feeds grounding/relevance."""

    item: Item
    source_chunk_index: Optional[int] = None
    confidence: float = 1.0
    extraction_failed: bool = False


class MetricResult(BaseModel):
    """One row in a metric node's output — score plus optional per-item detail."""

    name: str
    category: MetricCategory
    score: float | None = None
    result: list[Any] | None = None
    error: str | None = None


class Faithfulness(Claim):
    """Claim annotated with a faithfulness verdict against retrieved evidence."""

    verdict: Literal["ACCEPTED", "REJECTED"]


class Relevancy(Claim):
    """Claim annotated with a relevancy verdict against the question."""

    verdict: Literal["ACCEPTED", "REJECTED"]


class Inputs(BaseModel):
    """Per-case input bundle built by the scanner node.

    ``has_*`` flags are derived from content presence in the ``@model_validator``
    below; node eligibility rules (see ``topology.NodeSpec``) read them to
    decide whether to run or skip.
    """

    case_id: str
    generation: Item
    question: Optional[Item] = None
    reference: Optional[Item] = None
    context: Optional[Item] = None
    geval: Optional[Geval] = None
    redteam: Optional[Redteam] = None

    has_generation: bool = False
    has_question: bool = False
    has_reference: bool = False
    has_context: bool = False
    has_geval: bool = False
    has_redteam: bool = False

    @model_validator(mode="after")
    def _set_has_flags(self) -> "Inputs":
        self.has_generation = bool(self.generation and self.generation.text)
        self.has_question = bool(self.question and self.question.text.strip())
        self.has_reference = bool(self.reference and self.reference.text.strip())
        self.has_context = bool(self.context and self.context.text.strip())
        self.has_geval = bool(self.geval and self.geval.metrics)
        self.has_redteam = bool(self.redteam and self.redteam.metrics)
        return self


class CostEstimate(BaseModel):
    """USD cost + token counts attached to every node artifact."""

    cost: float
    input_tokens: Optional[float] = None
    output_tokens: Optional[float] = None


# -----
# NODES Essentials
#
class ChunkArtifacts(BaseModel):
    """Output of the ``chunk`` node: split generation + cost."""

    chunks: list[Chunk]
    cost: CostEstimate


class ClaimArtifacts(BaseModel):
    """Output of the ``claims`` / ``dedup`` nodes: extracted claims + cost."""

    claims: list[Claim]
    cost: CostEstimate


class DedupArtifacts(BaseModel):
    """Raw dedup output; ``dedup_map`` records source→kept index relations."""

    items: list[Item]
    dropped: int
    dedup_map: dict[int, int]
    cost: CostEstimate


class GroundingMetrics(BaseModel):
    """Output of the ``grounding`` metric node."""

    metrics: list[MetricResult]
    cost: CostEstimate


class RelevanceMetrics(BaseModel):
    """Output of the ``relevance`` metric node."""

    metrics: list[MetricResult]
    cost: CostEstimate


class RedteamMetrics(BaseModel):
    """Output of the ``redteam`` metric node."""

    metrics: list[MetricResult]
    cost: CostEstimate


class GevalStepsResolved(BaseModel):
    """Resolved evaluation steps for one GEval metric on one case.

    ``steps_source`` tells reports whether the steps were provided by the
    caller, freshly generated, or served from the artifact cache.
    ``signature`` is populated whenever the steps came from (or were written
    into) the cache.
    """

    key: str
    name: str
    item_fields: list[GevalItemField]
    evaluation_steps: list[Item]
    steps_source: Literal["provided", "generated", "cache_used"]
    signature: str | None = None


class GevalStepsArtifacts(BaseModel):
    """Output of the ``geval_steps`` node: one resolved entry per metric."""

    resolved_steps: list[GevalStepsResolved] = Field(default_factory=list)
    cost: CostEstimate | None = None


class GevalCacheArtifact(BaseModel):
    """Persisted artifact for one GEval-step signature.

    Cached independently of per-case state: keyed on
    (model, prompt_version, parser_version, item_fields, criteria) so N cases
    sharing a criterion reuse one LLM call.
    """

    signature: str
    model: str
    prompt_version: str
    parser_version: str
    item_fields: list[str]
    criteria: Item
    evaluation_steps: list[Item]
    created_at: str


class GevalMetrics(BaseModel):
    """Output of the ``geval`` metric node."""

    metrics: list[MetricResult] = Field(default_factory=list)
    cost: CostEstimate | None = None


class ReferenceMetrics(BaseModel):
    """Output of the ``reference`` metric node."""

    metrics: list[MetricResult]
    cost: CostEstimate


class EvalPayload(BaseModel):
    """Output of the ``eval`` join node: flattened metrics + aggregate cost."""

    metrics: list[MetricResult]
    cost: CostEstimate


class EvalReport(BaseModel):
    """Compatibility report payload used by cache/API surfaces."""

    metrics: list[MetricResult | dict[str, Any]] = Field(default_factory=list)
    cost_estimate: CostEstimate | None = None
    cost_actual_usd: float = 0.0


# ── Graph state ────────────────────────────────────────────────────────────
class EvalCase(TypedDict):
    """Canonical dataset row used by adapters and dataset runners."""

    # Required
    record: dict[str, str]
    llm_overrides: Optional[Mapping[str, Any]]
    target_node: str
    execution_mode: ExecutionMode
    estimated_costs: dict[str, CostEstimate]
    reference_files: list[Path] = []

    # Pipeline inputs
    inputs: Optional[Inputs]

    # Pipeline artifacts — None until the corresponding node runs
    generation_chunk: Optional[ChunkArtifacts]
    generation_claims: Optional[ClaimArtifacts]
    generation_dedup_claims: Optional[ClaimArtifacts]
    grounding_metrics: Optional[GroundingMetrics]
    relevance_metrics: Optional[RelevanceMetrics]
    redteam_metrics: Optional[RedteamMetrics]
    geval_steps: Optional[GevalStepsArtifacts]
    geval_metrics: Optional[GevalMetrics]
    reference_metrics: Optional[ReferenceMetrics]
    node_model_usage: dict[str, dict[str, Any]]
