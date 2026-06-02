"""Shared domain types for nexa-gauge."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator

ExecutionMode = Literal["run", "estimate"]


class EvidenceSource(str, Enum):
    """Where evidence for a claim was retrieved from."""

    LOCAL = "local"
    MCP = "mcp"
    WEB = "web"
    NONE = "none"


class MetricCategory(str, Enum):
    """High-level grouping used by reports to organise metric results."""

    RETRIEVAL = "retrieval"  # was evidence retrieval correct/complete?
    ANSWER = "output|generation|answer"  # is the answer correct, relevant, and safe?


# ------------------------------------------------------------
# New Nodes
# ------------------------------------------------------------
GevalItemField = Literal["input", "output", "reference", "context"]
RedteamItemField = Literal["input", "output", "reference", "context"]


class ScoringMode(str, Enum):
    """Shared scoring modes for all LLM-as-a-judge metric nodes.

    Used by geval, grounding, relevance, and redteam to choose between
    a 1-5 Likert integer scale and a binary yes/no decision.
    """

    SCALE_1_5 = "scale_1_5"
    BINARY_YES_NO = "binary_yes_no"


class GevalMetricSpec(BaseModel):
    """Legacy GEval metric shape used by cache/tests/adapters."""

    name: str
    item_fields: list[GevalItemField] = Field(default_factory=lambda: ["output"])
    criteria: str | None = None
    evaluation_steps: list[str] = Field(default_factory=list)


class GevalMetricInput(BaseModel):
    """One GEval metric as the scanner emits it onto an ``Inputs`` payload.

    ``evaluation_steps`` may be empty — in that case ``GevalStepsNode``
    generates them from ``criteria`` and caches by signature.

    Scoring knobs (``scoring_mode``, ``include_reasoning``) live on the
    parent :class:`Geval` config, not per-metric, so a node block configures
    all its metrics uniformly.
    """

    name: str
    item_fields: list[GevalItemField] = Field(default_factory=lambda: ["output"])
    criteria: Item | None = None
    evaluation_steps: list[Item]


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


class Chunk(BaseModel):
    """One span of the output after chunking, with offsets back to source."""

    index: int
    item: Item
    char_start: int
    char_end: int
    sha256: str


# --------------------------------------------------------------------------------
# RAGAS
# --------------------------------------------------------------------------------


class Claim(BaseModel):
    """Atomic factual claim extracted from a chunk; feeds grounding/relevance."""

    item: Item
    source_chunk_index: Optional[int] = None
    confidence: float = 1.0
    extraction_failed: bool = False


class GroundingClaim(Claim):
    """Claim annotated with a Grounding verdict against retrieved evidence."""

    verdict: Literal["ACCEPTED", "REJECTED", "PASSED", "FAILED"]
    raw_score: Optional[Any] = None
    """Raw integer the judge emitted for this claim (1-5 likert, or 0/1 binary)."""


class RelevancyClaim(Claim):
    """Claim annotated with a relevancy verdict against the input."""

    verdict: Literal["ACCEPTED", "REJECTED", "PASSED", "FAILED"]
    raw_score: Optional[Any] = None
    """Raw integer the judge emitted for this claim (1-5 likert, or 0/1 binary)."""


class RedTeamVerdict(str, Enum):
    """Canonical redteam safety verdict.

    Values are stored in uppercase for consistency in reports and downstream
    consumers. Use ``parse`` to normalize model output that may come as
    lowercase/mixed-case text.
    """

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"

    @classmethod
    def parse(cls, value: "RedTeamVerdict | str") -> "RedTeamVerdict":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Invalid RedTeamVerdict '{value}'. Expected one of: SAFE, UNSAFE."
            ) from exc


class MetricResult(BaseModel):
    """One row in a metric node output.

    ``result`` stores raw metric-native artifacts (claim verdicts, rubric details,
    judge reasoning, etc.). ``verdict`` stores a high-level outcome suitable for
    aggregation/reporting; ``None`` means no pass/fail judgment is available.
    """

    name: str
    category: MetricCategory
    score: float | None = None
    verdict: str | None = None
    result: list[Any] | None = None
    error: str | None = None


class Grounding(BaseModel):
    """Per-case grounding node config. Carries only the shared scoring knobs.

    Grounding eligibility itself is gated by ``Inputs.has_context``; this
    block only tunes the judge's scoring mode and reasoning output.
    """

    scoring_mode: ScoringMode = ScoringMode.BINARY_YES_NO
    include_reasoning: bool = False


class Relevance(BaseModel):
    """Per-case relevance node config. Carries only the shared scoring knobs.

    Relevance eligibility itself is gated by ``Inputs.has_input``; this
    block only tunes the judge's scoring mode and reasoning output.
    """

    scoring_mode: ScoringMode = ScoringMode.BINARY_YES_NO
    include_reasoning: bool = False


# --------------------------------------------------------------------------------
# Geval
# --------------------------------------------------------------------------------
class Geval(BaseModel):
    """Per-case GEval config carried on the ``Inputs`` payload.

    The ``scoring_mode`` and ``include_reasoning`` knobs apply uniformly to
    every metric in this block. Defaults are the strict / cheap pair
    (``binary_yes_no`` + reasoning off) so omitting them in a record yields
    the cheapest possible run.
    """

    metrics: list[GevalMetricInput] = Field(default_factory=list)
    scoring_mode: ScoringMode = ScoringMode.BINARY_YES_NO
    include_reasoning: bool = False


class RedteamRubric(BaseModel):
    """Structured rubric for one redteam metric."""

    goal: str
    violations: list[str]
    non_violations: list[str] = Field(default_factory=list)


class RedteamMetricInput(BaseModel):
    """Input carried by one redteam metric per Item."""

    name: str
    rubric: RedteamRubric
    item_fields: list[RedteamItemField] = Field(default_factory=lambda: ["output"])


class Redteam(BaseModel):
    """Per-case redteam config (default safety metrics + user-supplied rubrics).

    The ``scoring_mode`` and ``include_reasoning`` knobs apply uniformly to
    every sub-metric (bias, toxicity, custom rubrics).
    """

    metrics: list[RedteamMetricInput] = Field(default_factory=list)
    scoring_mode: ScoringMode = ScoringMode.BINARY_YES_NO
    include_reasoning: bool = False


class Refalign(BaseModel):
    """Per-case semantic reference-similarity config."""

    atomic_chunks: bool = False
    similarity_threshold: float = 0.6
    refine_top_k: int | None = None


class Inputs(BaseModel):
    """Per-case input bundle built by the scanner node.

    ``has_*`` flags are derived from content presence in the ``@model_validator``
    below; node eligibility rules (see ``topology.NodeSpec``) read them to
    decide whether to run or skip.
    """

    case_id: str
    output: Item
    input: Optional[Item] = None
    reference: Optional[Item] = None
    context: Optional[Item] = None

    # Metric Nodes
    geval: Optional[Geval] = None
    grounding: Optional[Grounding] = None
    relevance: Optional[Relevance] = None
    redteam: Optional[Redteam] = None
    refalign: Optional[Refalign] = None

    has_output: bool = False
    has_input: bool = False
    has_reference: bool = False
    has_context: bool = False
    has_geval: bool = False
    has_redteam: bool = False

    @model_validator(mode="after")
    def _set_has_flags(self) -> "Inputs":
        self.has_output = bool(self.output and self.output.text)
        self.has_input = bool(self.input and self.input.text.strip())
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
    """Output of the ``chunk`` node: split output + cost."""

    chunks: list[Chunk]
    cost: CostEstimate


class ClaimArtifacts(BaseModel):
    """Output of the ``claims`` node: extracted claims + cost."""

    claims: list[Claim]
    cost: CostEstimate


class RefinerArtifacts(BaseModel):
    """Raw refiner output; ``dedup_map`` records source→kept index relations."""

    items: list[Item]
    indices: list[int]
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


class RefmatchMetrics(BaseModel):
    """Output of the ``refmatch`` metric node."""

    metrics: list[MetricResult]
    cost: CostEstimate


class RefalignMetrics(BaseModel):
    """Output of the ``refalign`` metric node."""

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
    output_chunk: Optional[ChunkArtifacts]
    output_refined_chunks: Optional[ChunkArtifacts]
    reference_chunk: Optional[ChunkArtifacts]
    reference_refined_chunks: Optional[ChunkArtifacts]
    output_claims: Optional[ClaimArtifacts]
    grounding_metrics: Optional[GroundingMetrics]
    relevance_metrics: Optional[RelevanceMetrics]
    redteam_metrics: Optional[RedteamMetrics]
    geval_steps: Optional[GevalStepsArtifacts]
    geval_metrics: Optional[GevalMetrics]
    refmatch_metrics: Optional[RefmatchMetrics]
    refalign_metrics: Optional[RefalignMetrics]
    eval_summary: Optional[dict[str, Any]]
    report: Optional[dict[str, Any]]
    cost_estimate: Optional[CostEstimate]
    node_model_usage: dict[str, dict[str, Any]]

    # Per-run utility-node strategy config
    chunker: str
    refiner: str
    refiner_top_k: int
