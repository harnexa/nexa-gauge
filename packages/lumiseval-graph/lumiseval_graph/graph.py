"""
LangGraph Orchestration Graph — the core evaluation pipeline.

Node sequence:
  scan → chunk → claims → dedup →
  geval_steps →
  [parallel: relevance, grounding, redteam, geval, reference] →
  eval → report → result

TODO:
  - Implement async TaskIQ dispatch for batch jobs.
  - Stream progress to CLI/API consumers.
  - Persist EvalReport to SQLite via SQLModel.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph
from lumiseval_core.config import config as cfg
from lumiseval_core.constants import DEFAULT_JUDGE_MODEL, GENERATION_CHUNK_SIZE_TOKENS
from lumiseval_core.types import (
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    GevalMetrics,
    GevalStepsArtifacts,
    GroundingMetrics,
    Inputs,
    ReferenceMetrics,
    RedteamMetrics,
    RelevanceMetrics,
)
from lumiseval_core.utils import pprint_model

from .llm import get_judge_model
from .log import get_node_logger, print_pipeline_footer, print_pipeline_header
from .nodes import claim_extractor, report
from .nodes.chunk_extractor import ChunkExtractorNode
from .nodes.dedup import DedupNode
from .nodes.metrics.geval import GevalNode, GevalStepsNode
from .nodes.metrics.grounding import GroundingNode
from .nodes.metrics.redteam import RedteamNode
from .nodes.metrics.reference import ReferenceNode
from .nodes.metrics.relevance import RelevanceNode
from .observability import observe, score_trace, update_trace
from .scanner import scan as scan_record

logger = logging.getLogger(__name__)

# Module-level node loggers — one per pipeline node
_log_scanner = get_node_logger("scan")
_log_chunker = get_node_logger("chunk")
_log_claims = get_node_logger("claims")
_log_mmr = get_node_logger("dedup")
_log_ragas = get_node_logger("relevance")
_log_hallucination = get_node_logger("grounding")
_log_adversarial = get_node_logger("redteam")
_log_geval_steps = get_node_logger("geval_steps")
_log_geval = get_node_logger("geval")
_log_reference = get_node_logger("reference")
_log_agg = get_node_logger("eval")
_log_report = get_node_logger("report")


# ── Graph state ────────────────────────────────────────────────────────────
class EvalCase(TypedDict):
    """Canonical dataset row used by adapters and dataset runners."""
    # Required
    record: dict[str, str]
    llm_overrides: Optional[Mapping[str, Any]]

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


@observe(name="node_scan")
def node_metadata_scanner(state: dict[str, Any]) -> dict[str, Any]:
    """
    In LangGraph, each node gets the current full state snapshot.
    So after node_metadata_scanner returns {"inputs": ...}, that patch is merged into state, and downstream nodes like node_chunk receive the same full state with
    inputs included.

    Important in your current file:

    - node_chunk still reads state["generation"], not state["inputs"].
    - So inputs is available downstream, but node_chunk won't use it unless you change it.
    """
    # Per-record scan only: hydrate `inputs` from the current state payload.    
    records = state["record"]

    raw_record = {
        "case_id": records.get("case_id") or "record-0",
        "generation": records.get("generation"),
        "question": records.get("question"),
        "reference": records.get("reference"),
        "context": records.get("context"),
        "geval": records.get("geval"),
        "reference_files": state.get("reference_files") or [],
    }

    case = scan_record(raw_record, idx=0)

    return {"inputs": case.get("inputs")}


@observe(name="node_generation_chunk")
def node_generation_chunk(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs or not inputs.generation:
        return {"generation_chunk": None}
    chunks: ChunkArtifacts = ChunkExtractorNode(
        chunk_size=GENERATION_CHUNK_SIZE_TOKENS
    ).run(item=inputs.generation)

    return {"generation_chunk": chunks}


@observe(name="node_generation_claims")
def node_generation_claims(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    generation_chunk = state.get("generation_chunk")
    if not inputs or not inputs.generation or not generation_chunk:
        return {"generation_claims": None}
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("claims", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    claims = claim_extractor.ClaimExtractorNode(
        model=model,
        llm_overrides=llm_overrides,
    ).run(
        generation_chunk.chunks
    )

    return {"generation_claims": claims}


@observe(name="node_generation_claims_dedup")
def node_generation_claims_dedup(state: EvalCase) -> dict[str, Any]:
    generation_claims = state["generation_claims"]
    if not generation_claims:
        return {"generation_dedup_claims": None}
    claims_list = generation_claims.claims
    dedup_result = DedupNode().run(items=[c.item for c in claims_list])
    discarded_indices = set(dedup_result.dedup_map.keys())

    claims = []
    costs = []
    for idx, (claim, cost) in enumerate(zip(claims_list, generation_claims.cost)):
        if idx not in discarded_indices:
            claims.append(claim)
            costs.append(cost)

    return {"generation_dedup_claims": ClaimArtifacts(claims=claims, cost=costs)}


@observe(name="node_grounding")
def node_grounding(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs:
        return {"grounding_metrics": None}
    dedup_claims = state["generation_dedup_claims"]
    claims: list[Claim] = dedup_claims.claims if dedup_claims else []
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("grounding", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    context_item = inputs.context
    enable_grounding = bool(context_item and context_item.text.strip())
    results = GroundingNode(judge_model=model, llm_overrides=llm_overrides).run(
        claims=claims,
        context=context_item or inputs.generation,
        enable_grounding=enable_grounding,
    )
    return {"grounding_metrics": results}


@observe(name="node_relevance")
def node_relevance(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs:
        return {"relevance_metrics": None}
    dedup_claims = state["generation_dedup_claims"]
    claims: list[Claim] = dedup_claims.claims if dedup_claims else []
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("relevance", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    results = RelevanceNode(judge_model=model, llm_overrides=llm_overrides).run(
        claims=claims,
        question=inputs.question,
        enable_relevance=bool(inputs.question and inputs.question.text.strip()),
    )
    return {"relevance_metrics": results}


@observe(name="node_redteam")
def node_redteam(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs:
        return {"redteam_metrics": None}
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("redteam", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    results = RedteamNode(judge_model=model).run(item=inputs.generation)
    return {"redteam_metrics": results}


@observe(name="node_geval_steps")
def node_geval_steps(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs or not inputs.has_geval:
        return {"geval_steps": None}
    geval_cfg = inputs.geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not metrics:
        return {"geval_steps": None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("geval_steps", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    artifacts: GevalStepsArtifacts = GevalStepsNode(
        judge_model=model,
        llm_overrides=llm_overrides,
    ).run(metrics=metrics)
    return {"geval_steps": artifacts}


@observe(name="node_geval")
def node_geval(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs or not inputs.has_geval:
        return {"geval_metrics": None}
    geval_cfg = inputs.geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not metrics:
        return {"geval_metrics": None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("geval", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    geval_steps = state.get("geval_steps")
    resolved_artifacts = geval_steps.resolved_steps if geval_steps else []
    results = GevalNode(judge_model=model).run(
        resolved_artifacts=resolved_artifacts,
        generation=inputs.generation,
        question=inputs.question,
        reference=inputs.reference,
        context=inputs.context,
    )
    return {"geval_metrics": results}


@observe(name="node_reference")
def node_reference(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    if not inputs or not inputs.reference:
        return {"reference_metrics": None}
    results = ReferenceNode().run(
        generation=inputs.generation,
        reference=inputs.reference,
        enable_generation_metrics=True,
    )
    return {"reference_metrics": results}


@observe(name="node_eval")
def node_eval(state: EvalCase) -> dict[str, Any]:
    # Orchestration-only join node. Final aggregation happens in node_report.
    return {}


def _unwrap_metrics(value: Any) -> list[Any]:
    """Accept either a wrapper object with .metrics or a plain list of MetricResult."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(getattr(value, "metrics", None) or [])


@observe(name="node_report")
def node_report(state: EvalCase) -> dict[str, Any]:
    metrics = report.aggregate(
        job_id=str(state.get("job_id") or ""),
        grounding_metrics=_unwrap_metrics(state.get("grounding_metrics")),
        relevance_metrics=_unwrap_metrics(state.get("relevance_metrics")),
        redteam_metrics=_unwrap_metrics(state.get("redteam_metrics")),
        geval_metrics=_unwrap_metrics(state.get("geval_metrics")),
        reference_metrics=_unwrap_metrics(state.get("reference_metrics")),
        cost_estimate=state.get("cost_estimate"),
        cost_actual_usd=cast(float, state.get("cost_actual_usd") or 0.0),
    )
    return {"report": metrics}


# ── Graph construction ─────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    g = StateGraph(EvalCase)

    g.add_node("metadata_scanner", node_metadata_scanner)
    g.add_node("generation_chunk", node_generation_chunk)
    g.add_node("generation_claims", node_generation_claims)
    g.add_node("generation_claims_dedup", node_generation_claims_dedup)
    g.add_node("geval_steps", node_geval_steps)
    g.add_node("grounding", node_grounding)
    g.add_node("relevance", node_relevance)
    g.add_node("redteam", node_redteam)
    g.add_node("geval", node_geval)
    g.add_node("reference", node_reference)
    g.add_node("eval", node_eval)
    g.add_node("report", node_report)

    g.set_entry_point("metadata_scanner")
    g.add_edge("metadata_scanner", "generation_chunk")
    g.add_edge("metadata_scanner", "geval_steps")
    g.add_edge("metadata_scanner", "redteam")
    g.add_edge("metadata_scanner", "reference")
    g.add_edge("generation_chunk", "generation_claims")
    g.add_edge("generation_claims", "generation_claims_dedup")
    g.add_edge("geval_steps", "geval")
    g.add_edge("generation_claims_dedup", "relevance")
    g.add_edge("generation_claims_dedup", "grounding")
    g.add_edge("grounding", "eval")
    g.add_edge("relevance", "eval")
    g.add_edge("redteam", "eval")
    g.add_edge("geval", "eval")
    g.add_edge("reference", "eval")
    g.add_edge("eval", "report")
    g.add_edge("report", END)

    return g

def build_initial_state(
    generation: Optional[str] = None,
    question: Optional[str] = None,
    reference: Optional[str] = None,
    context: Optional[list[str]] = None,
    geval: Optional[Any] = None,
    reference_files: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "case_id": f"record-{uuid.uuid4().hex[:8]}",
        "generation": generation,
        "question": question,
        "reference": reference,
        "context": context,
        "geval": geval,
        "reference_files": reference_files or [],
    }
