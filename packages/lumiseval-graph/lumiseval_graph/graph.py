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
from lumiseval_core.constants import GENERATION_CHUNK_SIZE_TOKENS
from lumiseval_core.types import (
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    ExecutionMode,
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
from .nodes.scanner import scan as scan_record

logger = logging.getLogger(__name__)

# ── Graph state ────────────────────────────────────────────────────────────
class EvalCase(TypedDict):
    """Canonical dataset row used by adapters and dataset runners."""
    # Required
    record: dict[str, str]
    llm_overrides: Optional[Mapping[str, Any]]
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
    

def _is_estimate_mode(state: Mapping[str, Any]) -> bool:
    return state.get("execution_mode") == "estimate"


def _record_estimated_cost(
    state: Mapping[str, Any],
    node_name: str,
    cost: CostEstimate,
) -> dict[str, CostEstimate]:
    costs = dict((state.get("estimated_costs") or {}))
    costs[node_name] = cost
    return costs


def _sum_cost_estimates(costs: Mapping[str, CostEstimate] | None) -> CostEstimate:
    if not costs:
        return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)

    total_cost = 0.0
    total_input = 0.0
    total_output = 0.0
    saw_input = False
    saw_output = False

    for estimate in costs.values():
        total_cost += float(estimate.cost or 0.0)
        if estimate.input_tokens is not None:
            total_input += float(estimate.input_tokens)
            saw_input = True
        if estimate.output_tokens is not None:
            total_output += float(estimate.output_tokens)
            saw_output = True

    return CostEstimate(
        cost=total_cost,
        input_tokens=total_input if saw_input else None,
        output_tokens=total_output if saw_output else None,
    )


@observe(name="node_scan")
def node_metadata_scanner(state: EvalCase) -> dict[str, Any]:
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
    estimate_mode = _is_estimate_mode(state)

    should_run = bool(inputs and inputs.has_generation)
    
    if not should_run:
        return {"generation_chunk": None}

    node = ChunkExtractorNode(
        chunk_size=GENERATION_CHUNK_SIZE_TOKENS
    )
    if estimate_mode:
        chunk_artifact = node.run(item=inputs.generation)
        return {
            "generation_chunk": chunk_artifact,
            "estimated_costs": _record_estimated_cost(state, "chunk", chunk_artifact.cost),
        }

    chunk_artifact: ChunkArtifacts = node.run(item=inputs.generation)
    return {"generation_chunk": chunk_artifact}


@observe(name="node_generation_claims")
def node_generation_claims(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    generation_chunk = state.get("generation_chunk")

    should_run = bool(inputs and inputs.has_generation)
    if not estimate_mode:
        should_run = bool(should_run and generation_chunk and generation_chunk.chunks)

    if not should_run:
        return {"generation_claims": None}
    
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("claims", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = claim_extractor.ClaimExtractorNode(
        model=model,
        llm_overrides=llm_overrides,
    )
    if estimate_mode:
        estimate_chunks = generation_chunk.chunks if generation_chunk else []
        estimated_cost = node.estimate(chunks=estimate_chunks)
        return {
            "generation_claims": ClaimArtifacts(claims=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "claims", estimated_cost),
        }

    claim_artifact = node.run(generation_chunk.chunks)

    return {"generation_claims": claim_artifact}


@observe(name="node_generation_claims_dedup")
def node_generation_claims_dedup(state: EvalCase) -> dict[str, Any]:
    generation_claims = state["generation_claims"]
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    claims_list = generation_claims.claims if generation_claims else []

    should_run = bool(inputs and inputs.has_generation)
    if not estimate_mode:
        should_run = bool(should_run and claims_list)

    if not should_run:
        return {"generation_dedup_claims": None}
    
    node = DedupNode()
    if estimate_mode:
        dedup_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        return {
            "generation_dedup_claims": ClaimArtifacts(claims=[], cost=dedup_cost),
            "estimated_costs": _record_estimated_cost(state, "dedup", dedup_cost),
        }

    dedup_result = node.run(items=[c.item for c in claims_list])
    discarded_indices = set(dedup_result.dedup_map.keys())

    claims: list[Claim] = []
    for idx, claim in enumerate(claims_list):
        if idx not in discarded_indices:
            claims.append(claim)

    dedup_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
    return {"generation_dedup_claims": ClaimArtifacts(claims=claims, cost=dedup_cost)}


@observe(name="node_grounding")
def node_grounding(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    dedup_claims = state["generation_dedup_claims"]
    should_run = bool(inputs and inputs.has_generation and inputs.has_context)

    if not estimate_mode:
        should_run = bool(should_run and dedup_claims)

    if not should_run:
        return {"grounding_metrics": None}

    context_item = inputs.context
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("grounding", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = GroundingNode(judge_model=model, llm_overrides=llm_overrides)

    if estimate_mode:
        estimated_cost = node.estimate(context=context_item)
        return {
            "grounding_metrics": GroundingMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "grounding", estimated_cost),
        }
    claims: list[Claim] = dedup_claims.claims if dedup_claims else []

    results = node.run(
        claims=claims,
        context=context_item or inputs.generation,
        enable_grounding=inputs.has_context,
    )
    return {"grounding_metrics": results}


@observe(name="node_relevance")
def node_relevance(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    dedup_claims = state["generation_dedup_claims"]

    should_run = bool(inputs and inputs.has_generation and inputs.has_question)
    if not estimate_mode:
        should_run = bool(should_run and dedup_claims)

    if not should_run:
        return {"relevance_metrics": None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("relevance", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = RelevanceNode(judge_model=model, llm_overrides=llm_overrides)

    if estimate_mode:
        estimated_cost = node.estimate(question=inputs.question)
        return {
            "relevance_metrics": RelevanceMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "relevance", estimated_cost),
        }

    claims: list[Claim] = dedup_claims.claims if dedup_claims else []

    results = node.run(
        claims=claims,
        question=inputs.question,
    )
    return {"relevance_metrics": results}


@observe(name="node_redteam")
def node_redteam(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)

    should_run = bool(inputs and inputs.has_generation)

    if not should_run:
        return {"redteam_metrics": None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("redteam", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = RedteamNode(judge_model=model, llm_overrides=llm_overrides)
    if estimate_mode:
        estimated_cost = node.estimate(
            generation=inputs.generation,
            question=inputs.question,
            reference=inputs.reference,
            context=inputs.context,
            redteam=inputs.redteam,
        )
        return {
            "redteam_metrics": RedteamMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "redteam", estimated_cost),
        }

    results = node.run(
        generation=inputs.generation,
        question=inputs.question,
        reference=inputs.reference,
        context=inputs.context,
        redteam=inputs.redteam,
    )
    return {"redteam_metrics": results}


@observe(name="node_geval_steps")
def node_geval_steps(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)

    should_run = bool(inputs and inputs.has_generation and inputs.has_geval)

    if not should_run:
        return {"geval_steps": None}
    
    geval_cfg = inputs.geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("geval_steps", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = GevalStepsNode(
        judge_model=model,
        llm_overrides=llm_overrides,
    )
    if estimate_mode:
        estimated_cost = node.estimate(input_tokens=0.0, output_tokens=0.0)
        return {
            "geval_steps": GevalStepsArtifacts(resolved_steps=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "geval_steps", estimated_cost),
        }

    artifacts: GevalStepsArtifacts = node.run(metrics=metrics)
    return {"geval_steps": artifacts}


@observe(name="node_geval")
def node_geval(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)

    should_run = bool(inputs and inputs.has_generation and inputs.has_geval)

    if not should_run:
        return {"geval_metrics": None}
    
    geval_cfg = inputs.geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("geval", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = GevalNode(judge_model=model)

    if estimate_mode:
        estimated_cost = node.estimate(
            generation=inputs.generation,
            question=inputs.question,
            reference=inputs.reference,
            context=inputs.context,
            geval=inputs.geval,
        )
        return {
            "geval_metrics": GevalMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "geval", estimated_cost),
        }

    geval_steps = state.get("geval_steps")
    resolved_artifacts = geval_steps.resolved_steps if geval_steps else []

    results = node.run(
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
    estimate_mode = _is_estimate_mode(state)

    should_run = bool(inputs and inputs.has_generation and inputs.has_reference)

    if not should_run:
        return {"reference_metrics": None}
    
    node = ReferenceNode()
    if estimate_mode:
        estimated_cost = node.estimate(input_tokens=0.0, output_tokens=0.0)
        return {
            "reference_metrics": ReferenceMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "reference", estimated_cost),
        }

    results = node.run(
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
    out: dict[str, Any] = {"report": metrics}
    if _is_estimate_mode(state):
        out["cost_estimate"] = _sum_cost_estimates(state.get("estimated_costs"))
    return out
