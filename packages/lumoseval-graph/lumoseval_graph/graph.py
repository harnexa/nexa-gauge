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
from typing import Any, Mapping

from lumoseval_core.config import config as cfg
from lumoseval_core.constants import GENERATION_CHUNK_SIZE_TOKENS
from lumoseval_core.types import (
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    EvalCase,
    GevalMetrics,
    GevalStepsArtifacts,
    GroundingMetrics,
    RedteamMetrics,
    ReferenceMetrics,
    RelevanceMetrics,
)

from .llm import get_judge_model
from .nodes import claim_extractor, report
from .nodes.chunk_extractor import ChunkExtractorNode
from .nodes.dedup import DedupNode
from .nodes.metrics.geval import GevalNode, GevalStepsNode
from .nodes.metrics.grounding import GroundingNode
from .nodes.metrics.redteam import RedteamNode
from .nodes.metrics.reference import ReferenceNode
from .nodes.metrics.relevance import RelevanceNode
from .nodes.scanner import scan as scan_record
from .observability import observe

logger = logging.getLogger(__name__)


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


def _record_node_model_usage(
    state: Mapping[str, Any],
    node_name: str,
    node: Any,
) -> dict[str, dict[str, Any]]:
    del state
    usage = node.get_model_usage() if hasattr(node, "get_model_usage") else {}
    return {node_name: usage}


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

    node = ChunkExtractorNode(chunk_size=GENERATION_CHUNK_SIZE_TOKENS)
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
            "node_model_usage": _record_node_model_usage(state, "claims", node),
        }

    claim_artifact = node.run(generation_chunk.chunks)

    return {
        "generation_claims": claim_artifact,
        "node_model_usage": _record_node_model_usage(state, "claims", node),
    }


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
            "node_model_usage": _record_node_model_usage(state, "grounding", node),
        }
    claims: list[Claim] = dedup_claims.claims if dedup_claims else []

    results = node.run(
        claims=claims,
        context=context_item or inputs.generation,
        enable_grounding=inputs.has_context,
    )
    return {
        "grounding_metrics": results,
        "node_model_usage": _record_node_model_usage(state, "grounding", node),
    }


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
            "node_model_usage": _record_node_model_usage(state, "relevance", node),
        }

    claims: list[Claim] = dedup_claims.claims if dedup_claims else []

    results = node.run(
        claims=claims,
        question=inputs.question,
    )
    return {
        "relevance_metrics": results,
        "node_model_usage": _record_node_model_usage(state, "relevance", node),
    }


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
            "node_model_usage": _record_node_model_usage(state, "redteam", node),
        }

    results = node.run(
        generation=inputs.generation,
        question=inputs.question,
        reference=inputs.reference,
        context=inputs.context,
        redteam=inputs.redteam,
    )
    return {
        "redteam_metrics": results,
        "node_model_usage": _record_node_model_usage(state, "redteam", node),
    }


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
            "node_model_usage": _record_node_model_usage(state, "geval_steps", node),
        }

    artifacts: GevalStepsArtifacts = node.run(metrics=metrics)
    return {
        "geval_steps": artifacts,
        "node_model_usage": _record_node_model_usage(state, "geval_steps", node),
    }


@observe(name="node_geval")
def node_geval(state: EvalCase) -> dict[str, Any]:
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)

    should_run = bool(inputs and inputs.has_generation and inputs.has_geval)

    if not should_run:
        return {"geval_metrics": None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("geval", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = GevalNode(judge_model=model, llm_overrides=llm_overrides)

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
            "node_model_usage": _record_node_model_usage(state, "geval", node),
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
    return {
        "geval_metrics": results,
        "node_model_usage": _record_node_model_usage(state, "geval", node),
    }


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
    report_payload = report.aggregate(state=state)
    out: dict[str, Any] = {"report": report_payload}
    if _is_estimate_mode(state):
        out["cost_estimate"] = _sum_cost_estimates(state.get("estimated_costs"))
    return out
