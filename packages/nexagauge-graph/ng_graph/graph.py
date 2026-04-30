"""LangGraph node handlers for the topology-driven evaluation pipeline."""

import logging
from typing import Any, Mapping

from ng_core.config import config as cfg
from ng_core.constants import GENERATION_CHUNK_SIZE_TOKENS
from ng_core.types import (
    Chunk,
    ChunkArtifacts,
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
from .nodes import eval as eval_node
from .nodes.chunk_extractor import ChunkExtractorNode
from .nodes.metrics.geval import GevalNode, GevalStepsNode
from .nodes.metrics.grounding import GroundingNode
from .nodes.metrics.redteam import RedteamNode
from .nodes.metrics.reference import ReferenceNode
from .nodes.metrics.relevance import RelevanceNode
from .nodes.refiner import RefinerNode
from .nodes.scanner import scan as scan_record
from .topology import NODES_BY_NAME, NodeSpec

logger = logging.getLogger(__name__)


def _is_estimate_mode(state: Mapping[str, Any]) -> bool:
    """Return whether this execution is in estimate-only mode.

    Estimate mode runs node estimators (or zero-cost placeholders) instead of
    billable LLM-heavy runtime paths.
    """
    return state.get("execution_mode") == "estimate"


def _record_estimated_cost(
    state: Mapping[str, Any],
    node_name: str,
    cost: CostEstimate,
) -> dict[str, CostEstimate]:
    """Merge one node's estimate into ``estimated_costs`` and return a patch.

    Node handlers stay pure: they return patches and never mutate shared state
    in place.
    """
    costs = dict((state.get("estimated_costs") or {}))
    costs[node_name] = cost
    return costs


def _record_node_model_usage(
    state: Mapping[str, Any],
    node_name: str,
    node: Any,
) -> dict[str, dict[str, Any]]:
    """Capture per-node model usage telemetry as a state patch.

    Nodes that expose ``get_model_usage`` can report primary/fallback call
    counts; other nodes return an empty usage payload.
    """
    del state
    usage = node.get_model_usage() if hasattr(node, "get_model_usage") else {}
    return {node_name: usage}


def _sum_cost_estimates(costs: Mapping[str, CostEstimate] | None) -> CostEstimate:
    """Aggregate per-node cost estimates into one total estimate object.

    Token totals are only emitted when at least one component estimate
    provided that token dimension.
    """
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


def _node_spec(node_name: str) -> NodeSpec:
    """Fetch validated topology spec for a node, or fail fast."""
    try:
        return NODES_BY_NAME[node_name]
    except KeyError as exc:
        raise ValueError(f"Unknown node '{node_name}' in graph handler.") from exc


def _skip_output(node_name: str) -> dict[str, Any]:
    """Return the canonical skip patch for an ineligible node.

    Uses explicit topology ``skip_output`` when defined; otherwise defaults to
    setting the node's ``state_key`` to ``None``.
    """
    spec = _node_spec(node_name)
    if spec.skip_output:
        return dict(spec.skip_output)
    if spec.state_key:
        return {spec.state_key: None}
    return {}


def _eligible_for_node(state: Mapping[str, Any], node_name: str) -> bool:
    """Evaluate topology eligibility gates for a node against scanned inputs.

    This centralizes generation/context/question/geval/reference guards so
    every node follows one consistent skip policy.
    """
    spec = _node_spec(node_name)
    if node_name == "scan":
        return True

    inputs = state.get("inputs")
    if inputs is None:
        return False
    if spec.requires_generation and not bool(inputs.has_generation):
        return False
    if spec.requires_context and not bool(inputs.has_context):
        return False
    if spec.requires_question and not bool(inputs.has_question):
        return False
    if spec.requires_geval and not bool(inputs.has_geval):
        return False
    if spec.requires_reference and not bool(inputs.has_reference):
        return False
    return True


def _skip_if_ineligible(state: Mapping[str, Any], node_name: str) -> dict[str, Any] | None:
    """Return skip patch when node is ineligible, otherwise ``None``."""
    if _eligible_for_node(state, node_name):
        return None
    return _skip_output(node_name)


def _find_upstream_artifact_producer(node_name: str, artifact_kind: str) -> str:
    """Resolve the nearest unique upstream producer for an artifact kind.

    Traverses topology prerequisites, then chooses the shallowest matching
    producer. Raises on zero matches or ambiguous nearest matches.
    """
    visited: set[str] = set()
    producers_by_depth: dict[str, int] = {}

    def _visit(name: str, depth: int) -> None:
        for parent in _node_spec(name).prerequisites:
            if parent in visited:
                continue
            visited.add(parent)
            parent_spec = _node_spec(parent)
            if parent_spec.artifact_out_kind == artifact_kind and parent_spec.state_key:
                existing = producers_by_depth.get(parent)
                producers_by_depth[parent] = depth if existing is None else min(existing, depth)
            _visit(parent, depth + 1)

    _visit(node_name, 1)
    if not producers_by_depth:
        raise ValueError(
            f"No upstream producer found for node '{node_name}' and artifact '{artifact_kind}'."
        )
    min_depth = min(producers_by_depth.values())
    nearest = sorted(node for node, depth in producers_by_depth.items() if depth == min_depth)
    if len(nearest) > 1:
        raise ValueError(
            f"Ambiguous upstream producers for node '{node_name}' and artifact "
            f"'{artifact_kind}': {nearest}."
        )
    return nearest[0]


def _upstream_artifact(state: Mapping[str, Any], node_name: str, artifact_kind: str) -> Any:
    """Load an upstream artifact instance from state via topology contract."""
    producer = _find_upstream_artifact_producer(node_name, artifact_kind)
    state_key = _node_spec(producer).state_key
    if not state_key:
        return None
    return state.get(state_key)


def _empty_artifact(kind: str) -> ChunkArtifacts | ClaimArtifacts:
    """Build an empty typed artifact used for estimate-mode placeholders."""
    zero = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
    if kind == "chunks":
        return ChunkArtifacts(chunks=[], cost=zero)
    if kind == "claims":
        return ClaimArtifacts(claims=[], cost=zero)
    raise ValueError(f"Unsupported artifact kind '{kind}'.")


def node_metadata_scanner(state: EvalCase) -> dict[str, Any]:
    """Normalize one raw record into typed ``inputs`` for all downstream nodes.

    Why needed:
    - Adapters provide loosely-shaped records.
    - Execution nodes need a stable typed contract with availability flags.

    Major points:
    - Builds a minimal raw record payload from graph state.
    - Runs scanner once per case.
    - Returns only ``inputs`` patch; downstream nodes read from this contract.
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


def node_generation_chunk(state: EvalCase) -> dict[str, Any]:
    """Split generation text into chunk artifacts used by later utility nodes.

    Why needed:
    - Claims and refiners operate on bounded units, not full-generation text.

    Major points:
    - Enforces configured chunker strategy.
    - Skips cleanly when generation is unavailable.
    - In estimate mode, emits chunk artifact + node-level estimated cost patch.
    """
    skip = _skip_if_ineligible(state, "chunk")
    if skip is not None:
        return skip

    chunker = str(state["chunker"])
    if chunker != "semchunk":
        raise ValueError(
            f"Unsupported chunker strategy '{chunker}'. Supported strategies: semchunk."
        )

    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = _node_spec("chunk").state_key or "generation_chunk"
    if not out_key:
        raise ValueError(f"node: Chunk out_key: {out_key} should be generation_chunk")

    node = ChunkExtractorNode(chunk_size=GENERATION_CHUNK_SIZE_TOKENS)
    if estimate_mode:
        chunk_artifact = node.run(item=inputs.generation)
        return {
            out_key: chunk_artifact,
            "estimated_costs": _record_estimated_cost(state, "chunk", chunk_artifact.cost),
        }

    chunk_artifact: ChunkArtifacts = node.run(item=inputs.generation)
    return {out_key: chunk_artifact}


def node_generation_refiner(state: EvalCase) -> dict[str, Any]:
    """Select a refined top-k subset of chunks from the chunk artifact stream.

    Why needed:
    - Reduces downstream claim extraction scope and cost while preserving
      representative content.

    Major points:
    - Resolves chunk input via topology artifact contract.
    - Applies configured refiner strategy and ``refiner_top_k``.
    - In estimate mode, returns an empty typed chunk artifact plus cost patch.
    """
    skip = _skip_if_ineligible(state, "refiner")
    if skip is not None:
        return skip

    spec = _node_spec("refiner")
    estimate_mode = _is_estimate_mode(state)
    in_kind = spec.artifact_in_kind
    if in_kind == "none":
        raise ValueError("Topology contract error: 'refiner' must declare artifact_in_kind.")
    
    out_key = spec.state_key

    if not out_key:
        raise ValueError(f"node: Refiner out_key: {out_key} should be generation_refined_chunks")

    artifact = _upstream_artifact(state, "refiner", in_kind)
    chunks_list = artifact.chunks if artifact else []
    if not estimate_mode and not chunks_list:
        return {out_key: None}

    refiner_top_k = state["refiner_top_k"]
    node = RefinerNode(
        strategy=str(state["refiner"]),
        top_k=int(refiner_top_k),
    )
    if estimate_mode:
        empty = _empty_artifact("chunks")
        return {
            out_key: empty,
            "estimated_costs": _record_estimated_cost(state, "refiner", empty.cost),
        }

    refiner_result = node.run(items=[c.item for c in chunks_list])
    selected_indices = set(refiner_result.indices)

    chunks: list[Chunk] = []
    for idx, chunk in enumerate(chunks_list):
        if idx in selected_indices:
            chunks.append(chunk)

    refine_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
    return {out_key: ChunkArtifacts(chunks=chunks, cost=refine_cost)}


def node_generation_claims(state: EvalCase) -> dict[str, Any]:
    """Extract structured claims from refined chunks.

    Why needed:
    - Grounding/relevance metrics reason about claims, not raw chunks.

    Major points:
    - Resolves upstream chunks via topology artifact routing.
    - Supports estimate-mode cost-only execution.
    - Records model-usage telemetry for LLM routing observability.
    """
    skip = _skip_if_ineligible(state, "claims")
    if skip is not None:
        return skip

    spec = _node_spec("claims")
    _ = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = spec.state_key

    if not out_key:
        raise ValueError(f"node: Claims out_key: {out_key} should be generation_claims")


    in_kind = spec.artifact_in_kind
    if in_kind == "none":
        raise ValueError("Topology contract error: 'claims' must declare artifact_in_kind.")
    chunk_artifact = _upstream_artifact(state, "claims", in_kind)
    estimate_chunks = chunk_artifact.chunks if chunk_artifact else []
    if not estimate_mode and not estimate_chunks:
        return {out_key: None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("claims", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = claim_extractor.ClaimExtractorNode(
        model=model,
        llm_overrides=llm_overrides,
    )
    if estimate_mode:
        estimated_cost = node.estimate(chunks=estimate_chunks)
        return {
            out_key: ClaimArtifacts(claims=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "claims", estimated_cost),
            "node_model_usage": _record_node_model_usage(state, "claims", node),
        }

    claim_artifact = node.run(estimate_chunks)

    return {
        out_key: claim_artifact,
        "node_model_usage": _record_node_model_usage(state, "claims", node),
    }


def node_grounding(state: EvalCase) -> dict[str, Any]:
    """Score whether extracted claims are supported by context evidence.

    Why needed:
    - Grounds answer faithfulness against provided context.

    Major points:
    - Reads claim artifacts through topology input contract.
    - Honors context eligibility gates.
    - Emits typed ``GroundingMetrics`` and model-usage telemetry.
    """
    skip = _skip_if_ineligible(state, "grounding")
    if skip is not None:
        return skip

    spec = _node_spec("grounding")
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)

    out_key = spec.state_key

    if not out_key:
        raise ValueError(f"node: Grounding out_key: {out_key} should be grounding_metrics")
    

    in_kind = spec.artifact_in_kind
    if in_kind == "none":
        raise ValueError("Topology contract error: 'grounding' must declare artifact_in_kind.")
    claim_artifact = _upstream_artifact(state, "grounding", in_kind)
    claims = claim_artifact.claims if claim_artifact else []
    if not estimate_mode and not claims:
        return {out_key: None}

    context_item = inputs.context
    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("grounding", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = GroundingNode(judge_model=model, llm_overrides=llm_overrides)

    if estimate_mode:
        estimated_cost = node.estimate(context=context_item)
        return {
            out_key: GroundingMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "grounding", estimated_cost),
            "node_model_usage": _record_node_model_usage(state, "grounding", node),
        }

    results = node.run(
        claims=claims,
        context=context_item or inputs.generation,
        enable_grounding=inputs.has_context,
    )
    return {
        out_key: results,
        "node_model_usage": _record_node_model_usage(state, "grounding", node),
    }


def node_relevance(state: EvalCase) -> dict[str, Any]:
    """Score whether extracted claims are relevant to the user question.

    Why needed:
    - Distinguishes faithful-but-off-topic answers from relevant answers.

    Major points:
    - Consumes claim artifacts from upstream contract.
    - Requires question availability.
    - Emits typed ``RelevanceMetrics`` and model usage.
    """
    skip = _skip_if_ineligible(state, "relevance")
    if skip is not None:
        return skip

    spec = _node_spec("relevance")
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = spec.state_key

    if not out_key:
        raise ValueError(f"node: Relevance out_key: {out_key} should be relevance_metrics")
    
    in_kind = spec.artifact_in_kind
    if in_kind == "none":
        raise ValueError("Topology contract error: 'relevance' must declare artifact_in_kind.")
    claim_artifact = _upstream_artifact(state, "relevance", in_kind)
    claims = claim_artifact.claims if claim_artifact else []
    if not estimate_mode and not claims:
        return {out_key: None}

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("relevance", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = RelevanceNode(judge_model=model, llm_overrides=llm_overrides)

    if estimate_mode:
        estimated_cost = node.estimate(question=inputs.question)
        return {
            out_key: RelevanceMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "relevance", estimated_cost),
            "node_model_usage": _record_node_model_usage(state, "relevance", node),
        }

    results = node.run(
        claims=claims,
        question=inputs.question,
    )
    return {
        out_key: results,
        "node_model_usage": _record_node_model_usage(state, "relevance", node),
    }


def node_redteam(state: EvalCase) -> dict[str, Any]:
    """Run safety/red-team evaluation metrics over the case inputs.

    Why needed:
    - Captures policy and vulnerability signals independent of claim flow.

    Major points:
    - Uses generation/question/reference/context/redteam config directly.
    - Supports estimate-mode token/cost simulation.
    - Emits typed ``RedteamMetrics`` plus model-usage telemetry.
    """
    skip = _skip_if_ineligible(state, "redteam")
    if skip is not None:
        return skip

    spec = _node_spec("redteam")
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = spec.state_key
    if not out_key:
        raise ValueError(f"node: RedTeam out_key: {out_key} should be redteam_metrics")

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
            out_key: RedteamMetrics(metrics=[], cost=estimated_cost),
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
        out_key: results,
        "node_model_usage": _record_node_model_usage(state, "redteam", node),
    }


def node_geval_steps(state: EvalCase) -> dict[str, Any]:
    """Resolve GEval rubric steps per metric before GEval scoring.

    Why needed:
    - GEval scoring expects normalized evaluation-step artifacts.

    Major points:
    - Reads GEval metric configs from ``inputs``.
    - Can reuse/generate steps with cache-backed artifact store.
    - Emits typed ``GevalStepsArtifacts`` and model usage.
    """
    skip = _skip_if_ineligible(state, "geval_steps")
    if skip is not None:
        return skip

    spec = _node_spec("geval_steps")
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = spec.state_key
    if not out_key:
        raise ValueError(f"node: GevalSteps out_key: {out_key} should be geval_steps")

    geval_cfg = inputs.geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []

    llm_overrides = state.get("llm_overrides")
    model = get_judge_model("geval_steps", cfg.LLM_MODEL, llm_overrides=llm_overrides)
    node = GevalStepsNode(
        judge_model=model,
        llm_overrides=llm_overrides,
        artifact_cache_store=state.get("__cache_store"),
    )
    if estimate_mode:
        estimated_cost = node.estimate(input_tokens=0.0, output_tokens=0.0)
        return {
            out_key: GevalStepsArtifacts(resolved_steps=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "geval_steps", estimated_cost),
            "node_model_usage": _record_node_model_usage(state, "geval_steps", node),
        }

    artifacts: GevalStepsArtifacts = node.run(metrics=metrics)
    return {
        out_key: artifacts,
        "node_model_usage": _record_node_model_usage(state, "geval_steps", node),
    }


def node_geval(state: EvalCase) -> dict[str, Any]:
    """Execute GEval scoring using resolved GEval steps + case inputs.

    Why needed:
    - Produces rubric-based quality signals complementary to other metrics.

    Major points:
    - Depends on ``geval_steps`` artifact output.
    - Uses direct input fields (generation/question/reference/context).
    - Emits typed ``GevalMetrics`` and model-usage telemetry.
    """
    skip = _skip_if_ineligible(state, "geval")
    if skip is not None:
        return skip

    spec = _node_spec("geval")
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = spec.state_key 
    if not out_key:
        raise ValueError(f"node: Geval out_key: {out_key} should be geval_metrics")

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
            out_key: GevalMetrics(metrics=[], cost=estimated_cost),
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
        out_key: results,
        "node_model_usage": _record_node_model_usage(state, "geval", node),
    }


def node_reference(state: EvalCase) -> dict[str, Any]:
    """Compute reference-based quality metrics against target answers.

    Why needed:
    - Measures answer quality where a reference answer is available.

    Major points:
    - Uses generation + reference inputs directly.
    - Skips when reference is absent.
    - Returns typed ``ReferenceMetrics``; estimate mode emits cost placeholder.
    """
    skip = _skip_if_ineligible(state, "reference")
    if skip is not None:
        return skip

    spec = _node_spec("reference")
    inputs = state["inputs"]
    estimate_mode = _is_estimate_mode(state)
    out_key = spec.state_key or "reference_metrics"

    if not out_key:
        raise ValueError(f"node: Reference out_key: {out_key} should be reference_metrics")

    node = ReferenceNode()
    if estimate_mode:
        estimated_cost = node.estimate(input_tokens=0.0, output_tokens=0.0)
        return {
            out_key: ReferenceMetrics(metrics=[], cost=estimated_cost),
            "estimated_costs": _record_estimated_cost(state, "reference", estimated_cost),
        }

    results = node.run(
        generation=inputs.generation,
        reference=inputs.reference,
        enable_generation_metrics=True,
    )
    return {out_key: results}


def node_eval(state: EvalCase) -> dict[str, Any]:
    """Aggregate per-case metric rows into the normalized eval summary patch.

    This is an orchestration join node; metric computations happen upstream.
    """
    return eval_node.node_eval(state)


def node_report(state: EvalCase) -> dict[str, Any]:
    """Project final case state into a report-ready payload.

    Why needed:
    - Produces stable per-case output contract for CLI/API reporting.

    Major points:
    - Delegates projection to ``nodes.report.aggregate``.
    - In estimate mode, also emits total cost estimate aggregation.
    """
    report_payload = report.aggregate(state=state)
    out: dict[str, Any] = {"report": report_payload}
    if _is_estimate_mode(state):
        out["cost_estimate"] = _sum_cost_estimates(state.get("estimated_costs"))
    return out
