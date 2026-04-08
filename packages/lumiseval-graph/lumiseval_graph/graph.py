"""
LangGraph Orchestration Graph — the core evaluation pipeline.

Node sequence:
  scan → chunk → claims → dedup →
  geval_steps →
  [parallel: relevance, grounding, redteam, geval, reference] →
  eval → result

TODO:
  - Implement async TaskIQ dispatch for batch jobs.
  - Stream progress to CLI/API consumers.
  - Persist EvalReport to SQLite via SQLModel.
"""

import logging
import uuid
from typing import Any, NotRequired, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph
from lumiseval_core.config import config as cfg
from lumiseval_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from lumisenoval_core.types import (
    ChunkArtifacts,
    ClaimArtifacts,
    CostEstimate,
    DedupArtifacts,
    Geval,
    GevalMetrics,
    GevalStepsArtifacts,
    GroundingMetrics,
    Inputs,
    Item,
    MetricResult,
    ReferencePayload,
    RedteamMetrics,
    RelevanceMetrics,
)
from lumiseval_evidence.indexer import index_file

from .llm import get_judge_model
from .log import get_node_logger, print_pipeline_footer, print_pipeline_header
from .nodes import claim_extractor, eval
from .nodes.chunk_extractor import ChunkExtractorNode
from .nodes.dedupe import DedupNode
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


# ── Graph state ────────────────────────────────────────────────────────────
class EvalCase(TypedDict):
    """Canonical dataset row used by adapters and dataset runners."""
    # Required
    case_id: str
    record: dict[str, Any]

    # Optional with defaults handled at construction time
    dataset: NotRequired[str]          # default: DEFAULT_DATASET_NAME
    split: NotRequired[str]            # default: DEFAULT_SPLIT
    job_id: NotRequired[str]           # default: uuid4
    reference_files: NotRequired[list[str]]

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
    reference_metrics: Optional[ReferencePayload]


@observe(name="node_scan")
def node_metadata_scanner(state: dict[str, Any]) -> dict[str, Any]:
    """
    In LangGraph, each node gets the current full state snapshot.
    So after node_metadata_scanner returns {"inputs": ...}, that patch is merged into state, and downstream nodes like node_chunk receive the same full state with
    inputs included.

    Important in your current file:

    - node_chunk still reads state["generation"], not state["inputs"].
    - So inputs is available downstream, but node_chunk won’t use it unless you change it.
    """
    # Per-record scan only: hydrate `inputs` from the current state payload.
    geval_raw = state.get("geval")
    if hasattr(geval_raw, "model_dump"):
        geval_raw = geval_raw.model_dump()

    raw_record = {
        "case_id": state.get("case_id") or "record-0",
        "generation": state.get("generation"),
        "question": state.get("question"),
        "reference": state.get("reference"),
        "context": state.get("context"),
        "geval": geval_raw,
        "reference_files": state.get("reference_files") or [],
    }

    case = scan_record(raw_record, idx=0)
    return {"inputs": case.get("inputs")}


@observe(name="node_generation_chunk")
def node_generation_chunk(state: EvalCase) -> EvalCase:
    if not state["inputs"].generation:
        return {"generation_chunk": None}
    text = state["inputs"].generation.text
    chunks: ChunkArtifacts = ChunkExtractorNode(
        chunk_size=GENERATION_CHUNK_SIZE_TOKENS
    ).run(record=text)
    return {"generation_chunk": chunks}


@observe(name="node_generation_claims")
def node_generation_claims(state: EvalCase) -> EvalCase:
    if not state["inputs"].generation or not state.get("generation_chunk"):
        return {"generation_claims": None}
    model = get_judge_model("node_generation_claims")
    claims = claim_extractor.ClaimExtractorNode(model=model).run(
        state["generation_chunk"].chunks
    )
    return {"generation_claims": claims}


@observe(name="node_dedup")
def node_generation_claims_dedup(state: EvalCase) -> EvalCase:
    if not state["generation_claims"]:
        return {"generation_dedup_claims": None}
    unique_items, dedup_map = DedupNode().run(items=state["generation_claims"].claims)
    selected_ids = set(dedup_map.values())
    claims = [claim for claim in claims if claim.id in selected_ids]
    return {"generation_dedup_claims": claims}


@observe(name="node_grounding")
def node_grounding(state: EvalCase) -> EvalCase:
    claims: list[Claim] = state["generation_dedup_claims"].claims or []
    context = state["context"]
    model = get_judge_model("node_grounding")
    results = GroundingNode(judge_model=model).run(
        claims=claims,
        context=context,
        enable_grounding=state["inputs"].enable_grounding,
    )
    return {"grounding_metrics": results}



@observe(name="node_relevance")
def node_relevance(state: EvalCase) -> EvalCase:
    claims: list[Claim] = state["generation_dedup_claims"].claims or []
    question = state["inputs"].question
    model = get_judge_model("node_relevance")
    results = RelevanceNode(judge_model=model).run(
        claims=claims,
        question=state.get("question"),
        enable_relevance=state["inputs"].enable_relevance,
    )
    return {"relevance_metrics": results}


@observe(name="node_redteam")
def node_redteam(state: EvalCase) -> EvalCase:
    model = get_judge_model("node_redteam")
    results = RedteamNode(judge_model=model).run(item=state["inputs"].generation)
    return {"redteam_metrics": results}



@observe(name="node_geval_steps")
def node_geval_steps(state: EvalCase) -> EvalCase:
    geval_cfg = state["input"].geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not state["inputs"].enable_geval or not metrics:
        return {"geval_steps_by_signature": {}}

    model = get_judge_model("geval_steps", state["job_config"].judge_model)
    steps_by_signature = GevalStepsNode(judge_model=model).run(metrics=metrics)
    return {"geval_steps_by_signature": steps_by_signature}



@observe(name="node_geval")
def node_geval(state: EvalCase) -> EvalCase:
    geval_cfg = state["geval"]
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not state["job_config"].enable_geval or not metrics:
        return {"geval_metrics": []}

    model = get_judge_model("geval", state["judge_model"])
    results = GevalNode(judge_model=model).run(
        metrics=metrics,
        generation=state.get("inputs").generation,
        question=state.get("inputs").question,
        reference=state.get("inputs").reference,
        context=state.get("inputs").context,
        steps_by_signature=state.get("geval_steps_by_signature") or {},
    )
    return {"geval_metrics": results}


@observe(name="node_reference")
def node_reference(state: EvalCase) -> EvalCase:
    if not state["job_config"].enable_reference:
        return {"reference_metrics": []}
    reference = state.get("reference")
    if not reference:
        return {"reference_metrics": []}
    results = ReferenceNode().run(
        generation=state["generation"],
        reference=reference,
        enable_generation_metrics=True,
    )
    return {"reference_metrics": results}


@observe(name="node_eval")
def node_eval(state: EvalCase) -> EvalCase:
    report = eval.aggregate(
        job_id=state["job_config"].job_id,
        grounding_metrics=state.get("grounding_metrics") or [],
        relevance_metrics=state.get("relevance_metrics") or [],
        redteam_metrics=state.get("redteam_metrics") or [],
        geval_metrics=state.get("geval_metrics") or [],
        reference_metrics=state.get("reference_metrics") or [],
        cost_estimate=state.get("cost_estimate"),
        cost_actual_usd=state.get("cost_actual_usd", 0.0),
        job_config=state["job_config"],
    )
    return {"report": report}


# ── Graph construction ─────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    g = StateGraph(EvalCase)

    g.add_node("scan", node_metadata_scanner)
    g.add_node("generation_chunk", node_generation_chunk)
    g.add_node("claims", node_claims)
    g.add_node("dedup", node_dedup)
    g.add_node("geval_steps", node_geval_steps)
    g.add_node("relevance", node_relevance)
    g.add_node("grounding", node_grounding)
    g.add_node("redteam", node_adversarial)
    g.add_node("geval", node_geval)
    g.add_node("reference", node_reference)
    g.add_node("eval", node_eval)

    g.set_entry_point("scan")
    g.add_edge("scan", "chunk")
    g.add_edge("scan", "geval_steps")
    g.add_edge("scan", "redteam")
    g.add_edge("scan", "reference")
    g.add_edge("chunk", "claims")
    g.add_edge("claims", "dedup")
    g.add_edge("geval_steps", "geval")
    g.add_edge("dedup", "relevance")
    g.add_edge("dedup", "grounding")
    g.add_edge("relevance", "eval")
    g.add_edge("grounding", "eval")
    g.add_edge("redteam", "eval")
    g.add_edge("geval", "eval")
    g.add_edge("reference", "eval")
    g.add_edge("eval", END)

    return g



@observe(name="lumiseval_pipeline")
def run_graph(
    generation: str,
    question: Optional[str] = None,
    reference: Optional[str] = None,
    context: Optional[list[str]] = None,
    geval: Optional[GevalConfig] = None,
    reference_files: Optional[list[str]] = None,
    job_config: Optional[EvalJobConfig] = None,
) -> EvalReport:
    """Execute the full evaluation pipeline synchronously.

    Args:
        generation: The LLM-generated text to evaluate.
        job_config: Evaluation configuration. A default config is created if not provided.
        question: Optional query that produced the generation (improves RAGAS scores).
        reference: Optional reference answer (enables context recall).
        context: Optional retrieval context passages for retrieval-path metrics.
        geval: Optional GEval metric contract (canonical custom-judge path).
        reference_files: Optional list of file paths to index before evaluation.

    Returns:
        EvalReport with retrieval_score, answer_score, composite_score, and per-claim verdicts.
    """
    initial_state = build_initial_state(
        generation=generation,
        job_config=job_config,
        question=question,
        reference=reference,
        context=context,
        geval=geval,
        reference_files=reference_files,
        target_node="eval",
    )
    job_config = initial_state["job_config"]

    print_pipeline_header(
        job_id=job_config.job_id,
        model=job_config.judge_model,
        web_search=job_config.web_search,
    )

    # Index reference files into local LanceDB before graph runs
    if reference_files:
        _log_scanner.info(f"Indexing {len(reference_files)} reference file(s) into LanceDB")

        for fpath in reference_files:
            # try:
            count = index_file(fpath)
            _log_scanner.success(f"Indexed {count} passages from {fpath}")
            # except Exception as exc:
            #     _log_scanner.warning(f"Failed to index {fpath}: {exc}")

    graph = build_graph().compile()
    final_state = cast(EvalState, graph.invoke(cast(Any, initial_state)))

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    report = final_state.get("report")
    if report is None:
        raise RuntimeError("Graph completed without an evaluation report.")
    print_pipeline_footer(
        composite_score=report.composite_score,
        cost_usd=report.cost_actual_usd,
    )

    update_trace(
        metadata={
            "job_id": job_config.job_id,
            "judge_model": job_config.judge_model,
            "cost_actual_usd": report.cost_actual_usd,
            "warnings": report.warnings,
        }
    )
    if report.composite_score is not None:
        score_trace("composite_score", report.composite_score)

    return report
