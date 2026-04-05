"""
LangGraph Orchestration Graph — the core evaluation pipeline.

Node sequence:
  scan → chunk → claims → dedupe →
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
from typing import Any, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph
from lumiseval_core.config import config as cfg
from lumiseval_core.types import (
    Chunk,
    Claim,
    CostEstimate,
    EvalCase,
    EvalJobConfig,
    EvalReport,
    GevalConfig,
    InputMetadata,
    MetricResult,
)
from lumiseval_evidence.indexer import index_file
from lumiseval_ingest.chunker import chunk_text
from lumiseval_ingest.scanner import scan_cases

from .llm import get_judge_model
from .log import get_node_logger, print_pipeline_footer, print_pipeline_header
from .nodes import claim_extractor, eval
from .nodes.metrics.dedupe import DedupeNode
from .nodes.metrics.geval import GevalNode, GevalStepsNode
from .nodes.metrics.grounding import GroundingNode
from .nodes.metrics.redteam import RedteamNode
from .nodes.metrics.reference import ReferenceNode
from .nodes.metrics.relevance import RelevanceNode
from .observability import observe, score_trace, update_trace

logger = logging.getLogger(__name__)

# Module-level node loggers — one per pipeline node
_log_scanner = get_node_logger("scan")
_log_chunker = get_node_logger("chunk")
_log_claims = get_node_logger("claims")
_log_mmr = get_node_logger("dedupe")
_log_ragas = get_node_logger("relevance")
_log_hallucination = get_node_logger("grounding")
_log_adversarial = get_node_logger("redteam")
_log_geval_steps = get_node_logger("geval_steps")
_log_geval = get_node_logger("geval")
_log_reference = get_node_logger("reference")
_log_agg = get_node_logger("eval")


# ── Graph state ────────────────────────────────────────────────────────────


class EvalState(TypedDict):
    # ── Inputs: set at graph entry, never mutated by nodes ────────────────
    generation: str               # LLM output being evaluated; read by scan, chunk, redteam, reference
    question: Optional[str]       # Original query; read by relevance for answer-relevancy scoring
    reference: Optional[str]      # Ground-truth answer; read by scan (eligibility flag) and reference node (ROUGE/BLEU/METEOR)
    context: Optional[list[str]]  # Retrieved passages; gates chunk → claims → grounding path; read by grounding
    geval: Optional[GevalConfig]  # Canonical GEval contract; gates geval_steps/geval nodes


    target_node: Optional[str]    # Pipeline stop point used by CachedNodeRunner
    reference_files: list[str]    # File paths indexed into LanceDB before the graph runs (in run_graph, not inside nodes)
    job_config: EvalJobConfig     # Controls enable_* flags, judge_model, job_id; read by every node

    # ── Intermediate: written by one node, consumed by downstream nodes ───
    metadata: Optional[InputMetadata]      # Token/eligibility stats; written by scan
    cost_estimate: Optional[CostEstimate]  # Optional pre-run cost breakdown; injected by caller/runner for eval
    chunks: list[Chunk]                    # Generation split into ~512-token chunks; written by chunk, read by claims
    raw_claims: list[Claim]                # Atomic claims extracted per chunk; written by claims, read by dedupe
    unique_claims: list[Claim]             # MMR-deduplicated claims; written by dedupe, read by relevance and grounding
    geval_steps_by_signature: dict[str, list[str]]  # Signature-keyed evaluation steps; written by geval_steps

    # ── Metric results: written by parallel metric nodes, aggregated by eval ─
    grounding_metrics: list[MetricResult]  # Faithfulness verdicts per claim; written by grounding
    relevance_metrics: list[MetricResult]  # Answer-relevancy verdicts per claim; written by relevance
    redteam_metrics: list[MetricResult]    # Bias + toxicity scores; written by redteam
    geval_metrics: list[MetricResult]      # Authoritative GEval scoring branch; written by geval
    reference_metrics: list[MetricResult]  # ROUGE/BLEU/METEOR scores; written by reference

    # ── Output ────────────────────────────────────────────────────────────
    cost_actual_usd: float         # Actual LLM spend (currently always 0.0 — TODO: wire up token tracking)
    report: Optional[EvalReport]   # Final aggregated report; written by eval, returned by run_graph and written to --output-dir
    error: Optional[str]           # Pipeline error message; checked by run_graph to raise RuntimeError (unused today)


# ── Node implementations ───────────────────────────────────────────────────


@observe(name="node_scan")
def node_metadata_scanner(state: EvalState) -> dict:
    case = EvalCase(
        case_id="runtime-case",
        generation=state["generation"],
        question=state.get("question"),
        reference=state.get("reference"),
        context=state.get("context") or [],
        geval=state.get("geval"),
    )
    meta = scan_cases([case], show_progress=False)
    return {"metadata": meta}


@observe(name="node_chunk")
def node_chunk(state: EvalState) -> dict:
    if not state.get("generation"):
        return {"chunks": []}
    chunks = chunk_text(
        state["generation"],
        chunk_size=state["job_config"].chunk_size
    )
    return {"chunks": chunks}


@observe(name="node_claims")
def node_claims(state: EvalState) -> dict:
    if not state.get("generation") or not state.get("chunks"):
        return {"raw_claims": []}
    model = state["job_config"].judge_model
    claims = claim_extractor.ClaimExtractorNode(model=model).run(state["chunks"])
    return {"raw_claims": claims}


@observe(name="node_dedupe")
def node_dedupe(state: EvalState) -> dict:
    raw = state["raw_claims"]
    if not raw:
        return {"unique_claims": []}
    unique = DedupeNode(strategy="mmr").run(claims=raw)
    return {"unique_claims": unique}


@observe(name="node_geval_steps")
def node_geval_steps(state: EvalState) -> dict:
    geval_cfg = state["geval"]
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not state["job_config"].enable_geval or not metrics:
        return {"geval_steps_by_signature": {}}

    model = get_judge_model("geval_steps", state["job_config"].judge_model)
    steps_by_signature = GevalStepsNode(judge_model=model).run(metrics=metrics)
    return {"geval_steps_by_signature": steps_by_signature}


@observe(name="node_relevance")
def node_relevance(state: EvalState) -> dict:
    claims = state.get("unique_claims") or []
    if not claims:
        return {"relevance_metrics": []}
    if not state["job_config"].enable_relevance:
        return {"relevance_metrics": []}
    model = get_judge_model("relevance", state["job_config"].judge_model)
    results = RelevanceNode(judge_model=model).run(
        claims=claims,
        question=state.get("question"),
        enable_relevance=True,
    )
    return {"relevance_metrics": results}


@observe(name="node_grounding")
def node_grounding(state: EvalState) -> dict:
    claims = state.get("unique_claims") or []
    context = state["context"]
    if not claims or not context:
        return {"grounding_metrics": []}
    if not state["job_config"].enable_grounding:
        return {"grounding_metrics": []}
    model = get_judge_model("grounding", state["job_config"].judge_model)
    results = GroundingNode(judge_model=model).run(
        claims=claims,
        context=context,
        enable_grounding=True,
    )
    return {"grounding_metrics": results}


@observe(name="node_redteam")
def node_adversarial(state: EvalState) -> dict:
    if not state["job_config"].enable_redteam:
        return {"redteam_metrics": []}
    model = get_judge_model("redteam", state["job_config"].judge_model)
    results = RedteamNode(judge_model=model).run(generation=state["generation"])
    return {"redteam_metrics": results}


@observe(name="node_geval")
def node_geval(state: EvalState) -> dict:
    geval_cfg = state["geval"]
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not state["job_config"].enable_geval or not metrics:
        return {"geval_metrics": []}

    model = get_judge_model("geval", state["job_config"].judge_model)
    results = GevalNode(judge_model=model).run(
        metrics=metrics,
        generation=state["generation"],
        question=state.get("question"),
        reference=state.get("reference"),
        context=state.get("context"),
        steps_by_signature=state.get("geval_steps_by_signature") or {},
    )
    return {"geval_metrics": results}


@observe(name="node_reference")
def node_reference(state: EvalState) -> dict:
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
def node_eval(state: EvalState) -> dict:
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
    g = StateGraph(EvalState)

    g.add_node("scan", node_metadata_scanner)
    g.add_node("chunk", node_chunk)
    g.add_node("claims", node_claims)
    g.add_node("dedupe", node_dedupe)
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
    g.add_edge("claims", "dedupe")
    g.add_edge("geval_steps", "geval")
    g.add_edge("dedupe", "relevance")
    g.add_edge("dedupe", "grounding")
    g.add_edge("relevance", "eval")
    g.add_edge("grounding", "eval")
    g.add_edge("redteam", "eval")
    g.add_edge("geval", "eval")
    g.add_edge("reference", "eval")
    g.add_edge("eval", END)

    return g


def build_initial_state(
    generation: str,
    job_config: Optional[EvalJobConfig] = None,
    question: Optional[str] = None,
    reference: Optional[str] = None,
    context: Optional[list[str]] = None,
    target_node: Optional[str] = None,
    geval: Optional[GevalConfig] = None,
    reference_files: Optional[list[str]] = None,
) -> EvalState:
    """Build the canonical graph state for one evaluation input."""
    if job_config is None:
        job_config = EvalJobConfig(
            job_id=str(uuid.uuid4()),
            judge_model=cfg.LLM_MODEL,
            web_search=cfg.WEB_SEARCH_ENABLED,
            evidence_threshold=cfg.EVIDENCE_THRESHOLD,
        )
    return EvalState(
        generation=generation,
        question=question,
        reference=reference,
        context=context or [],
        target_node=target_node,
        reference_files=reference_files or [],
        geval=geval,
        job_config=job_config,
        metadata=None,
        cost_estimate=None,
        chunks=[],
        raw_claims=[],
        unique_claims=[],
        geval_steps_by_signature={},
        grounding_metrics=[],
        relevance_metrics=[],
        redteam_metrics=[],
        geval_metrics=[],
        reference_metrics=[],
        cost_actual_usd=0.0,
        report=None,
        error=None,
    )


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
