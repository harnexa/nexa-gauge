"""
LangGraph Orchestration Graph — the core evaluation pipeline.

Node sequence:
  scan → estimate → [user approve gate] → chunk →
  claims → dedupe → retrieve →
  [parallel: relevance, grounding, redteam, rubric] →
  eval → result

TODO:
  - Implement async TaskIQ dispatch for batch jobs.
  - Stream progress to CLI/API consumers.
  - Persist EvalReport to SQLite via SQLModel.
"""

import logging
import uuid
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from lumiseval_core.config import config as cfg
from lumiseval_core.types import (
    Chunk,
    Claim,
    CostEstimate,
    EvalJobConfig,
    EvalReport,
    InputMetadata,
    MetricResult,
    Rubric,
)
from lumiseval_evidence.indexer import index_file
from lumiseval_evidence.mmr import deduplicate
from lumiseval_ingest.chunker import chunk_text
from lumiseval_ingest.scanner import scan_text

from .llm import get_judge_model
from .log import get_node_logger, print_pipeline_footer, print_pipeline_header
from .nodes import claim_extractor, cost_estimator, eval
from .nodes.metrics.reference import ReferenceMetricsNode
from .nodes.metrics.grounding import GroundingNode
from .nodes.metrics.redteam import RedteamNode
from .nodes.metrics.relevance import RelevanceNode
from .nodes.metrics.rubric import RubricNode
from .observability import observe, score_trace, update_trace

logger = logging.getLogger(__name__)

# Module-level node loggers — one per pipeline node
_log_scanner = get_node_logger("scan")
_log_cost = get_node_logger("estimate")
_log_confirm = get_node_logger("approve")
_log_chunker = get_node_logger("chunk")
_log_claims = get_node_logger("claims")
_log_mmr = get_node_logger("dedupe")
_log_ragas = get_node_logger("relevance")
_log_hallucination = get_node_logger("grounding")
_log_adversarial = get_node_logger("redteam")
_log_rubric = get_node_logger("rubric")
_log_reference = get_node_logger("reference")
_log_agg = get_node_logger("eval")


# ── Graph state ────────────────────────────────────────────────────────────


class EvalState(TypedDict):
    # ── Inputs: set at graph entry, never mutated by nodes ────────────────
    generation: str               # LLM output being evaluated; read by scan, chunk, redteam, rubric, reference
    question: Optional[str]       # Original query; read by relevance for answer-relevancy scoring
    reference: Optional[str]      # Ground-truth answer; read by scan (eligibility flag) and reference node (ROUGE/BLEU/METEOR)
    context: Optional[list[str]]  # Retrieved passages; gates chunk → claims → grounding path; read by grounding
    rubric: list[Rubric]          # GEval rules; gates rubric node; count passed to estimate for cost
    
    
    target_node: Optional[str]    # Pipeline stop point used by CachedNodeRunner; passed to estimate for per-target cost
    reference_files: list[str]    # File paths indexed into LanceDB before the graph runs (in run_graph, not inside nodes)
    job_config: EvalJobConfig     # Controls enable_* flags, judge_model, job_id; read by every node

    # ── Intermediate: written by one node, consumed by downstream nodes ───
    metadata: Optional[InputMetadata]      # Token/eligibility stats; written by scan, read by estimate
    cost_estimate: Optional[CostEstimate]  # Pre-run cost breakdown; written by estimate, read by eval
    confirmed: bool                        # Passthrough flag; written by approve (always True); not read by other nodes
    chunks: list[Chunk]                    # Generation split into ~512-token chunks; written by chunk, read by claims
    raw_claims: list[Claim]                # Atomic claims extracted per chunk; written by claims, read by dedupe
    unique_claims: list[Claim]             # MMR-deduplicated claims; written by dedupe, read by relevance and grounding

    # ── Metric results: written by parallel metric nodes, aggregated by eval ─
    grounding_metrics: list[MetricResult]  # Faithfulness verdicts per claim; written by grounding
    relevance_metrics: list[MetricResult]  # Answer-relevancy verdicts per claim; written by relevance
    redteam_metrics: list[MetricResult]    # Bias + toxicity scores; written by redteam
    rubric_metrics: list[MetricResult]     # GEval pass/fail per rule; written by rubric
    reference_metrics: list[MetricResult]  # ROUGE/BLEU/METEOR scores; written by reference

    # ── Output ────────────────────────────────────────────────────────────
    cost_actual_usd: float         # Actual LLM spend (currently always 0.0 — TODO: wire up token tracking)
    report: Optional[EvalReport]   # Final aggregated report; written by eval, returned by run_graph and written to --output-dir
    error: Optional[str]           # Pipeline error message; checked by run_graph to raise RuntimeError (unused today)


# ── Node implementations ───────────────────────────────────────────────────


@observe(name="node_scan")
def node_metadata_scanner(state: EvalState) -> dict:
    gen = state["generation"]
    has_context = any(
        isinstance(item, str) and item.strip() for item in (state.get("context") or [])
    )
    meta = scan_text(
        gen,
        reference=state.get("reference"),
        has_context=has_context,
        has_rubric=bool(state.get("rubric")),
    )
    return {"metadata": meta}


@observe(name="node_estimate")
def node_cost_estimator(state: EvalState) -> dict:
    estimate = cost_estimator.estimate(
        state["metadata"],
        state["job_config"],
        target_node=state.get("target_node"),
        rubric_rule_count=len(state.get("rubric") or []),
    )
    if estimate.approximate_warning:
        _log_cost.warning(estimate.approximate_warning)

    return {"cost_estimate": estimate}


@observe(name="node_approve")
def node_confirm_gate(state: EvalState) -> dict:
    # In API mode, confirmation is handled via the request payload (acknowledge=True).
    # In CLI mode, the CLI layer prompts interactively before calling run_graph().
    # This node is a passthrough — confirmation is expected before graph execution.
    _log_confirm.info("Passthrough — confirmation already handled by caller")
    return {"confirmed": True}


@observe(name="node_chunk")
def node_chunk(state: EvalState) -> dict:
    if not state.get("context"):
        return {"chunks": []}
    chunks = chunk_text(state["generation"])
    return {"chunks": chunks}


@observe(name="node_claims")
def node_claims(state: EvalState) -> dict:
    if not state.get("context") or not state.get("chunks"):
        return {"raw_claims": []}
    model = state["job_config"].judge_model
    claims = claim_extractor.ClaimExtractorNode(model=model).run(state["chunks"])
    return {"raw_claims": claims}


@observe(name="node_dedupe")
def node_dedupe(state: EvalState) -> dict:
    raw = state["raw_claims"]
    if not raw:
        return {"unique_claims": []}
    unique, dedup_map = deduplicate(raw)
    return {"unique_claims": unique}


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
    if not claims or not state.get("context"):
        return {"grounding_metrics": []}
    if not state["job_config"].enable_grounding:
        return {"grounding_metrics": []}
    model = get_judge_model("grounding", state["job_config"].judge_model)
    results = GroundingNode(judge_model=model).run(
        claims=claims,
        context=state["context"],
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


@observe(name="node_rubric")
def node_rubric(state: EvalState) -> dict:
    if not state["job_config"].enable_rubric or not state["rubric"]:
        return {"rubric_metrics": []}
    rules = state["rubric"]
    model = get_judge_model("rubric", state["job_config"].judge_model)
    results = RubricNode(judge_model=model).run(
        generation=state["generation"],
        rubric=rules,
    )
    return {"rubric_metrics": results}


@observe(name="node_reference")
def node_reference(state: EvalState) -> dict:
    if not state["job_config"].enable_reference:
        return {"reference_metrics": []}
    reference = state.get("reference")
    if not reference:
        return {"reference_metrics": []}
    results = ReferenceMetricsNode().run(
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
        rubric_metrics=state.get("rubric_metrics") or [],
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
    g.add_node("estimate", node_cost_estimator)
    g.add_node("approve", node_confirm_gate)
    g.add_node("chunk", node_chunk)
    g.add_node("claims", node_claims)
    g.add_node("dedupe", node_dedupe)
    g.add_node("relevance", node_relevance)
    g.add_node("grounding", node_grounding)
    g.add_node("redteam", node_adversarial)
    g.add_node("rubric", node_rubric)
    g.add_node("reference", node_reference)
    g.add_node("eval", node_eval)

    g.set_entry_point("scan")
    g.add_edge("scan", "estimate")
    g.add_edge("estimate", "approve")
    g.add_edge("approve", "chunk")
    g.add_edge("chunk", "claims")
    g.add_edge("claims", "dedupe")
    g.add_edge("dedupe", "relevance")
    g.add_edge("dedupe", "grounding")
    g.add_edge("approve", "redteam")
    g.add_edge("approve", "rubric")
    g.add_edge("approve", "reference")
    g.add_edge("relevance", "eval")
    g.add_edge("grounding", "eval")
    g.add_edge("redteam", "eval")
    g.add_edge("rubric", "eval")
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
    rubric: Optional[list[Rubric]] = None,
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
        rubric=rubric or [],
        job_config=job_config,
        metadata=None,
        cost_estimate=None,
        confirmed=False,
        chunks=[],
        raw_claims=[],
        unique_claims=[],
        grounding_metrics=[],
        relevance_metrics=[],
        redteam_metrics=[],
        rubric_metrics=[],
        reference_metrics=[],
        cost_actual_usd=0.0,
        report=None,
        error=None,
    )


@observe(name="lumiseval_pipeline")
def run_graph(
    generation: str,
    job_config: Optional[EvalJobConfig] = None,
    question: Optional[str] = None,
    reference: Optional[str] = None,
    context: Optional[list[str]] = None,
    rubric: Optional[list[Rubric]] = None,
    reference_files: Optional[list[str]] = None,
) -> EvalReport:
    """Execute the full evaluation pipeline synchronously.

    Args:
        generation: The LLM-generated text to evaluate.
        job_config: Evaluation configuration. A default config is created if not provided.
        question: Optional query that produced the generation (improves RAGAS scores).
        reference: Optional reference answer (enables context recall).
        context: Optional retrieval context passages for retrieval-path metrics.
        rubric: Optional list of rubric rules to evaluate against.
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
        rubric=rubric,
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
    final_state = graph.invoke(initial_state)

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    report = final_state["report"]
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
