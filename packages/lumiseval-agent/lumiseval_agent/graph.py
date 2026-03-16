"""
LangGraph Orchestration Graph — the core evaluation pipeline.

Node sequence:
  metadata_scanner → cost_estimator → [user confirm gate] → chunker →
  claim_extractor → mmr_deduplicator → evidence_router →
  [parallel: ragas_node, deepeval_node, giskard_node, rubric_node] →
  aggregation → result

TODO:
  - Implement async TaskIQ dispatch for batch jobs.
  - Stream progress to CLI/API consumers.
  - Persist EvalReport to SQLite via SQLModel.
"""

import logging
import uuid
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from lumiseval_core.types import (
    Chunk,
    Claim,
    CostEstimate,
    DeepEvalMetricResult,
    EvalJobConfig,
    EvalReport,
    EvidenceResult,
    GiskardScanResult,
    InputMetadata,
    RAGASMetricResult,
    RubricEvalResult,
    RubricRule,
)
from lumiseval_ingest.chunker import chunk_text
from lumiseval_ingest.scanner import scan_text
from lumiseval_evidence.mmr import deduplicate
from lumiseval_evidence.router import route

from .nodes import aggregation
from .nodes import claim_extractor
from .nodes import cost_estimator
from .nodes.metrics import deepeval_node, giskard_node, ragas_node, rubric_node

logger = logging.getLogger(__name__)


# ── Graph state ────────────────────────────────────────────────────────────

class EvalState(TypedDict):
    generation: str
    question: Optional[str]
    ground_truth: Optional[str]
    reference_files: list[str]
    rubric_rules: list[RubricRule]
    job_config: EvalJobConfig
    # Populated as nodes run
    metadata: Optional[InputMetadata]
    cost_estimate: Optional[CostEstimate]
    confirmed: bool
    chunks: list[Chunk]
    raw_claims: list[Claim]
    unique_claims: list[Claim]
    evidence_results: list[EvidenceResult]
    ragas_result: Optional[RAGASMetricResult]
    deepeval_result: Optional[DeepEvalMetricResult]
    giskard_result: Optional[GiskardScanResult]
    rubric_result: Optional[RubricEvalResult]
    cost_actual_usd: float
    report: Optional[EvalReport]
    error: Optional[str]


# ── Node implementations ───────────────────────────────────────────────────

def node_metadata_scanner(state: EvalState) -> dict:
    meta = scan_text(state["generation"])
    return {"metadata": meta}


def node_cost_estimator(state: EvalState) -> dict:
    try:
        estimate = cost_estimator.estimate(state["metadata"], state["job_config"])
        return {"cost_estimate": estimate}
    except Exception as exc:
        return {"error": str(exc)}


def node_confirm_gate(state: EvalState) -> dict:
    # In API mode, confirmation is handled via the request payload (acknowledge=True).
    # In CLI mode, the CLI layer prompts interactively before calling run_graph().
    # This node is a passthrough — confirmation is expected before graph execution.
    return {"confirmed": True}


def node_chunker(state: EvalState) -> dict:
    chunks = chunk_text(state["generation"])
    return {"chunks": chunks}


def node_claim_extractor(state: EvalState) -> dict:
    claims = claim_extractor.extract_claims(
        state["chunks"],
        model=state["job_config"].judge_model,
    )
    return {"raw_claims": claims}


def node_mmr_deduplicator(state: EvalState) -> dict:
    unique, _ = deduplicate(state["raw_claims"])
    return {"unique_claims": unique}


def node_evidence_router(state: EvalState) -> dict:
    cfg = state["job_config"]
    results = [
        route(
            claim,
            web_search=cfg.web_search,
            threshold=cfg.evidence_threshold,
        )
        for claim in state["unique_claims"]
    ]
    return {"evidence_results": results}


def node_ragas(state: EvalState) -> dict:
    if not state["job_config"].enable_ragas:
        return {"ragas_result": None}
    result = ragas_node.run(
        generation=state["generation"],
        evidence_results=state["evidence_results"],
        question=state.get("question"),
        ground_truth=state.get("ground_truth"),
        judge_model=state["job_config"].judge_model,
    )
    return {"ragas_result": result}


def node_deepeval(state: EvalState) -> dict:
    if not state["job_config"].enable_deepeval:
        return {"deepeval_result": None}
    result = deepeval_node.run(
        generation=state["generation"],
        evidence_results=state["evidence_results"],
        rubric_rules=state["rubric_rules"] if state["job_config"].enable_rubric_eval else None,
        adversarial=state["job_config"].adversarial,
        judge_model=state["job_config"].judge_model,
    )
    return {"deepeval_result": result}


def node_giskard(state: EvalState) -> dict:
    if not state["job_config"].enable_giskard or not state["job_config"].adversarial:
        return {"giskard_result": None}
    result = giskard_node.run(generation=state["generation"])
    return {"giskard_result": result}


def node_rubric_eval(state: EvalState) -> dict:
    if not state["job_config"].enable_rubric_eval or not state["rubric_rules"]:
        return {"rubric_result": None}
    result = rubric_node.run(
        generation=state["generation"],
        rubric_rules=state["rubric_rules"],
        judge_model=state["job_config"].judge_model,
    )
    return {"rubric_result": result}


def node_aggregation(state: EvalState) -> dict:
    report = aggregation.aggregate(
        job_id=state["job_config"].job_id,
        claim_verdicts=state["evidence_results"],
        ragas=state.get("ragas_result"),
        deepeval=state.get("deepeval_result"),
        giskard=state.get("giskard_result"),
        rubric=state.get("rubric_result"),
        cost_estimate=state.get("cost_estimate"),
        cost_actual_usd=state.get("cost_actual_usd", 0.0),
        job_config=state["job_config"],
    )
    return {"report": report}


# ── Graph construction ─────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(EvalState)

    g.add_node("metadata_scanner", node_metadata_scanner)
    g.add_node("cost_estimator", node_cost_estimator)
    g.add_node("confirm_gate", node_confirm_gate)
    g.add_node("chunker", node_chunker)
    g.add_node("claim_extractor", node_claim_extractor)
    g.add_node("mmr_deduplicator", node_mmr_deduplicator)
    g.add_node("evidence_router", node_evidence_router)
    g.add_node("ragas", node_ragas)
    g.add_node("deepeval", node_deepeval)
    g.add_node("giskard", node_giskard)
    g.add_node("rubric_eval", node_rubric_eval)
    g.add_node("aggregation", node_aggregation)

    g.set_entry_point("metadata_scanner")
    g.add_edge("metadata_scanner", "cost_estimator")
    g.add_edge("cost_estimator", "confirm_gate")
    g.add_edge("confirm_gate", "chunker")
    g.add_edge("chunker", "claim_extractor")
    g.add_edge("claim_extractor", "mmr_deduplicator")
    g.add_edge("mmr_deduplicator", "evidence_router")
    g.add_edge("evidence_router", "ragas")
    g.add_edge("evidence_router", "deepeval")
    g.add_edge("evidence_router", "giskard")
    g.add_edge("evidence_router", "rubric_eval")
    g.add_edge("ragas", "aggregation")
    g.add_edge("deepeval", "aggregation")
    g.add_edge("giskard", "aggregation")
    g.add_edge("rubric_eval", "aggregation")
    g.add_edge("aggregation", END)

    return g


def run_graph(
    generation: str,
    job_config: Optional[EvalJobConfig] = None,
    question: Optional[str] = None,
    ground_truth: Optional[str] = None,
    rubric_rules: Optional[list[RubricRule]] = None,
    reference_files: Optional[list[str]] = None,
) -> EvalReport:
    """Execute the full evaluation pipeline synchronously.

    Args:
        generation: The LLM-generated text to evaluate.
        job_config: Evaluation configuration. A default config is created if not provided.
        question: Optional query that produced the generation (improves RAGAS scores).
        ground_truth: Optional reference answer (enables context recall).
        rubric_rules: Optional list of rubric rules to evaluate against.
        reference_files: Optional list of file paths to index before evaluation.

    Returns:
        EvalReport with all available metric scores and per-claim verdicts.
    """
    from lumiseval_core.config import config as cfg

    if job_config is None:
        job_config = EvalJobConfig(
            job_id=str(uuid.uuid4()),
            judge_model=cfg.LLM_MODEL,
            web_search=cfg.WEB_SEARCH_ENABLED,
            evidence_threshold=cfg.EVIDENCE_THRESHOLD,
        )

    # Index reference files into local LanceDB before graph runs
    if reference_files:
        from lumiseval_evidence.indexer import index_file

        for fpath in reference_files:
            try:
                count = index_file(fpath)
                logger.info("Indexed %d passages from %s", count, fpath)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", fpath, exc)

    initial_state = EvalState(
        generation=generation,
        question=question,
        ground_truth=ground_truth,
        reference_files=reference_files or [],
        rubric_rules=rubric_rules or [],
        job_config=job_config,
        metadata=None,
        cost_estimate=None,
        confirmed=False,
        chunks=[],
        raw_claims=[],
        unique_claims=[],
        evidence_results=[],
        ragas_result=None,
        deepeval_result=None,
        giskard_result=None,
        rubric_result=None,
        cost_actual_usd=0.0,
        report=None,
        error=None,
    )

    graph = build_graph().compile()
    final_state = graph.invoke(initial_state)

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    return final_state["report"]
