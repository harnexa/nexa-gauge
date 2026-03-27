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
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from lumiseval_core.config import config as cfg
from lumiseval_core.types import (
    Chunk,
    Claim,
    ClaimVerdict,
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
from lumiseval_evidence.indexer import index_file
from lumiseval_evidence.mmr import deduplicate
from lumiseval_evidence.router import route
from lumiseval_ingest.chunker import chunk_text
from lumiseval_ingest.scanner import scan_text

from .log import get_node_logger, print_pipeline_footer, print_pipeline_header
from .nodes import aggregation, claim_extractor, cost_estimator
from .nodes.metrics import deepeval_node, giskard_node, ragas_node, rubric_node

logger = logging.getLogger(__name__)

# Module-level node loggers — one per pipeline node
_log_scanner = get_node_logger("metadata_scanner")
_log_cost = get_node_logger("cost_estimator")
_log_confirm = get_node_logger("confirm_gate")
_log_chunker = get_node_logger("chunker")
_log_claims = get_node_logger("claim_extractor")
_log_mmr = get_node_logger("mmr_deduplicator")
_log_evidence = get_node_logger("evidence_router")
_log_ragas = get_node_logger("ragas")
_log_deepeval = get_node_logger("deepeval")
_log_giskard = get_node_logger("giskard")
_log_rubric = get_node_logger("rubric_eval")
_log_agg = get_node_logger("aggregation")


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
    gen = state["generation"]
    _log_scanner.start(f"Scanning generation ({len(gen):,} chars)")
    meta = scan_text(gen)
    _log_scanner.success(
        f"{meta.total_tokens:,} tokens  ·  "
        f"~{meta.estimated_chunk_count} chunk(s)  ·  "
        f"~{meta.estimated_claim_count} claim(s) estimated"
    )
    return {"metadata": meta}


def node_cost_estimator(state: EvalState) -> dict:
    model = state["job_config"].judge_model
    _log_cost.start(f"Estimating cost  (model={model})")
    # try:
    estimate = cost_estimator.estimate(state["metadata"], state["job_config"])
    if estimate.approximate_warning:
        _log_cost.warning(estimate.approximate_warning)
    _log_cost.success(
        f"total=${estimate.total_estimated_usd:.6f}"
        f"  (low=${estimate.low_usd:.6f}  /  high=${estimate.high_usd:.6f})"
        f"  ·  judge calls={estimate.estimated_judge_calls}"
        f"  ·  tavily calls={estimate.estimated_tavily_calls}"
    )
    return {"cost_estimate": estimate}
    # except Exception as exc:
    #     _log_cost.error(str(exc))
    #     return {"error": str(exc)}


def node_confirm_gate(state: EvalState) -> dict:
    # In API mode, confirmation is handled via the request payload (acknowledge=True).
    # In CLI mode, the CLI layer prompts interactively before calling run_graph().
    # This node is a passthrough — confirmation is expected before graph execution.
    _log_confirm.info("Passthrough — confirmation already handled by caller")
    return {"confirmed": True}


def node_chunker(state: EvalState) -> dict:
    _log_chunker.start("Chunking text  (target 512 tokens/chunk)")
    chunks = chunk_text(state["generation"])
    _log_chunker.success(f"{len(chunks)} chunk(s) produced")
    for c in chunks:
        preview = c.text[:80].replace("\n", " ")
        _log_chunker.info(f'  chunk {c.index}  [{c.char_start}:{c.char_end}]  "{preview}…"')
    return {"chunks": chunks}


def node_claim_extractor(state: EvalState) -> dict:
    n = len(state["chunks"])
    model = state["job_config"].judge_model
    _log_claims.start(f"Extracting claims from {n} chunk(s)  (model={model})")
    claims = claim_extractor.extract_claims(
        state["chunks"],
        model=model,
    )
    _log_claims.success(f"{len(claims)} claim(s) extracted")
    return {"raw_claims": claims}


def node_mmr_deduplicator(state: EvalState) -> dict:
    raw = state["raw_claims"]
    _log_mmr.start(f"Deduplicating {len(raw)} claim(s)  (similarity threshold=0.9)")
    unique, dedup_map = deduplicate(raw)
    removed = len(raw) - len(unique)
    _log_mmr.success(f"{len(unique)} unique claim(s)  ·  {removed} duplicate(s) removed")
    return {"unique_claims": unique}


def node_evidence_router(state: EvalState) -> dict:
    cfg = state["job_config"]
    claims = state["unique_claims"]
    web = cfg.web_search
    threshold = cfg.evidence_threshold
    _log_evidence.start(
        f"Routing evidence for {len(claims)} claim(s)"
        f"  (web_search={'on' if web else 'off'}  threshold={threshold})"
    )

    results = []
    for i, claim in enumerate(claims, 1):
        preview = claim.text[:70].replace("\n", " ")
        _log_evidence.info(f'  [{i}/{len(claims)}] "{preview}…"')
        result = route(claim, web_search=web, threshold=threshold)
        results.append(result)
        _log_evidence.info(
            f"         → verdict={result.verdict.value}  "
            f"source={result.source.value}  "
            f"passages={len(result.passages)}"
        )

    supported = sum(1 for r in results if r.verdict == ClaimVerdict.SUPPORTED)
    contradicted = sum(1 for r in results if r.verdict == ClaimVerdict.CONTRADICTED)
    unverifiable = sum(1 for r in results if r.verdict == ClaimVerdict.UNVERIFIABLE)
    _log_evidence.success(
        f"SUPPORTED={supported}  CONTRADICTED={contradicted}  UNVERIFIABLE={unverifiable}"
    )
    return {"evidence_results": results}


def node_ragas(state: EvalState) -> dict:
    if not state["job_config"].enable_ragas:
        _log_ragas.info("Skipped  (enable_ragas=False)")
        return {"ragas_result": None}
    _log_ragas.start("Running RAGAS metrics  (faithfulness, answer_relevancy)")
    result = ragas_node.run(
        generation=state["generation"],
        evidence_results=state["evidence_results"],
        question=state.get("question"),
        ground_truth=state.get("ground_truth"),
        judge_model=state["job_config"].judge_model,
    )
    if result.error:
        _log_ragas.error(f"RAGAS failed: {result.error}")
    else:
        _log_ragas.success(
            f"faithfulness={result.faithfulness}  answer_relevancy={result.answer_relevancy}"
        )
    return {"ragas_result": result}


def node_deepeval(state: EvalState) -> dict:
    if not state["job_config"].enable_deepeval:
        _log_deepeval.info("Skipped  (enable_deepeval=False)")
        return {"deepeval_result": None}
    adversarial = state["job_config"].adversarial
    rubric = state["rubric_rules"] if state["job_config"].enable_rubric_eval else None
    _log_deepeval.start(
        f"Running DeepEval metrics"
        f"  (adversarial={'on' if adversarial else 'off'}"
        f"  rubric={'yes' if rubric else 'no'})"
    )
    result = deepeval_node.run(
        generation=state["generation"],
        evidence_results=state["evidence_results"],
        rubric_rules=rubric,
        adversarial=adversarial,
        judge_model=state["job_config"].judge_model,
    )
    if result.error:
        _log_deepeval.error(f"DeepEval failed: {result.error}")
    else:
        parts = [f"hallucination={result.hallucination_score}"]
        if result.g_eval_score is not None:
            parts.append(f"g_eval={result.g_eval_score}")
        if result.privacy_score is not None:
            parts.append(f"privacy={result.privacy_score}")
        if result.bias_score is not None:
            parts.append(f"bias={result.bias_score}")
        _log_deepeval.success("  ".join(parts))
    return {"deepeval_result": result}


def node_giskard(state: EvalState) -> dict:
    if not state["job_config"].enable_giskard or not state["job_config"].adversarial:
        _log_giskard.info("Skipped  (enable_giskard=False or adversarial=False)")
        return {"giskard_result": None}
    _log_giskard.start("Running Giskard adversarial probes")
    result = giskard_node.run(generation=state["generation"])
    if not result.giskard_available:
        _log_giskard.warning("Giskard not installed — adversarial coverage reduced")
    elif result.error:
        _log_giskard.error(f"Giskard scan failed: {result.error}")
    else:
        n = len(result.vulnerabilities)
        _log_giskard.success(f"{n} vulnerability/vulnerabilities found")
        for v in result.vulnerabilities:
            _log_giskard.info(f"  [{v.severity.value}] {v.probe_type}: {v.description[:80]}")
    return {"giskard_result": result}


def node_rubric_eval(state: EvalState) -> dict:
    if not state["job_config"].enable_rubric_eval or not state["rubric_rules"]:
        _log_rubric.info("Skipped  (enable_rubric_eval=False or no rubric rules)")
        return {"rubric_result": None}
    rules = state["rubric_rules"]
    _log_rubric.start(f"Evaluating {len(rules)} rubric rule(s) in parallel")
    result = rubric_node.run(
        generation=state["generation"],
        rubric_rules=rules,
        judge_model=state["job_config"].judge_model,
    )
    _log_rubric.success(
        f"compliance_rate={result.compliance_rate:.2f}"
        f"  composite_adherence={result.composite_adherence_score:.4f}"
    )
    for r in result.rule_results:
        _log_rubric.info(f"  [{r.compliance.value}] {r.rule_id}  score={r.confidence:.3f}")
    return {"rubric_result": result}


def node_aggregation(state: EvalState) -> dict:
    _log_agg.start("Aggregating all metric scores")
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
    score_str = f"{report.composite_score:.4f}" if report.composite_score is not None else "n/a"
    band_str = f"{report.confidence_band:.4f}" if report.confidence_band is not None else "n/a"
    _log_agg.success(
        f"composite_score={score_str}  confidence_band=±{band_str}  warnings={len(report.warnings)}"
    )
    for w in report.warnings:
        _log_agg.warning(w)
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

    if job_config is None:
        job_config = EvalJobConfig(
            job_id=str(uuid.uuid4()),
            judge_model=cfg.LLM_MODEL,
            web_search=cfg.WEB_SEARCH_ENABLED,
            evidence_threshold=cfg.EVIDENCE_THRESHOLD,
        )

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

    report = final_state["report"]
    print_pipeline_footer(
        composite_score=report.composite_score,
        cost_usd=report.cost_actual_usd,
    )
    return report
