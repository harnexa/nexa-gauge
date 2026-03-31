"""
Cost Estimator — computes a pre-run cost estimate before any LLM calls are made.

Uses tokencost for per-token pricing and presents a ±20% confidence band.
Blocks execution if the estimate exceeds the configured budget cap.

TODO: Track actual cost via LiteLLM response `usage` fields and log estimate vs. actual.
"""

from typing import Optional

from lumiseval_core.config import config
from lumiseval_core.constants import (
    COST_AVG_CONTEXT_TOKENS,
    COST_AVG_EMBEDDING_TOKENS,
    COST_AVG_QUESTION_TOKENS,
    COST_ESTIMATE_BAND_HIGH,
    COST_ESTIMATE_BAND_LOW,
    COST_FALLBACK_PER_CALL_USD,
    COST_TAVILY_PER_CALL_USD,
    COST_WEB_SEARCH_CLAIM_FRACTION,
)
from lumiseval_core.errors import BudgetExceededError, ClaimExtractionError
from lumiseval_core.pipeline import NODE_PREREQUISITES, normalize_node_name
from lumiseval_core.types import CostEstimate, EvalJobConfig, InputMetadata, NodeCostBreakdown

from lumiseval_graph.llm.config import get_judge_model
from lumiseval_graph.log import get_node_logger

from lumiseval_graph.nodes.claim_extractor import ClaimExtractorNode
from lumiseval_graph.nodes.metrics.grounding import GroundingNode
from lumiseval_graph.nodes.metrics.redteam import RedteamNode
from lumiseval_graph.nodes.metrics.relevance import RelevanceNode
from lumiseval_graph.nodes.metrics.rubric import RubricNode

log = get_node_logger("estimate")

_TAVILY_PER_CALL_USD = COST_TAVILY_PER_CALL_USD
_WEB_SEARCH_CLAIM_FRACTION = COST_WEB_SEARCH_CLAIM_FRACTION
_AVG_EMBEDDING_TOKENS = COST_AVG_EMBEDDING_TOKENS

# Map pipeline node names to their BaseMetricNode subclass
_NODE_CLASSES = {
    "claims": ClaimExtractorNode,
    "grounding": GroundingNode,
    "relevance": RelevanceNode,
    "redteam": RedteamNode,
    "rubric": RubricNode,
}


def _estimate_call_counts(
    *,
    metadata: InputMetadata,
    job_config: EvalJobConfig,
    target_node: str,
    rubric_rule_count: int,
) -> tuple[dict[str, int], int, int]:
    """Estimate judge call counts per node, plus embedding and Tavily totals.

    Returns:
        (node_judge_calls, estimated_embedding_calls, estimated_tavily_calls)
        where node_judge_calls maps node name → number of judge API calls.
    """
    records = metadata.record_count
    # Claims are extracted from generation chunks; fall back to total if not set
    chunks = metadata.generation_chunk_count or metadata.estimated_chunk_count
    claims = metadata.estimated_claim_count
    eligible_records = metadata.eligible_record_count or {}
    eligible_chunks = metadata.eligible_chunk_count or {}
    eligible_claims = metadata.eligible_claim_count or {}

    target_plan = set(NODE_PREREQUISITES.get(target_node, [])) | {target_node}
    needs_claim_path = (
        bool(target_plan & {"chunk", "claims", "dedupe", "relevance", "grounding"})
        or target_node == "estimate"
    )
    needs_relevance = "relevance" in target_plan
    needs_grounding = "grounding" in target_plan and job_config.enable_hallucination
    needs_redteam = "redteam" in target_plan and job_config.enable_adversarial
    needs_rubric = "rubric" in target_plan and job_config.enable_rubric

    claim_path_chunks = eligible_chunks.get("claims", chunks)
    relevance_records = eligible_records.get("relevance", records)
    grounding_records = eligible_records.get("grounding", records)
    redteam_records = eligible_records.get("redteam", records)

    # node_judge_calls: dict[str, int] = {
    #     "claims": claim_path_chunks if needs_claim_path else 0,
    #     "relevance": (
    #         relevance_records
    #         if needs_relevance
    #         and (job_config.enable_faithfulness or job_config.enable_answer_relevancy)
    #         else 0
    #     ),
    #     "grounding": grounding_records if needs_grounding else 0,
    #     "redteam": redteam_records if needs_redteam else 0,
    #     "rubric": rubric_rule_count if needs_rubric else 0,
    # }

    # estimated_embedding_calls = claim_path_chunks if needs_claim_path else 0

    # retrieval_path_claims = eligible_claims.get("retrieve", claims)
    # estimated_tavily_calls = (
    #     int(retrieval_path_claims * _WEB_SEARCH_CLAIM_FRACTION)
    #     if job_config.web_search and needs_claim_path
    #     else 0
    # )

    return node_judge_calls


def _build_cost_kwargs(metadata: InputMetadata, rubric_rule_count: int) -> dict:
    """Derive keyword arguments for node cost_estimate() calls from InputMetadata."""
    records = metadata.record_count or 1
    eligible_claims = metadata.eligible_claim_count or {}
    eligible_records = metadata.eligible_record_count or {}

    # Average claims per record for the claim-based nodes (relevance, grounding)
    rel_records = eligible_records.get("relevance", records) or 1
    total_rel_claims = eligible_claims.get("relevance", metadata.estimated_claim_count)
    avg_claims_per_record = total_rel_claims / rel_records

    return {
        "avg_claims_per_record": avg_claims_per_record,
        "avg_context_tokens": COST_AVG_CONTEXT_TOKENS,
        "avg_question_tokens": COST_AVG_QUESTION_TOKENS,
        "rubric_rule_count": rubric_rule_count,
    }


def _compute_node_breakdown(
    *,
    node_judge_calls: dict[str, int],
    metadata: InputMetadata,
    job_config: EvalJobConfig,
    rubric_rule_count: int,
) -> dict[str, NodeCostBreakdown]:
    """Build per-node NodeCostBreakdown by calling each metric node's cost_estimate().

    For non-metric nodes (e.g. "claims") that have no BaseMetricNode subclass,
    a simple fallback is used: judge_calls × COST_FALLBACK_PER_CALL_USD.
    """
    cost_kwargs = _build_cost_kwargs(metadata, rubric_rule_count)
    eligible_records = metadata.eligible_record_count or {}
    records = metadata.record_count

    breakdown: dict[str, NodeCostBreakdown] = {}

    for node_name, n_calls in node_judge_calls.items():
        if n_calls == 0:
            breakdown[node_name] = NodeCostBreakdown(judge_calls=0, cost_usd=0.0)
            continue

        node_cls = _NODE_CLASSES.get(node_name)
        if node_cls is None:
            # Non-metric node (e.g. "claims") — flat fallback
            breakdown[node_name] = NodeCostBreakdown(
                judge_calls=n_calls,
                cost_usd=round(n_calls * COST_FALLBACK_PER_CALL_USD, 6),
            )
            continue

        effective_model = get_judge_model(node_name, job_config.judge_model)
        node_instance = node_cls(judge_model=effective_model)
        node_eligible = eligible_records.get(node_name, records)

        breakdown[node_name] = node_instance.cost_estimate(
            eligible_records=node_eligible,
            **cost_kwargs,
        )

    return breakdown


def estimate(
    metadata: InputMetadata,
    job_config: EvalJobConfig,
    *,
    target_node: Optional[str] = None,
    rubric_rule_count: Optional[int] = None,
) -> CostEstimate:
    """Compute a pre-run cost estimate.

    Args:
        metadata: Output of the Metadata Scanner.
        job_config: The evaluation job configuration.
        target_node: Node target for strict-target runs. Defaults to full `eval`.
        rubric_rule_count: Total rubric rule evaluations expected for this run.

    Returns:
        CostEstimate with USD breakdowns and ±20% band.

    Raises:
        BudgetExceededError: If total estimate exceeds job_config.budget_cap_usd.
    """

    normalized_target = normalize_node_name(target_node) if target_node else "eval"
    approximate = False
    approximate_warning: Optional[str] = None
    # _rubric_rule_count = rubric_rule_count or 0

    print("metadata: ", metadata)
    print(1/0)

    # node_judge_calls, estimated_embedding_calls, estimated_tavily_calls = _estimate_call_counts(
    #     metadata=metadata,
    #     job_config=job_config,
    #     target_node=normalized_target,
    #     rubric_rule_count=_rubric_rule_count,
    # )
    # estimated_judge_calls = sum(node_judge_calls.values())

    # # ── Per-node judge cost via metric node cost_estimate() ───────────────────
    # node_breakdown = _compute_node_breakdown(
    #     node_judge_calls=node_judge_calls,
    #     metadata=metadata,
    #     job_config=job_config,
    #     rubric_rule_count=_rubric_rule_count,
    # )

    # judge_cost_usd = sum(b.cost_usd for b in node_breakdown.values())

    # # Embedding cost (sentence-transformers are local — no API cost today)
    # embedding_cost_usd = 0.0

    # # Tavily cost
    # tavily_cost_usd = estimated_tavily_calls * _TAVILY_PER_CALL_USD

    # total = judge_cost_usd + embedding_cost_usd + tavily_cost_usd
    # cap = job_config.budget_cap_usd or config.BUDGET_CAP_USD

    # if cap is not None and total > cap:
    #     raise BudgetExceededError(
    #         f"Estimated cost ${total:.4f} exceeds budget cap ${cap:.4f}. "
    #         "Adjust config (disable web search, reduce batch size, or increase budget cap) "
    #         "before running."
    #     )

    # return CostEstimate(
    #     estimated_judge_calls=estimated_judge_calls,
    #     estimated_embedding_calls=estimated_embedding_calls,
    #     estimated_tavily_calls=estimated_tavily_calls,
    #     judge_cost_usd=round(judge_cost_usd, 6),
    #     embedding_cost_usd=round(embedding_cost_usd, 6),
    #     tavily_cost_usd=round(tavily_cost_usd, 6),
    #     total_estimated_usd=round(total, 6),
    #     low_usd=round(total * COST_ESTIMATE_BAND_LOW, 6),
    #     high_usd=round(total * COST_ESTIMATE_BAND_HIGH, 6),
    #     approximate=approximate,
    #     approximate_warning=approximate_warning,
    #     node_breakdown=node_breakdown,
    # )
