"""
Cost Estimator — computes a pre-run cost estimate before any LLM calls are made.

Uses tokencost for per-token pricing and presents a ±20% confidence band.
Blocks execution if the estimate exceeds the configured budget cap.

TODO: Track actual cost via LiteLLM response `usage` fields and log estimate vs. actual.
"""

from typing import Optional

from lumiseval_core.config import config
from lumiseval_core.constants import (
    COST_AVG_EMBEDDING_TOKENS,
    COST_AVG_JUDGE_TOKENS,
    COST_ESTIMATE_BAND_HIGH,
    COST_ESTIMATE_BAND_LOW,
    COST_TAVILY_PER_CALL_USD,
    COST_WEB_SEARCH_CLAIM_FRACTION,
)
from lumiseval_core.errors import BudgetExceededError
from lumiseval_core.types import CostEstimate, EvalJobConfig, InputMetadata
from tokencost import calculate_prompt_cost

from lumiseval_agent.log import get_node_logger

log = get_node_logger("estimate")

_TAVILY_PER_CALL_USD = COST_TAVILY_PER_CALL_USD
_AVG_JUDGE_TOKENS = COST_AVG_JUDGE_TOKENS
_WEB_SEARCH_CLAIM_FRACTION = COST_WEB_SEARCH_CLAIM_FRACTION
_AVG_EMBEDDING_TOKENS = COST_AVG_EMBEDDING_TOKENS

_LEGACY_NODE_ALIASES = {
    "metadata_scanner": "scan",
    "cost_estimator": "estimate",
    "confirm_gate": "approve",
    "chunker": "chunk",
    "claim_extractor": "claims",
    "mmr_deduplicator": "dedupe",
    "ragas": "relevance",
    "hallucination": "grounding",
    "adversarial": "redteam",
    "eval": "eval",
}


def _normalize_target_node(node_name: Optional[str]) -> str:
    if not node_name:
        return "eval"
    return _LEGACY_NODE_ALIASES.get(node_name, node_name)


def _estimate_call_counts(
    *,
    metadata: InputMetadata,
    job_config: EvalJobConfig,
    target_node: str,
    rubric_rule_count: int,
) -> tuple[int, int, int]:
    """Estimate judge/embedding/tavily call counts for a specific target node."""
    records = metadata.record_count
    chunks = metadata.estimated_chunk_count
    claims = metadata.estimated_claim_count
    eligible_records = metadata.eligible_record_count or {}
    eligible_chunks = metadata.eligible_chunk_count or {}
    eligible_claims = metadata.eligible_claim_count or {}

    needs_claim_path = target_node in {
        "estimate",
        "claims",
        "dedupe",
        "relevance",
        "grounding",
        "eval",
    }  # "retrieve",
    # needs_retrieval_path = target_node in {"retrieve", "relevance", "grounding", "eval"}
    needs_relevance = target_node in {"relevance", "eval"}
    needs_grounding = target_node in {"grounding", "eval"} and job_config.enable_hallucination
    needs_redteam = target_node in {"redteam", "eval"} and job_config.enable_adversarial
    needs_rubric = target_node in {"rubric", "eval"} and job_config.enable_rubric

    claim_path_chunks = eligible_chunks.get("claims", chunks)
    # retrieval_path_claims = eligible_claims.get("retrieve", claims)
    relevance_records = eligible_records.get("relevance", records)
    grounding_records = eligible_records.get("grounding", records)
    redteam_records = eligible_records.get("redteam", records)

    claim_extraction_calls = claim_path_chunks if needs_claim_path else 0
    relevance_calls = (
        relevance_records
        if needs_relevance
        and (job_config.enable_faithfulness or job_config.enable_answer_relevancy)
        else 0
    )
    grounding_calls = grounding_records if needs_grounding else 0
    redteam_calls = redteam_records if needs_redteam else 0
    rubric_calls = rubric_rule_count if needs_rubric else 0

    estimated_judge_calls = (
        claim_extraction_calls + relevance_calls + grounding_calls + redteam_calls + rubric_calls
    )
    estimated_embedding_calls = claim_path_chunks if needs_claim_path else 0

    retrieval_path_claims = eligible_claims.get("retrieve", claims)
    estimated_tavily_calls = (
        int(retrieval_path_claims * _WEB_SEARCH_CLAIM_FRACTION)
        if job_config.web_search and needs_claim_path
        else 0
    )

    return estimated_judge_calls, estimated_embedding_calls, estimated_tavily_calls


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

    model = job_config.judge_model
    normalized_target = _normalize_target_node(target_node)
    # print("normalized_targetnormalized_targetnormalized_target: ", normalized_target)
    # print(1/0)
    approximate = False
    approximate_warning: Optional[str] = None

    estimated_judge_calls, estimated_embedding_calls, estimated_tavily_calls = (
        _estimate_call_counts(
            metadata=metadata,
            job_config=job_config,
            target_node=normalized_target,
            rubric_rule_count=rubric_rule_count or 0,
        )
    )

    # Judge call cost
    # try:
    # tokencost expects a message list; we approximate with a dummy prompt
    sample_prompt = [{"role": "user", "content": "x" * _AVG_JUDGE_TOKENS}]
    cost_per_call = float(calculate_prompt_cost(sample_prompt, model))
    # log.info(f"Pricing: ${cost_per_call:.6f}/call for {model}")
    # except Exception:
    #     # Fall back to gpt-4o-mini pricing if model not found
    #     try:
    #         sample_prompt = [{"role": "user", "content": "x" * _AVG_JUDGE_TOKENS}]
    #         cost_per_call = float(calculate_prompt_cost(sample_prompt, "gpt-4o-mini"))
    #         approximate = True
    #         approximate_warning = (
    #             f"Model '{model}' not found in tokencost pricing tables. "
    #             "Using gpt-4o-mini pricing as approximation."
    #         )
    #         log.warning(approximate_warning)
    #     except Exception:
    #         cost_per_call = 0.0003  # conservative fallback
    #         approximate = True
    #         approximate_warning = "Could not compute pricing; using $0.0003/call fallback."
    #         log.warning(approximate_warning)

    judge_cost_usd = estimated_judge_calls * cost_per_call

    # Embedding cost (sentence-transformers are local — no API cost)
    embedding_cost_usd = 0.0

    # Tavily cost
    tavily_cost_usd = estimated_tavily_calls * _TAVILY_PER_CALL_USD

    total = judge_cost_usd + embedding_cost_usd + tavily_cost_usd
    cap = job_config.budget_cap_usd or config.BUDGET_CAP_USD

    # log.info(
    #     f"judge=${judge_cost_usd:.6f}  embedding=${embedding_cost_usd:.6f}"
    #     f"  tavily=${tavily_cost_usd:.6f}  total=${total:.6f}"
    #     f"  target={normalized_target}"
    # )

    if cap is not None and total > cap:
        # log.error(f"Estimated cost ${total:.4f} exceeds budget cap ${cap:.4f}")
        raise BudgetExceededError(
            f"Estimated cost ${total:.4f} exceeds budget cap ${cap:.4f}. "
            "Adjust config (disable web search, reduce batch size, or increase budget cap) "
            "before running."
        )

    return CostEstimate(
        estimated_judge_calls=estimated_judge_calls,
        estimated_embedding_calls=estimated_embedding_calls,
        estimated_tavily_calls=estimated_tavily_calls,
        judge_cost_usd=round(judge_cost_usd, 6),
        embedding_cost_usd=round(embedding_cost_usd, 6),
        tavily_cost_usd=round(tavily_cost_usd, 6),
        total_estimated_usd=round(total, 6),
        low_usd=round(total * COST_ESTIMATE_BAND_LOW, 6),
        high_usd=round(total * COST_ESTIMATE_BAND_HIGH, 6),
        approximate=approximate,
        approximate_warning=approximate_warning,
    )
