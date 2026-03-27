"""
Cost Estimator — computes a pre-run cost estimate before any LLM calls are made.

Uses tokencost for per-token pricing and presents a ±20% confidence band.
Blocks execution if the estimate exceeds the configured budget cap.

TODO: Track actual cost via LiteLLM response `usage` fields and log estimate vs. actual.
"""

from typing import Optional

from lumiseval_core.config import config
from lumiseval_core.errors import BudgetExceededError
from lumiseval_core.types import CostEstimate, EvalJobConfig, InputMetadata
from tokencost import calculate_prompt_cost

from lumiseval_agent.log import get_node_logger

log = get_node_logger("cost_estimator")

# Tavily price per search call (approximate, 2024 free tier = $0.001 each)
_TAVILY_PER_CALL_USD = 0.001
# Average tokens per judge call (claim verification prompt + response)
_AVG_JUDGE_TOKENS = 250
# Fraction of claims expected to need web search
_WEB_SEARCH_CLAIM_FRACTION = 0.4
# Embedding tokens per chunk (average)
_AVG_EMBEDDING_TOKENS = 400


def estimate(
    metadata: InputMetadata,
    job_config: EvalJobConfig,
) -> CostEstimate:
    """Compute a pre-run cost estimate.

    Args:
        metadata: Output of the Metadata Scanner.
        job_config: The evaluation job configuration.

    Returns:
        CostEstimate with USD breakdowns and ±20% band.

    Raises:
        BudgetExceededError: If total estimate exceeds job_config.budget_cap_usd.
    """
    model = job_config.judge_model
    approximate = False
    approximate_warning: Optional[str] = None

    estimated_judge_calls = metadata.estimated_claim_count
    estimated_embedding_calls = metadata.estimated_chunk_count
    estimated_tavily_calls = 0
    if job_config.web_search:
        estimated_tavily_calls = int(metadata.estimated_claim_count * _WEB_SEARCH_CLAIM_FRACTION)

    # Judge call cost
    # try:
    # tokencost expects a message list; we approximate with a dummy prompt
    sample_prompt = [{"role": "user", "content": "x" * _AVG_JUDGE_TOKENS}]
    cost_per_call = float(calculate_prompt_cost(sample_prompt, model))
    log.info(f"Pricing: ${cost_per_call:.6f}/call for {model}")
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

    log.info(
        f"judge=${judge_cost_usd:.6f}  embedding=${embedding_cost_usd:.6f}"
        f"  tavily=${tavily_cost_usd:.6f}  total=${total:.6f}"
    )

    if cap is not None and total > cap:
        log.error(f"Estimated cost ${total:.4f} exceeds budget cap ${cap:.4f}")
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
        low_usd=round(total * 0.8, 6),
        high_usd=round(total * 1.2, 6),
        approximate=approximate,
        approximate_warning=approximate_warning,
    )
