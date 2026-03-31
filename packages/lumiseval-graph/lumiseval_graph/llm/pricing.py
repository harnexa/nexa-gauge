"""
Model pricing registry for per-token cost estimation.

Provides a curated MODEL_PRICING dict mapping model names → ModelPricing.
Lookup order in get_model_pricing():
  1. Our registry (exact match after stripping provider prefix)
  2. tokencost library (covers many more models, kept up-to-date by LiteLLM)
  3. Flat-rate fallback derived from COST_FALLBACK_PER_CALL_USD

To add a new model or update a price: edit MODEL_PRICING below.
Prices are USD per 1,000 tokens (input / output) as of 2025-Q1.
"""

from dataclasses import dataclass

import tokencost
from lumiseval_core.constants import COST_AVG_JUDGE_TOKENS, COST_FALLBACK_PER_CALL_USD


@dataclass(frozen=True)
class ModelPricing:
    """USD cost per 1,000 tokens for a single model."""

    input_per_1k: float  # USD per 1k input (prompt) tokens
    output_per_1k: float  # USD per 1k output (completion) tokens


# ── Curated pricing registry ──────────────────────────────────────────────────
# Keys are bare model names (no provider prefix). get_model_pricing() strips
# "openai/", "anthropic/", etc. before looking up here.

MODEL_PRICING: dict[str, ModelPricing] = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o": ModelPricing(input_per_1k=0.002500, output_per_1k=0.010000),
    "gpt-4o-mini": ModelPricing(input_per_1k=0.000150, output_per_1k=0.000600),
    "gpt-4-turbo": ModelPricing(input_per_1k=0.010000, output_per_1k=0.030000),
    "gpt-4": ModelPricing(input_per_1k=0.030000, output_per_1k=0.060000),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.000500, output_per_1k=0.001500),
    "o1": ModelPricing(input_per_1k=0.015000, output_per_1k=0.060000),
    "o1-mini": ModelPricing(input_per_1k=0.003000, output_per_1k=0.012000),
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-opus-4-6": ModelPricing(input_per_1k=0.015000, output_per_1k=0.075000),
    "claude-sonnet-4-6": ModelPricing(input_per_1k=0.003000, output_per_1k=0.015000),
    "claude-3-5-sonnet-20241022": ModelPricing(input_per_1k=0.003000, output_per_1k=0.015000),
    "claude-3-5-haiku-20241022": ModelPricing(input_per_1k=0.001000, output_per_1k=0.005000),
    "claude-3-haiku-20240307": ModelPricing(input_per_1k=0.000250, output_per_1k=0.001250),
    "claude-3-opus-20240229": ModelPricing(input_per_1k=0.015000, output_per_1k=0.075000),
    # ── Google ────────────────────────────────────────────────────────────────
    "gemini-1.5-flash": ModelPricing(input_per_1k=0.000075, output_per_1k=0.000300),
    "gemini-1.5-pro": ModelPricing(input_per_1k=0.001250, output_per_1k=0.005000),
    "gemini-2.0-flash": ModelPricing(input_per_1k=0.000100, output_per_1k=0.000400),
}

# ── Lookup helpers ────────────────────────────────────────────────────────────

_FALLBACK_PRICING = ModelPricing(
    # Derive a per-1k rate from the conservative per-call fallback constant,
    # assuming COST_AVG_JUDGE_TOKENS tokens per call.
    input_per_1k=COST_FALLBACK_PER_CALL_USD / (COST_AVG_JUDGE_TOKENS / 1000),
    output_per_1k=COST_FALLBACK_PER_CALL_USD / (COST_AVG_JUDGE_TOKENS / 1000) * 3,
)


def get_model_pricing(model: str) -> ModelPricing:
    """Return pricing for *model*.

    Lookup order:
      1. MODEL_PRICING (curated, provider prefix stripped)
      2. tokencost (LiteLLM-maintained pricing tables)
      3. _FALLBACK_PRICING (conservative constant)

    Args:
        model: Model name, with or without a provider prefix
               (e.g. ``"gpt-4o-mini"`` or ``"openai/gpt-4o-mini"``).
    """
    # Strip provider prefix: "openai/gpt-4o-mini" → "gpt-4o-mini"
    key = model.split("/")[-1] if "/" in model else model

    if key in MODEL_PRICING:
        return MODEL_PRICING[key]

    # tokencost fallback — calculate per-1k rates from a 1k-token dummy call
    try:
        inp = float(tokencost.calculate_cost_by_tokens(1000, model, "input"))
        out = float(tokencost.calculate_cost_by_tokens(1000, model, "output"))
        return ModelPricing(input_per_1k=inp, output_per_1k=out)
    except Exception:
        return _FALLBACK_PRICING


def cost_usd(n_tokens: int, pricing: ModelPricing, token_type: str) -> float:
    """Compute USD cost for *n_tokens* at the given *pricing*.

    Args:
        n_tokens:   Number of tokens.
        pricing:    ModelPricing instance (from get_model_pricing).
        token_type: ``"input"`` or ``"output"``.
    """
    rate = pricing.input_per_1k if token_type == "input" else pricing.output_per_1k
    return (n_tokens / 1000.0) * rate
