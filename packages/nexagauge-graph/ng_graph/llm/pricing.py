"""
Model pricing registry for per-token cost estimation.

Provides a curated MODEL_PRICING dict mapping model names → ModelPricing.
Lookup order in get_model_pricing():
  1. Our registry (exact provider/model match)
  2. Our registry (bare-model compatibility fallback)
  3. tokencost library (covers many more models, kept up-to-date by LiteLLM)
  4. Flat-rate fallback derived from COST_FALLBACK_PER_CALL_USD

To add a new model or update a price: edit MODEL_PRICING below.
Prices are USD per 1,000 tokens (input / output), derived from current
published API pricing pages.
"""

from dataclasses import dataclass

import tokencost
from ng_core.constants import (
    COST_AVG_JUDGE_TOKENS,
    COST_FALLBACK_PER_CALL_USD,
    DEFAULT_LLM_PROVIDER,
)


@dataclass(frozen=True)
class ModelPricing:
    """USD cost per 1,000 tokens for a single model."""

    input_per_1k: float  # USD per 1k input (prompt) tokens
    output_per_1k: float  # USD per 1k output (completion) tokens


# ── Curated pricing registry ──────────────────────────────────────────────────
# Keys are explicit provider-qualified model names (provider/model).

MODEL_PRICING: dict[str, ModelPricing] = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    # - https://developers.openai.com/api/docs/models/gpt-4o
    # - https://developers.openai.com/api/docs/models/gpt-4o-mini
    # - https://developers.openai.com/api/docs/models/gpt-4-turbo
    # - https://developers.openai.com/api/docs/models/gpt-4
    # - https://developers.openai.com/api/docs/models/gpt-3.5-turbo
    # - https://developers.openai.com/api/docs/models/o1
    # - https://developers.openai.com/api/docs/models/o1-mini
    "openai/gpt-4o": ModelPricing(input_per_1k=0.002500, output_per_1k=0.010000),
    "openai/gpt-4o-mini": ModelPricing(input_per_1k=0.000150, output_per_1k=0.000600),
    "openai/gpt-4-turbo": ModelPricing(input_per_1k=0.010000, output_per_1k=0.030000),
    "openai/gpt-4": ModelPricing(input_per_1k=0.030000, output_per_1k=0.060000),
    "openai/gpt-3.5-turbo": ModelPricing(input_per_1k=0.000500, output_per_1k=0.001500),
    "openai/o1": ModelPricing(input_per_1k=0.015000, output_per_1k=0.060000),
    "openai/o1-mini": ModelPricing(input_per_1k=0.001100, output_per_1k=0.004400),
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "anthropic/claude-opus-4-6": ModelPricing(input_per_1k=0.015000, output_per_1k=0.075000),
    "anthropic/claude-sonnet-4-6": ModelPricing(input_per_1k=0.003000, output_per_1k=0.015000),
    "anthropic/claude-3-5-sonnet-20241022": ModelPricing(
        input_per_1k=0.003000, output_per_1k=0.015000
    ),
    "anthropic/claude-3-5-haiku-20241022": ModelPricing(
        input_per_1k=0.001000, output_per_1k=0.005000
    ),
    "anthropic/claude-3-haiku-20240307": ModelPricing(
        input_per_1k=0.000250, output_per_1k=0.001250
    ),
    "anthropic/claude-3-opus-20240229": ModelPricing(input_per_1k=0.015000, output_per_1k=0.075000),
    # ── Google ────────────────────────────────────────────────────────────────
    # Gemini Developer API standard text/image/video token pricing.
    # - https://ai.google.dev/gemini-api/docs/pricing
    "gemini/gemini-2.0-flash": ModelPricing(input_per_1k=0.000100, output_per_1k=0.000400),
    "gemini/gemini-2.0-flash-lite": ModelPricing(input_per_1k=0.000075, output_per_1k=0.000300),
    "gemini/gemini-2.5-flash": ModelPricing(input_per_1k=0.000300, output_per_1k=0.002500),
    "gemini/gemini-2.5-flash-lite": ModelPricing(input_per_1k=0.000100, output_per_1k=0.000400),
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
      1. MODEL_PRICING exact key match (provider/model)
      2. MODEL_PRICING bare-name compatibility match
      3. tokencost (LiteLLM-maintained pricing tables)
      4. _FALLBACK_PRICING (conservative constant)

    Args:
        model: Model name, with or without a provider prefix
               (e.g. ``"gpt-4o-mini"`` or ``"openai/gpt-4o-mini"``).
    """
    key = str(model).strip()

    if key in MODEL_PRICING:
        return MODEL_PRICING[key]

    # Backward compatibility for bare model names. If there is exactly one
    # provider-qualified match, use it. If ambiguous, prefer default provider.
    bare_key = key.split("/")[-1]
    matching_keys = [k for k in MODEL_PRICING if k.split("/")[-1] == bare_key]
    if len(matching_keys) == 1:
        return MODEL_PRICING[matching_keys[0]]
    preferred_key = f"{DEFAULT_LLM_PROVIDER}/{bare_key}"
    if preferred_key in MODEL_PRICING:
        return MODEL_PRICING[preferred_key]

    # tokencost fallback — calculate per-1k rates from a 1k-token dummy call
    try:
        inp = float(tokencost.calculate_cost_by_tokens(1000, model, "input"))
        out = float(tokencost.calculate_cost_by_tokens(1000, model, "output"))
        return ModelPricing(input_per_1k=inp, output_per_1k=out)
    except Exception:
        # tokencost lacks a pricing entry for this model (e.g. private deployment) — use fallback.
        return _FALLBACK_PRICING


def cost_usd(n_tokens: float, pricing: ModelPricing, token_type: str) -> float:
    """Compute USD cost for *n_tokens* at the given *pricing*.

    Args:
        n_tokens:   Number of tokens.
        pricing:    ModelPricing instance (from get_model_pricing).
        token_type: ``"input"`` or ``"output"``.
    """
    rate = pricing.input_per_1k if token_type == "input" else pricing.output_per_1k
    return (n_tokens / 1000.0) * rate
