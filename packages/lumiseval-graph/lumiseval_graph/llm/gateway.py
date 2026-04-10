"""
LLM Gateway — structured LLM calls via LiteLLM with fallback and token tracking.

Wraps ``litellm.completion`` so every call:
  - Picks the right model (env var override → job default)
  - Retries on the fallback model if the primary fails
  - Returns a unified dict with the parsed Pydantic object + token usage

Usage:
    from lumiseval_graph.llm import get_llm

    llm = get_llm("claims", _ClaimList, default_model="gpt-4o-mini")
    response = llm.invoke([{"role": "user", "content": "..."}])
    result   = response["parsed"]   # _ClaimList instance
    tokens   = response["usage"]    # {"prompt_tokens": int, "completion_tokens": int, ...}
"""

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar

import litellm
from lumiseval_core.constants import LLM_CALL_TIMEOUT_SECONDS
from pydantic import BaseModel

from lumiseval_graph.llm.config import get_node_config, normalize_node_name

logger = logging.getLogger(__name__)

litellm.set_verbose = False

# ── Automatic LiteLLM → Langfuse callback ────────────────────────────────────
# Enabled when LANGFUSE_SECRET_KEY is present.  All litellm.completion() calls
# are automatically forwarded to Langfuse for cost and latency tracking.
if os.getenv("LANGFUSE_SECRET_KEY"):
    litellm.success_callback = ["langfuse"]
    logger.debug("LiteLLM → Langfuse success callback registered")

T = TypeVar("T", bound=BaseModel)

# ── Module-level instance cache ─────────────────────────────────────────────
_cache: Dict[str, "StructuredLLM"] = {}


class StructuredLLM:
    """
    LiteLLM wrapper that returns structured Pydantic output plus token usage.

    Return format:
        {
            "parsed":        Pydantic instance (or None on parsing failure),
            "parsing_error": Exception or None,
            "usage":         {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            "model":         str,   # actual model that responded
        }
    """

    def __init__(
        self,
        *,
        node_name: str,
        schema: Type[T],
        model: str,
        temperature: float,
        fallback_model: Optional[str],
    ):
        self.node_name = node_name
        self.schema = schema
        self.model = model
        self.temperature = temperature
        self.fallback_model = fallback_model

    # ── Internal helpers ───────────────────────────────────────────────────

    def _call(self, messages: List[Dict[str, str]], model: str) -> Any:
        return litellm.completion(
            model=model,
            messages=messages,
            temperature=self.temperature,
            response_format=self.schema,
            timeout=LLM_CALL_TIMEOUT_SECONDS,
            metadata={"node_name": self.node_name},
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call the LLM and return structured output.

        Args:
            messages: LiteLLM-format message list,
                      e.g. ``[{"role": "user", "content": "..."}]``.

        Returns:
            Dict with keys ``parsed``, ``parsing_error``, ``usage``, ``model``.
        """
        try:
            response = self._call(messages, self.model)
            used_model = self.model
        except Exception:
            if self.fallback_model:
                response = self._call(messages, self.fallback_model)
                used_model = self.fallback_model
            else:
                raise

        choice = response.choices[0]
        content = choice.message.content

        parsing_error: Optional[Exception] = None
        parsed = None
        try:
            parsed = self.schema.model_validate_json(content)
        except Exception as exc:
            parsing_error = exc

        usage = response.usage
        return {
            "parsed": parsed,
            "parsing_error": parsing_error,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "model": used_model,
        }


# ── Public factory ──────────────────────────────────────────────────────────


def get_llm(
    node_name: str,
    schema: Type[T],
    default_model: str,
    llm_overrides: Optional[Mapping[str, Any]] = None,
) -> StructuredLLM:
    """
    Return a cached StructuredLLM for ``node_name``.

    Args:
        node_name:     Node identifier (e.g. ``"claims"``).
        schema:        Pydantic model class for structured output.
        default_model: Job-level judge model used when no env var overrides the node.
        llm_overrides: Optional per-run overrides for model/fallback/temperature.

    Returns:
        StructuredLLM — shared instance per (node_name, schema, default_model) triple.
    """
    canonical_node_name = normalize_node_name(node_name, strict=True)
    cfg = get_node_config(canonical_node_name, llm_overrides=llm_overrides)
    resolved_model = cfg.model or default_model
    key = (
        f"{canonical_node_name}:{schema.__name__}:"
        f"{resolved_model}:{cfg.fallback_model or ''}:{cfg.temperature}"
    )
    if key not in _cache:
        _cache[key] = StructuredLLM(
            node_name=canonical_node_name,
            schema=schema,
            model=resolved_model,
            temperature=cfg.temperature,
            fallback_model=cfg.fallback_model,
        )
    return _cache[key]
