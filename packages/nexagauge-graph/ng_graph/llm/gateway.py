"""
LLM Gateway — structured LLM calls via LiteLLM with fallback and token tracking.

Wraps ``litellm.completion`` so every call:
  - Picks the right model (env var override → job default)
  - Retries on the fallback model if the primary fails
  - Returns a unified dict with the parsed Pydantic object + token usage

Usage:
    from ng_graph.llm import get_llm

    llm = get_llm("claims", _ClaimList, default_model="gpt-4o-mini")
    response = llm.invoke([{"role": "user", "content": "..."}])
    result   = response["parsed"]   # _ClaimList instance
    tokens   = response["usage"]    # {"prompt_tokens": int, "completion_tokens": int, ...}
"""

from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar

import litellm
from ng_core.constants import LLM_CALL_TIMEOUT_SECONDS
from pydantic import BaseModel

from ng_graph.llm.config import get_node_config, normalize_node_name

litellm.set_verbose = False

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

    def _call_with_logprobs(
        self, messages: List[Dict[str, str]], model: str, top_logprobs: int
    ) -> Any:
        return litellm.completion(
            model=model,
            messages=messages,
            temperature=self.temperature,
            response_format=self.schema,
            logprobs=True,
            top_logprobs=top_logprobs,
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

    def invoke_with_logprobs(
        self, messages: List[Dict[str, str]], *, top_logprobs: int = 20
    ) -> Dict[str, Any]:
        """Structured call that also returns per-token logprobs.

        Same return shape as :meth:`invoke`, plus a ``logprobs`` key containing a
        list of ``{token, logprob, top_logprobs: [{token, logprob}, ...]}`` dicts
        — or ``None`` when the provider can't return logprobs (e.g. Ollama).

        Three-tier fallback:
          1. primary model with logprobs,
          2. fallback model with logprobs (covers quota / auth failures),
          3. primary model *without* logprobs (covers feature-unsupported).
        """
        logprobs_supported = True
        used_model = self.model
        try:
            response = self._call_with_logprobs(messages, self.model, top_logprobs)
        except Exception:
            try:
                if not self.fallback_model:
                    raise
                response = self._call_with_logprobs(messages, self.fallback_model, top_logprobs)
                used_model = self.fallback_model
            except Exception:
                response = self._call(messages, self.model)
                logprobs_supported = False

        choice = response.choices[0]
        content = choice.message.content

        parsing_error: Optional[Exception] = None
        parsed = None
        try:
            parsed = self.schema.model_validate_json(content)
        except Exception as exc:
            parsing_error = exc

        usage = response.usage
        logprobs_payload = _extract_logprobs(choice) if logprobs_supported else None

        return {
            "parsed": parsed,
            "parsing_error": parsing_error,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "model": used_model,
            "logprobs": logprobs_payload,
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


def _extract_logprobs(choice: Any) -> Optional[List[Dict[str, Any]]]:
    """Flatten LiteLLM's logprobs object into provider-agnostic dicts.

    Returns ``None`` when the provider didn't return logprobs so downstream
    scoring can fall back to the raw integer score.
    """
    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None:
        return None
    content = getattr(logprobs_obj, "content", None)
    if not content:
        return None

    out: List[Dict[str, Any]] = []
    for tok in content:
        top = [
            {
                "token": getattr(alt, "token", ""),
                "logprob": getattr(alt, "logprob", 0.0),
            }
            for alt in (getattr(tok, "top_logprobs", None) or [])
        ]
        out.append(
            {
                "token": getattr(tok, "token", ""),
                "logprob": getattr(tok, "logprob", 0.0),
                "top_logprobs": top,
            }
        )
    return out
