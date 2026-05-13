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

from threading import BoundedSemaphore, Lock
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar
from urllib.parse import urlparse

import litellm
from ng_core.constants import LLM_CALL_TIMEOUT_SECONDS
from pydantic import BaseModel

from ng_graph.llm.config import get_node_config, normalize_node_name

litellm.set_verbose = False

T = TypeVar("T", bound=BaseModel)

# ── Module-level instance cache ─────────────────────────────────────────────
_cache: Dict[str, "StructuredLLM"] = {}
_LLM_CONCURRENCY_LOCK = Lock()
_LLM_CONCURRENCY = 16
_LLM_SEMAPHORE = BoundedSemaphore(_LLM_CONCURRENCY)


def set_llm_concurrency(limit: int) -> None:
    """Set process-wide max number of concurrent LLM calls."""
    if limit < 1:
        raise ValueError("llm concurrency must be >= 1")

    global _LLM_CONCURRENCY, _LLM_SEMAPHORE
    with _LLM_CONCURRENCY_LOCK:
        _LLM_CONCURRENCY = int(limit)
        _LLM_SEMAPHORE = BoundedSemaphore(_LLM_CONCURRENCY)


def get_llm_concurrency() -> int:
    """Return process-wide LLM concurrency limit."""
    with _LLM_CONCURRENCY_LOCK:
        return _LLM_CONCURRENCY


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
        api_base: Optional[str],
        api_key: Optional[str],
    ):
        self.node_name = node_name
        self.schema = schema
        self.model = model
        self.temperature = temperature
        self.fallback_model = fallback_model
        self.api_base = api_base
        self.api_key = _resolve_api_key(api_base=api_base, api_key=api_key)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _call(self, messages: List[Dict[str, str]], model: str) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": self.schema,
            "timeout": LLM_CALL_TIMEOUT_SECONDS,
            "metadata": {"node_name": self.node_name},
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return litellm.completion(**kwargs)

    def _call_unstructured(self, messages: List[Dict[str, str]], model: str) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": LLM_CALL_TIMEOUT_SECONDS,
            "metadata": {"node_name": self.node_name},
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return litellm.completion(**kwargs)

    def _call_with_logprobs(
        self, messages: List[Dict[str, str]], model: str, top_logprobs: int
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": self.schema,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "timeout": LLM_CALL_TIMEOUT_SECONDS,
            "metadata": {"node_name": self.node_name},
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return litellm.completion(**kwargs)

    def _call_with_logprobs_unstructured(
        self, messages: List[Dict[str, str]], model: str, top_logprobs: int
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "timeout": LLM_CALL_TIMEOUT_SECONDS,
            "metadata": {"node_name": self.node_name},
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return litellm.completion(**kwargs)

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
        with _LLM_SEMAPHORE:
            try:
                response = self._invoke_with_schema_fallback(messages, self.model)
                used_model = self.model
            except Exception:
                # Primary model failed (quota, auth, network, etc.); retry on fallback if configured.
                if self.fallback_model:
                    response = self._invoke_with_schema_fallback(messages, self.fallback_model)
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
        with _LLM_SEMAPHORE:
            try:
                response = self._invoke_with_schema_fallback(
                    messages,
                    self.model,
                    with_logprobs=True,
                    top_logprobs=top_logprobs,
                )
            except Exception:
                # Primary logprobs call failed — try fallback, then retry without logprobs.
                try:
                    if not self.fallback_model:
                        raise
                    response = self._invoke_with_schema_fallback(
                        messages,
                        self.fallback_model,
                        with_logprobs=True,
                        top_logprobs=top_logprobs,
                    )
                    used_model = self.fallback_model
                except Exception:
                    # Provider likely doesn't support logprobs (e.g. Ollama) — degrade gracefully.
                    response = self._invoke_with_schema_fallback(messages, self.model)
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

    def _invoke_with_schema_fallback(
        self,
        messages: List[Dict[str, str]],
        model: str,
        *,
        with_logprobs: bool = False,
        top_logprobs: int = 20,
    ) -> Any:
        """Call LiteLLM with schema output; retry without schema when unsupported."""
        try:
            if with_logprobs:
                return self._call_with_logprobs(messages, model, top_logprobs)
            return self._call(messages, model)
        except Exception as exc:
            if not _is_response_format_unsupported_error(exc):
                raise
            if with_logprobs:
                return self._call_with_logprobs_unstructured(messages, model, top_logprobs)
            return self._call_unstructured(messages, model)


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
        f"{resolved_model}:{cfg.fallback_model or ''}:{cfg.temperature}:{cfg.api_base or ''}"
    )
    if key not in _cache:
        _cache[key] = StructuredLLM(
            node_name=canonical_node_name,
            schema=schema,
            model=resolved_model,
            temperature=cfg.temperature,
            fallback_model=cfg.fallback_model,
            api_base=cfg.api_base,
            api_key=cfg.api_key,
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


def _is_response_format_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "response_format",
        "json_schema",
        "schema",
        "not supported",
        "unsupported",
        "invalid parameter",
    )
    if "response_format" not in message and "json_schema" not in message:
        return False
    return any(marker in message for marker in markers)


def _is_local_api_base(api_base: Optional[str]) -> bool:
    if not api_base:
        return False
    parsed = urlparse(api_base)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _resolve_api_key(*, api_base: Optional[str], api_key: Optional[str]) -> Optional[str]:
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    if _is_local_api_base(api_base):
        return "local"
    return None
