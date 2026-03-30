"""
Observability helpers for LumisEval.

Provides Langfuse tracing utilities — all soft-imported so the package works
without langfuse installed.  Install the optional extra to activate tracing:

    uv pip install "lumiseval-graph[observability]"

Env vars (all optional):
    LANGFUSE_SECRET_KEY   — enable tracing when set
    LANGFUSE_PUBLIC_KEY   — required alongside SECRET_KEY
    LANGFUSE_HOST         — defaults to https://cloud.langfuse.com

Usage in graph nodes:
    from lumiseval_graph.observability import observe

    @observe(name="node_claims")
    def node_claims(state):
        ...

Usage after an LLM call:
    from lumiseval_graph.observability import log_llm_usage

    response = structured_llm.invoke(messages)
    log_llm_usage(response)   # no-op when langfuse is absent
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Soft-import Langfuse ─────────────────────────────────────────────────────

_langfuse_enabled = False
_langfuse_client = None
observe: Any = None  # will be replaced below

try:
    if os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client  # noqa: E402
        from langfuse import observe as _lf_observe

        _langfuse_client = get_client()
        observe = _lf_observe
        _langfuse_enabled = True
        logger.debug("Langfuse tracing enabled")
    else:
        logger.debug("LANGFUSE_SECRET_KEY not set — tracing disabled")
except ImportError:
    logger.debug("langfuse package not installed — tracing disabled")


def _noop_decorator(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    as_type: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """No-op stand-in for @observe when Langfuse is unavailable."""
    if func is not None:
        return func

    def _wrapper(f: Callable) -> Callable:
        return f

    return _wrapper


if observe is None:
    observe = _noop_decorator


# ── Helpers ──────────────────────────────────────────────────────────────────


def is_enabled() -> bool:
    """Return True if Langfuse tracing is active."""
    return _langfuse_enabled


def log_llm_usage(response: Dict[str, Any]) -> None:
    """
    Log token usage from a gateway response to the current Langfuse generation.

    Must be called inside a function decorated with ``@observe(as_type="generation")``.
    Safe to call when Langfuse is disabled — becomes a no-op.

    Args:
        response: Dict returned by ``StructuredLLM.invoke()`` — must have a
                  ``"usage"`` key with ``prompt_tokens``, ``completion_tokens``,
                  ``total_tokens`` and a ``"model"`` key.
    """
    if not _langfuse_enabled or _langfuse_client is None:
        return

    usage = response.get("usage", {})
    model = response.get("model", "")
    if not usage or not model:
        return

    usage_details: Dict[str, int] = {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }

    _langfuse_client.update_current_generation(
        model=model,
        usage_details=usage_details,
    )


def update_trace(
    input: Optional[Dict[str, Any]] = None,
    output: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Update the current Langfuse trace. No-op when tracing is disabled."""
    if not _langfuse_enabled or _langfuse_client is None:
        return

    kwargs: Dict[str, Any] = {}
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if tags is not None:
        kwargs["tags"] = tags
    if metadata is not None:
        kwargs["metadata"] = metadata

    _langfuse_client.update_current_trace(**kwargs)


def score_trace(name: str, value: float, comment: Optional[str] = None) -> None:
    """Add a named score to the current Langfuse trace. No-op when disabled."""
    if not _langfuse_enabled or _langfuse_client is None:
        return

    kwargs: Dict[str, Any] = {"name": name, "value": value}
    if comment is not None:
        kwargs["comment"] = comment

    _langfuse_client.score_current_trace(**kwargs)
