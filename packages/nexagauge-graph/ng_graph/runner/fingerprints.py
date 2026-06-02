from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ng_core.aliases import resolve_alias
from ng_core.cache import build_node_cache_key, compute_case_hash
from ng_core.config import config as cfg

from ng_graph.llm.config import get_node_config, normalize_api_base
from ng_graph.llm.host_model import resolve_host_model_identity
from ng_graph.nodes.scanner import scan


def _case_value(case: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a case whether it's a dict or an attribute-bearing object.

    The CLI feeds dicts loaded from JSON; tests and programmatic callers
    occasionally pass Pydantic models. One accessor keeps the rest of this
    module agnostic to that.
    """
    if isinstance(case, dict):
        return case.get(key, default)
    return getattr(case, key, default)


def _case_id(case: Any) -> str:
    """Return a non-empty, stripped case id, deriving one from content if missing.

    When ``case_id`` is absent we hash the case content (same fields as
    :func:`_compute_case_fingerprint`) so identical cases collapse to the same
    id across runs — preserving cache reuse — while distinct cases stay
    distinct.
    """
    value = resolve_alias(case, "case_id", "")
    text = str(value).strip()
    if text:
        return text
    return f"case-{_compute_case_fingerprint(case)[:16]}"


def _stable_json(obj: Any) -> str:
    """Deterministic JSON encoding for hashing (sorted keys, compact, Pydantic-aware).

    Pydantic models are unwrapped via ``model_dump``/``dict`` so two
    structurally-equal states always produce byte-identical output — the whole
    basis of the node-level cache.
    """

    def _default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return str(value)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _inputs_from_case(case: Any) -> Any | None:
    if isinstance(case, Mapping):
        return case.get("inputs")
    return getattr(case, "inputs", None)


def _input_field(inputs: Any, key: str) -> Any:
    if isinstance(inputs, Mapping):
        return inputs.get(key)
    return getattr(inputs, key, None)


def _item_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return str(value.get("text", "") or "")
    return str(getattr(value, "text", "") or "")


def _compute_case_fingerprint(case: dict[str, Any]) -> str:
    """Hash the *input* of a case — the root of every cache key for that case.

    Excludes run-time concerns (target node, execution_mode, model routing);
    those enter via :func:`_step_fingerprint` further down. The delegate lives
    in ``ng_core.cache`` so core and graph packages agree on the hash.

    Inputs are first run through :func:`scan` so the fingerprint sees the
    same normalized field values as the rest of the graph. This keeps the
    case-hash contract aligned with the typed ``Inputs`` produced by the
    scanner and avoids type-mismatch crashes when raw record fields (e.g.
    a scalar ``context: 42``) reach the cache layer.
    """
    inputs = _inputs_from_case(case)
    if inputs is None:
        inputs = scan(case)["inputs"]

    output_item = _input_field(inputs, "output")
    input_item = _input_field(inputs, "input")
    reference_item = _input_field(inputs, "reference")
    context_item = _input_field(inputs, "context")
    geval_cfg = _input_field(inputs, "geval")
    redteam_cfg = _input_field(inputs, "redteam")
    grounding_cfg = _input_field(inputs, "grounding")
    relevance_cfg = _input_field(inputs, "relevance")
    refalign_cfg = _input_field(inputs, "refalign")
    context_text = _item_text(context_item)
    return compute_case_hash(
        output=_item_text(output_item),
        input=_item_text(input_item) or None,
        reference=_item_text(reference_item) or None,
        geval=geval_cfg,
        redteam=redteam_cfg,
        grounding=grounding_cfg,
        relevance=relevance_cfg,
        refalign=refalign_cfg,
        context=[context_text] if context_text else [],
        reference_files=_case_value(case, "reference_files") or [],
    )


def _node_route_fingerprint(
    node_name: str, *, state: Mapping[str, Any], execution_mode: str
) -> str:
    """Hash the LLM routing for a node — model, fallback, temperature, mode.

    This is the "did the model selection change?" component of a cache key.
    Swapping ``gpt-4o-mini`` for ``gpt-4o`` (or flipping ``run``↔``estimate``)
    must invalidate cached outputs; changing an unrelated node's routing must
    not. Called from :func:`_step_fingerprint`.
    """
    llm_overrides = state.get("llm_overrides")
    node_cfg = get_node_config(node_name, llm_overrides=llm_overrides)
    resolved_model = node_cfg.model or cfg.LLM_MODEL
    api_base = normalize_api_base(node_cfg.api_base)
    payload = {
        "execution_mode": execution_mode,
        "node": node_name,
        "model": resolved_model,
        "fallback_model": node_cfg.fallback_model,
        "temperature": node_cfg.temperature,
        "api_base": api_base,
        "chunker": state.get("chunker"),
        "refiner": state.get("refiner"),
        "refiner_top_k": state.get("refiner_top_k"),
    }
    # Self-hosted OpenAI-compatible endpoints (e.g. llama.cpp / vLLM on
    # localhost) may reuse the same URL for whatever model is currently loaded.
    # The static ``model`` field doesn't distinguish them, so probe the
    # server's /v1/models endpoint for an identity tag. Best-effort: None on
    # failure leaves the fingerprint unchanged.
    if api_base:
        payload["host_served_model"] = resolve_host_model_identity(api_base)
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:16]


def _step_fingerprint(
    *,
    parent_fingerprint: str,
    node_name: str,
    state: Mapping[str, Any],
    execution_mode: str,
) -> str:
    """Chain ``parent_fingerprint`` with this node's identity + routing.

    Produces the path-dependent fingerprint used to build this step's cache
    key. Chaining guarantees that a node's cache entry is only valid when the
    *entire upstream path* that produced its inputs is identical.
    """
    route_fingerprint = _node_route_fingerprint(
        node_name,
        state=state,
        execution_mode=execution_mode,
    )
    raw = f"{parent_fingerprint}|{node_name}|{route_fingerprint}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_key_for_step(
    *,
    case_fingerprint: str,
    node_name: str,
    step_fingerprint: str,
    execution_mode: str,
) -> str:
    """Assemble the opaque backend cache key for a single step execution.

    Delegates to ``ng_core.cache.build_node_cache_key`` so the exact key format
    stays in one place and is backend-agnostic.
    """
    return build_node_cache_key(
        case_fingerprint=case_fingerprint,
        node_name=node_name,
        execution_mode=execution_mode,
        node_route_fingerprint=step_fingerprint,
    )


def _cache_namespace_mode(*, execution_mode: str) -> str:
    """Return the cache namespace mode for this execution.

    ``estimate`` runs never write cache by default and should reuse entries
    produced by ``run``; therefore estimate cache reads are looked up in the
    ``run`` namespace.
    """
    if execution_mode == "estimate":
        return "run"
    return execution_mode


def _step_fingerprint_for_node_in_plan(
    *,
    case_fingerprint: str,
    node_name: str,
    state: Mapping[str, Any],
    execution_mode: str,
    plan_transitive_prereqs: Mapping[str, tuple[str, ...]],
) -> str:
    """Compute a node fingerprint from ancestors present in the active plan."""
    parent_fingerprint = case_fingerprint
    for prereq in plan_transitive_prereqs.get(node_name, ()):
        parent_fingerprint = _step_fingerprint(
            parent_fingerprint=parent_fingerprint,
            node_name=prereq,
            state=state,
            execution_mode=execution_mode,
        )
    return _step_fingerprint(
        parent_fingerprint=parent_fingerprint,
        node_name=node_name,
        state=state,
        execution_mode=execution_mode,
    )
