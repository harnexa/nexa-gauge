"""
LLM Gateway Configuration — per-node model routing via environment variables.

Every node can be overridden independently, enabling per-node model selection,
fallback routing, and temperature tuning without touching code.

Environment variable convention (all optional):
  LLM_{NODE}_MODEL           — primary model (default: job's judge_model)
  LLM_{NODE}_FALLBACK_MODEL  — model to use if primary call fails
  LLM_{NODE}_TEMPERATURE     — float temperature (default: 0.0)

Where {NODE} is the node name uppercased, with spaces/hyphens → underscores.

Example .env:
  LLM_GROUNDING_MODEL=gpt-4o-mini
  LLM_GROUNDING_FALLBACK_MODEL=gpt-4o
  LLM_CLAIMS_MODEL=gpt-4o
  LLM_CLAIMS_FALLBACK_MODEL=gpt-4o-mini
"""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, TypedDict


@dataclass
class NodeModelConfig:
    """Resolved model config for a single evaluation node."""

    model: Optional[str]  # None = use job's judge_model
    temperature: float
    fallback_model: Optional[str]


class RuntimeLLMOverrides(TypedDict, total=False):
    """Optional per-run overrides applied before env/default routing."""

    models: dict[str, str]
    fallback_models: dict[str, str]
    temperatures: dict[str, float]


_KNOWN_NODES: frozenset[str] = frozenset({
    "claims", "chunk", "dedup", "relevance",
    "grounding", "reference", "redteam", "geval", "geval_steps",
})

_NODE_ALIASES: dict[str, str] = {
    f"{prefix}{n}": n
    for n in _KNOWN_NODES
    for prefix in ("node_", "generation_")
}


def normalize_node_name(node_name: str, strict: bool = False) -> str:
    """Normalize a node name to a consistent key (lowercase, underscores).

    Resolves aliases like ``"node_claims"`` or ``"generation_claims"`` to
    their canonical form ``"claims"`` via an explicit lookup table.
    """
    name = str(node_name).strip().lower().replace("-", "_").replace(" ", "_")
    return _NODE_ALIASES.get(name, name)


def normalize_runtime_overrides(
    llm_overrides: Optional[Mapping[str, Any]],
) -> RuntimeLLMOverrides:
    """Canonicalize runtime override dictionaries."""
    if not llm_overrides:
        return {}

    models: dict[str, str] = {}
    fallback_models: dict[str, str] = {}
    temperatures: dict[str, float] = {}

    raw_models = llm_overrides.get("models")
    if isinstance(raw_models, Mapping):
        for node_name, model in raw_models.items():
            if model is None:
                continue
            model_text = str(model).strip()
            if not model_text:
                continue
            models[normalize_node_name(node_name)] = model_text

    raw_fallbacks = llm_overrides.get("fallback_models")
    if isinstance(raw_fallbacks, Mapping):
        for node_name, model in raw_fallbacks.items():
            if model is None:
                continue
            model_text = str(model).strip()
            if not model_text:
                continue
            fallback_models[normalize_node_name(node_name)] = model_text

    raw_temps = llm_overrides.get("temperatures")
    if isinstance(raw_temps, Mapping):
        for node_name, temp in raw_temps.items():
            temperatures[normalize_node_name(node_name)] = float(temp)

    return RuntimeLLMOverrides(
        models=models,
        fallback_models=fallback_models,
        temperatures=temperatures,
    )


def _env_prefix(node_name: str) -> str:
    return normalize_node_name(node_name).upper()


def get_node_config(
    node_name: str,
    llm_overrides: Optional[Mapping[str, Any]] = None,
) -> NodeModelConfig:
    """
    Build NodeModelConfig for a node by reading env var overrides.

    Args:
        node_name: Node identifier, e.g. ``"claims"``.
        llm_overrides: Optional runtime override payload.

    Returns:
        NodeModelConfig — fields are None/defaults when the env var is absent.
    """
    key = _env_prefix(node_name)
    normalized = normalize_node_name(node_name)

    model = os.getenv(f"LLM_{key}_MODEL") or None
    fallback = os.getenv(f"LLM_{key}_FALLBACK_MODEL") or None
    temp_raw = os.getenv(f"LLM_{key}_TEMPERATURE")
    temperature = float(temp_raw) if temp_raw is not None else 0.0

    if llm_overrides:
        normalized_overrides = normalize_runtime_overrides(llm_overrides)
        model = normalized_overrides.get("models", {}).get(normalized, model)
        fallback = normalized_overrides.get("fallback_models", {}).get(normalized, fallback)
        if normalized in normalized_overrides.get("temperatures", {}):
            temperature = float(normalized_overrides["temperatures"][normalized])

    return NodeModelConfig(model=model, temperature=temperature, fallback_model=fallback)


def get_judge_model(
    node_name: str,
    default: str,
    llm_overrides: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Return the model to use for a node.

    Priority: runtime override → env var override → ``default``.

    Args:
        node_name: Node identifier.
        default: Fallback model string (typically ``job_config.judge_model``).
        llm_overrides: Optional runtime override payload.
    """
    cfg = get_node_config(node_name, llm_overrides=llm_overrides)
    return cfg.model or default
