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
    api_base: Optional[str]
    api_key: Optional[str]


class RuntimeLLMOverrides(TypedDict, total=False):
    """Optional per-run overrides applied before env/default routing."""

    models: dict[str, str]
    fallback_models: dict[str, str]
    temperatures: dict[str, float]
    api_bases: dict[str, str]
    api_keys: dict[str, str]


_KNOWN_NODES: frozenset[str] = frozenset(
    {
        "claims",
        "chunk",
        "refiner",
        "relevance",
        "grounding",
        "reference",
        "redteam",
        "geval",
        "geval_steps",
    }
)


def normalize_node_name(node_name: str, strict: bool = False) -> str:
    """Normalize a node name to a consistent key (lowercase, underscores)."""
    name = str(node_name).strip().lower().replace("-", "_").replace(" ", "_")
    if strict and name not in _KNOWN_NODES:
        valid = ", ".join(sorted(_KNOWN_NODES))
        raise ValueError(f"Unknown node '{node_name}'. Valid nodes: {valid}.")
    return name


def normalize_runtime_overrides(
    llm_overrides: Optional[Mapping[str, Any]],
) -> RuntimeLLMOverrides:
    """Canonicalize runtime override dictionaries."""
    if not llm_overrides:
        return {}

    models: dict[str, str] = {}
    fallback_models: dict[str, str] = {}
    temperatures: dict[str, float] = {}
    api_bases: dict[str, str] = {}
    api_keys: dict[str, str] = {}

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

    raw_api_bases = llm_overrides.get("api_bases")
    if isinstance(raw_api_bases, Mapping):
        for node_name, api_base in raw_api_bases.items():
            if api_base is None:
                continue
            api_base_text = str(api_base).strip()
            if not api_base_text:
                continue
            api_bases[normalize_node_name(node_name)] = api_base_text

    raw_api_keys = llm_overrides.get("api_keys")
    if isinstance(raw_api_keys, Mapping):
        for node_name, api_key in raw_api_keys.items():
            if api_key is None:
                continue
            api_key_text = str(api_key).strip()
            if not api_key_text:
                continue
            api_keys[normalize_node_name(node_name)] = api_key_text

    return RuntimeLLMOverrides(
        models=models,
        fallback_models=fallback_models,
        temperatures=temperatures,
        api_bases=api_bases,
        api_keys=api_keys,
    )


def _env_prefix(node_name: str) -> str:
    return normalize_node_name(node_name).upper()


def normalize_api_base(api_base: Optional[str]) -> Optional[str]:
    """Normalize API base URLs used in routing and cache fingerprints."""
    if api_base is None:
        return None
    text = str(api_base).strip()
    if not text:
        return None
    return text.rstrip("/")


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
    api_base = os.getenv(f"LLM_{key}_API_BASE") or os.getenv("LLM_API_BASE") or None
    api_key = os.getenv(f"LLM_{key}_API_KEY") or os.getenv("LLM_API_KEY") or None

    if llm_overrides:
        normalized_overrides = normalize_runtime_overrides(llm_overrides)
        model = normalized_overrides.get("models", {}).get(normalized, model)
        fallback = normalized_overrides.get("fallback_models", {}).get(normalized, fallback)
        if normalized in normalized_overrides.get("temperatures", {}):
            temperature = float(normalized_overrides["temperatures"][normalized])
        api_base = normalized_overrides.get("api_bases", {}).get(normalized, api_base)
        api_key = normalized_overrides.get("api_keys", {}).get(normalized, api_key)

    return NodeModelConfig(
        model=model,
        temperature=temperature,
        fallback_model=fallback,
        api_base=normalize_api_base(api_base),
        api_key=str(api_key).strip() if api_key is not None and str(api_key).strip() else None,
    )


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
