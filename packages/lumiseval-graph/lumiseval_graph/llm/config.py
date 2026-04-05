"""
LLM Gateway Configuration — per-node model routing via environment variables.

Every node can be overridden independently, enabling per-node model selection,
fallback routing, and temperature tuning without touching code.

Environment variable convention (all optional):
  LLM_{NODE}_MODEL           — primary model (default: job's judge_model)
  LLM_{NODE}_FALLBACK_MODEL  — model to use if primary call fails
  LLM_{NODE}_TEMPERATURE     — float temperature (default: 0.0)

Where {NODE} is the node name uppercased, with spaces/hyphens → underscores:

  Node name          Environment variable prefix
  ─────────────────  ──────────────────────────
  claims             LLM_CLAIMS_*
  grounding          LLM_GROUNDING_*
  relevance          LLM_RELEVANCE_*
  redteam            LLM_REDTEAM_*
  geval              LLM_GEVAL_*

Example .env:
  # Route grounding checks through a cheaper model
  LLM_GROUNDING_MODEL=gpt-4o-mini
  LLM_GROUNDING_FALLBACK_MODEL=gpt-4o

  # Use a stronger model for claim extraction
  LLM_CLAIMS_MODEL=gpt-4o
  LLM_CLAIMS_FALLBACK_MODEL=gpt-4o-mini
"""

import os
from dataclasses import dataclass
from typing import Optional

from lumiseval_core.pipeline import NODES_BY_NAME as _NODES_BY_NAME


@dataclass
class NodeModelConfig:
    """Resolved model config for a single evaluation node."""

    model: Optional[str]  # None = use job's judge_model
    temperature: float
    fallback_model: Optional[str]


# Sub-metric keys not represented as top-level pipeline nodes
_SUBMETRIC_ENV_KEYS: dict[str, list[str]] = {
    "faithfulness": ["FAITHFULNESS"],
    "answer_relevancy": ["ANSWER_RELEVANCY"],
}


def _candidate_env_prefixes(node_name: str) -> list[str]:
    if node_name in _SUBMETRIC_ENV_KEYS:
        return _SUBMETRIC_ENV_KEYS[node_name]
    spec = _NODES_BY_NAME.get(node_name)
    if spec and spec.env_key_suffixes:
        return list(spec.env_key_suffixes)
    return [node_name.upper().replace("-", "_").replace(" ", "_")]


def get_node_config(node_name: str) -> NodeModelConfig:
    """
    Build NodeModelConfig for a node by reading env var overrides.

    Args:
        node_name: Node identifier, e.g. ``"claims"``.

    Returns:
        NodeModelConfig — fields are None/defaults when the env var is absent.
    """
    model = None
    fallback = None
    temperature = 0.0
    temperature_set = False

    for key in _candidate_env_prefixes(node_name):
        if model is None:
            candidate_model = os.getenv(f"LLM_{key}_MODEL")
            if candidate_model:
                model = candidate_model
        if fallback is None:
            candidate_fallback = os.getenv(f"LLM_{key}_FALLBACK_MODEL")
            if candidate_fallback:
                fallback = candidate_fallback
        if not temperature_set:
            temp_raw = os.getenv(f"LLM_{key}_TEMPERATURE")
            if temp_raw is not None:
                temperature = float(temp_raw)
                temperature_set = True

    return NodeModelConfig(model=model, temperature=temperature, fallback_model=fallback)


def get_judge_model(node_name: str, default: str) -> str:
    """
    Return the model to use for a node.

    Priority: env var override → ``default`` (the job's ``judge_model``).

    Args:
        node_name: Node identifier.
        default: Fallback model string (typically ``job_config.judge_model``).
    """
    cfg = get_node_config(node_name)
    return cfg.model or default
