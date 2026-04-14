"""
LumisEval LLM Gateway.

Two entry points:

  get_llm(node_name, schema, default_model)
      → StructuredLLM for nodes that make direct LLM calls (e.g. claims).
        Returns {"parsed": Schema, "usage": {...}, "parsing_error": ...}.

  get_judge_model(node_name, default)
      → str: the model string for nodes that delegate to metric libraries
             (DeepEval, RAGAS) which accept a judge_model parameter.

Both respect per-node env var overrides (LLM_{NODE}_MODEL / LLM_{NODE}_FALLBACK_MODEL).
"""

from lumoseval_graph.llm.config import (
    NodeModelConfig,
    RuntimeLLMOverrides,
    get_judge_model,
    get_node_config,
    normalize_node_name,
    normalize_runtime_overrides,
)
from lumoseval_graph.llm.gateway import StructuredLLM, get_llm

__all__ = [
    # Structured output
    "StructuredLLM",
    "get_llm",
    # Judge model routing
    "get_judge_model",
    "get_node_config",
    "NodeModelConfig",
    "RuntimeLLMOverrides",
    "normalize_node_name",
    "normalize_runtime_overrides",
]
