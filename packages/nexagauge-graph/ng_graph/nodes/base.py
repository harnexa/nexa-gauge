"""
Base class for all NexaGauge metric nodes.

Every metric node inherits from BaseMetricNode and receives judge_model at
construction time, enabling cost estimation without running the metric.

Subclass contract:
  - Set `node_name` class attribute to the canonical pipeline node name.
  - Set `SYSTEM_PROMPT` and `USER_PROMPT` class attributes with the prompt
    templates (empty strings for nodes that delegate to an external library).
    `USER_PROMPT` may use `str.format()` placeholders for dynamic content.
  - Implement `evaluate(**kwargs)` with the concrete typed signature for that metric.
  - Implement `cost_estimate()` with the same typed signature defined on the base.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from ng_core.constants import DEFAULT_JUDGE_MODEL
from ng_core.types import (
    ChunkArtifacts,
    ClaimArtifacts,
    CostEstimate,
    GevalMetrics,
    GevalStepsArtifacts,
    GevalStepsResolved,
    GroundingMetrics,
    RedteamMetrics,
    ReferenceMetrics,
    RefinerArtifacts,
    RelevanceMetrics,
)

NodeEstimate = [
    ChunkArtifacts,
    ClaimArtifacts,
    RefinerArtifacts,
    GroundingMetrics,
    RelevanceMetrics,
    RedteamMetrics,
    GevalStepsResolved,
    GevalStepsArtifacts,
    GevalMetrics,
    ReferenceMetrics,
]


class BaseNode(ABC):
    def _reset_model_usage(self) -> None:
        self._model_call_counts: dict[str, int] = {}
        self._model_total_calls: int = 0
        self._model_fallback_hits: int = 0

    def _record_model_response(self, response: Mapping[str, Any], *, primary_model: str) -> None:
        if not hasattr(self, "_model_call_counts"):
            self._reset_model_usage()

        used_model = str(response.get("model") or "").strip()
        self._model_total_calls += 1
        if used_model:
            self._model_call_counts[used_model] = self._model_call_counts.get(used_model, 0) + 1
            if used_model != primary_model:
                self._model_fallback_hits += 1

    def _set_model_usage_counts(
        self,
        *,
        model_counts: Mapping[str, int],
        total_calls: int,
        fallback_hits: int = 0,
    ) -> None:
        self._model_call_counts = {
            str(model): int(count)
            for model, count in model_counts.items()
            if str(model).strip() and int(count) > 0
        }
        self._model_total_calls = int(total_calls)
        self._model_fallback_hits = int(fallback_hits)

    def get_model_usage(self) -> dict[str, Any]:
        model_counts = dict(sorted(getattr(self, "_model_call_counts", {}).items()))
        return {
            "used_models": list(model_counts.keys()),
            "used_model_counts": model_counts,
            "total_calls": int(getattr(self, "_model_total_calls", 0)),
            "fallback_hits": int(getattr(self, "_model_fallback_hits", 0)),
        }

    @abstractmethod
    def run(self, *args: Any, **kwargs) -> NodeEstimate:
        """Execute the metric evaluation and return results.

        Each subclass defines its own fully-typed signature. The base signature
        uses **kwargs to avoid mypy override-incompatibility errors (each metric
        has genuinely different required arguments).
        """
        ...

    @abstractmethod
    def estimate(self, *args: Any, **kwargs: Any) -> CostEstimate:
        """Estimate cost for this node without running any LLM calls."""
        ...


class BaseMetricNode(BaseNode):
    """Abstract base for all pipeline metric nodes."""

    node_name: str = ""
    SYSTEM_PROMPT: str = ""
    USER_PROMPT: str = ""

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        llm_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.judge_model = judge_model
        self.llm_overrides = llm_overrides

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} judge_model={self.judge_model!r}>"

    @property
    def prompt(self) -> str:
        """Return the system prompt for this metric.

        Returns an empty string for nodes that delegate entirely to an external
        library (e.g. DeepEval) and have no custom prompt.
        """
        return self.SYSTEM_PROMPT
