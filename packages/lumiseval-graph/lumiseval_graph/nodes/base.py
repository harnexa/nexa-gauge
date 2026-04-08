"""
Base class for all LumisEval metric nodes.

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
from typing import Any

from lumiseval_core.constants import DEFAULT_JUDGE_MODEL
from lumiseval_core.types import (
    ChunkArtifacts,
    ClaimArtifacts,
    DedupArtifacts,
    GroundingMetrics,
    RelevanceMetrics,
    RedteamMetrics,
    GevalStepsResolved,
    GevalStepsArtifacts,
    GevalMetrics,
    ReferenceMetrics,
    CostEstimate
)

NodeEstimate = [
    ChunkArtifacts,
    ClaimArtifacts,
    DedupArtifacts,
    GroundingMetrics,
    RelevanceMetrics,
    RedteamMetrics,
    GevalStepsResolved,
    GevalStepsArtifacts,
    GevalMetrics,
    ReferenceMetrics,

]


class BaseNode(ABC):
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

    def __init__(self, judge_model: str = DEFAULT_JUDGE_MODEL) -> None:
        self.judge_model = judge_model

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} judge_model={self.judge_model!r}>"

    @property
    def prompt(self) -> str:
        """Return the system prompt for this metric.

        Returns an empty string for nodes that delegate entirely to an external
        library (e.g. DeepEval) and have no custom prompt.
        """
        return self.SYSTEM_PROMPT
