"""
Base class for all LumisEval metric nodes.

Every metric node inherits from BaseMetricNode and receives judge_model at
construction time, enabling cost estimation without running the metric.

Subclass contract:
  - Set `node_name` class attribute to the canonical pipeline node name.
  - Set `SYSTEM_PROMPT` and `USER_PROMPT` class attributes with the prompt
    templates (empty strings for nodes that delegate to an external library).
    `USER_PROMPT` may use `str.format()` placeholders for dynamic content.
  - Implement `run(**kwargs)` with the concrete typed signature for that metric.
  - Override `cost_estimate(**kwargs)` when token-based cost math is implemented.
"""

from abc import ABC
from typing import Any

from lumiseval_core.constants import DEFAULT_JUDGE_MODEL
from lumiseval_core.types import MetricResult, NodeCostBreakdown


class BaseMetricNode(ABC):
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

    def run(self, **kwargs: Any) -> list[MetricResult]:
        """Execute the metric evaluation and return results.

        Each subclass defines its own fully-typed signature. The base signature
        uses **kwargs to avoid mypy override-incompatibility errors (each metric
        has genuinely different required arguments).
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() is not implemented")

    def cost_estimate(self, **kwargs) -> NodeCostBreakdown:
        """Estimate cost for one invocation of this metric without running it.

        Kwargs recognised by subclasses (all optional):
            record_count (int)       — number of records / cases
            claim_count (int)        — number of unique claims
            rubric_rule_count (int)  — number of rubric rules (rubric node only)

        The default returns zeros. Subclasses override this with real token
        arithmetic once the prompt templates stabilise.
        """
        return NodeCostBreakdown(judge_calls=0, cost_usd=0.0)
