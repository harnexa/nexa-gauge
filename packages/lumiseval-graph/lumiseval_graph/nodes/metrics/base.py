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
  - Override `cost_estimate()` with the same typed signature defined on the base.
"""

from abc import ABC
from typing import Any, Union

from lumiseval_core.constants import DEFAULT_JUDGE_MODEL
from lumiseval_core.types import (
    GorundingCostMeta,
    MetricResult,
    NodeCostBreakdown,
    RedTeamCostMeta,
    RelevanceCostMeta,
    RubricCostMeta,
)


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

    def cost_estimate(
        self,
        *,
        cost_meta: Union[GorundingCostMeta, RelevanceCostMeta, RubricCostMeta, RedTeamCostMeta],
    ) -> NodeCostBreakdown:
        """Estimate cost for this node without running any LLM calls.

        All arguments are keyword-only with safe defaults so the orchestrator
        can call every node uniformly with the same kwargs dict — each node
        ignores what it does not use.

        Args:
            eligible_records:      Records this node will process.
            avg_claims_per_record: Average extracted claims per eligible record.
            avg_context_tokens:    Average retrieved-context tokens per record
                                   (grounding).
            avg_question_tokens:   Average question length in tokens (relevance).
            rubric_rule_count:     Number of rubric rules to evaluate (rubric).

        Returns:
            NodeCostBreakdown. Base returns zeros; subclasses override with
            real token arithmetic.
        """
        return NodeCostBreakdown(model_calls=0, cost_usd=0.0)
