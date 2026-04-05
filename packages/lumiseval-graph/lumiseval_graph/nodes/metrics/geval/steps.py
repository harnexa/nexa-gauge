"""GEval-steps node.

Generates deterministic evaluation steps for GEval metrics that provide
``criteria`` but not ``evaluation_steps``. Artifacts are cached by signature
for cross-case and cross-run reuse.
"""

from __future__ import annotations

from typing import Optional

from lumiseval_core.geval_cache import (
    GEVAL_STEPS_PARSER_VERSION,
    GEVAL_STEPS_PROMPT_VERSION,
    GevalArtifactCache,
    collect_geval_signatures,
    compute_geval_signature,
)
from lumiseval_core.types import EvalCase, GevalMetricSpec, GevalStepsCostMeta, NodeCostBreakdown
from pydantic import BaseModel

from lumiseval_graph.llm.gateway import get_llm
from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.metrics.base import BaseMetricNode
from lumiseval_graph.nodes.metrics.token_utils import count_tokens, template_static_tokens

log = get_node_logger("geval_steps")


class _GevalStepsResponse(BaseModel):
    evaluation_steps: list[str]


class GevalStepsNode(BaseMetricNode):
    """Generate and cache GEval evaluation steps keyed by criteria signature."""

    node_name = "geval_steps"
    prompt_version = GEVAL_STEPS_PROMPT_VERSION
    parser_version = GEVAL_STEPS_PARSER_VERSION

    SYSTEM_PROMPT = "You are an expert evaluator that writes concrete evaluation steps."
    USER_PROMPT = (
        "Evaluation criteria:\n"
        "{criteria}\n\n"
        "Return 2-3 concrete, measurable evaluation steps as JSON:\n"
        '{"evaluation_steps": ["step 1", "step 2", "..."]}\n'
        "Each step must be specific, testable, and focused on the criterion above."
    )
    static_prompt_tokens: int = count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        artifact_cache: Optional[GevalArtifactCache] = None,
    ) -> None:
        super().__init__(judge_model=judge_model)
        self._artifact_cache = artifact_cache or GevalArtifactCache()

    def _signature(self, metric: GevalMetricSpec) -> str:
        if not metric.criteria:
            raise RuntimeError("GEval metric missing criteria for steps signature generation.")
        return str(
            compute_geval_signature(
                criteria=metric.criteria,
                model=self.judge_model,
                prompt_version=self.prompt_version,
                parser_version=self.parser_version,
            )
        )

    def _generate_steps(self, metric: GevalMetricSpec) -> list[str]:
        """Generate evaluation steps for one metric criteria using structured output."""

        llm = get_llm("geval_steps", _GevalStepsResponse, self.judge_model)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_PROMPT.format(criteria=metric.criteria),
            },
        ]
        response = llm.invoke(messages)
        parsed: _GevalStepsResponse | None = response["parsed"]
        if parsed is None or not parsed.evaluation_steps:
            raise RuntimeError(f"GEval steps generation failed for metric '{metric.name}'.")
        return parsed.evaluation_steps

    @classmethod
    def resolve_cost_kwargs(
        cls,
        *,
        cases: Optional[list[EvalCase]] = None,
        model: Optional[str] = None,
        **_ignored,
    ) -> dict[str, Optional[int]]:
        """Resolve runtime-only cost kwargs used by cost_estimate/cost_formula.

        For GEval step generation, the incremental (delta) cost depends on how many
        criteria signatures are uncached at estimation time. This cannot be derived
        from scanner metadata alone, so we compute it here and return it in the
        canonical kwargs shape consumed by both cost_estimate and cost_formula.
        """
        if not cases or not model:
            return {"uncached_unique_rules": None}

        signatures = collect_geval_signatures(
            cases=cases,
            model=model,
            prompt_version=cls.prompt_version,
            parser_version=cls.parser_version,
        )
        return {"uncached_unique_rules": GevalArtifactCache().count_missing(signatures)}

    def run(  # type: ignore[override]
        self,
        *,
        metrics: list[GevalMetricSpec],
    ) -> dict[str, list[str]]:
        """Return signature-keyed evaluation steps for criteria-only GEval metrics."""

        if not metrics:
            return {}

        steps_by_signature: dict[str, list[str]] = {}
        for metric in metrics:
            if metric.evaluation_steps:
                continue
            if not metric.criteria:
                continue

            signature = self._signature(metric)
            cached_steps = self._artifact_cache.get_steps(signature)
            if cached_steps is not None:
                steps_by_signature[signature] = cached_steps
                continue

            generated_steps = self._generate_steps(metric)
            self._artifact_cache.put_steps(
                signature=signature,
                model=self.judge_model,
                criteria=metric.criteria,
                evaluation_steps=generated_steps,
                prompt_version=self.prompt_version,
                parser_version=self.parser_version,
            )
            steps_by_signature[signature] = generated_steps
            log.info(f"generated GEval steps for metric={metric.name} signature={signature}")

        return steps_by_signature

    def cost_estimate(
        self,
        *,
        cost_meta: GevalStepsCostMeta,
        uncached_unique_rules: Optional[int] = None,
        **_ignored,
    ) -> NodeCostBreakdown:
        """Estimate GEval step-generation cost for cache misses only."""

        missing_rules = (
            max(0, uncached_unique_rules)
            if uncached_unique_rules is not None
            else cost_meta.unique_criteria_count
        )
        if missing_rules == 0 or cost_meta.unique_criteria_count == 0:
            return NodeCostBreakdown(model_calls=0, cost_usd=0.0)

        avg_rule_tokens = round(
            cost_meta.unique_criteria_tokens / max(1, cost_meta.unique_criteria_count)
        )
        pricing = get_model_pricing(self.judge_model)
        input_tokens = self.static_prompt_tokens + avg_rule_tokens
        output_tokens = round(cost_meta.avg_output_tokens)
        per_call_cost = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return NodeCostBreakdown(
            model_calls=missing_rules,
            cost_usd=round(missing_rules * per_call_cost, 6),
        )

    @classmethod
    def cost_formula(
        cls,
        cost_meta: GevalStepsCostMeta,
        *,
        uncached_unique_rules: Optional[int] = None,
    ) -> str:
        """Human-readable formula for GEval step-generation cost."""

        missing_rules = (
            max(0, uncached_unique_rules)
            if uncached_unique_rules is not None
            else cost_meta.unique_criteria_count
        )
        avg_rule_tokens = round(
            cost_meta.unique_criteria_tokens / max(1, cost_meta.unique_criteria_count)
        )
        input_tokens = cls.static_prompt_tokens + avg_rule_tokens
        output_tokens = round(cost_meta.avg_output_tokens)
        total_tokens = missing_rules * (input_tokens + output_tokens)
        return (
            f"calls         = {missing_rules} uncached unique GEval criteria signatures\n"
            f"input_tokens  = {cls.static_prompt_tokens} (prompt) + {avg_rule_tokens} (avg unique criteria tokens) = {input_tokens} tok/call\n"
            f"output_tokens = {output_tokens} tok/call\n"
            f"total_tokens  = {missing_rules} × ({input_tokens} + {output_tokens}) = {total_tokens} tok"
        )
