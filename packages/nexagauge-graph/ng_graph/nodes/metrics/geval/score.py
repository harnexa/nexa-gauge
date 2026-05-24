"""GEval scoring node — native implementation (no DeepEval).

Consumes resolved GEval metrics from ``EvalCase.node_geval_steps``.

Implements G-Eval (Liu et al., EMNLP 2023, arXiv:2303.16634):
  1. Form-filling: judge returns structured JSON with a bounded integer score.
  2. Probability weighting: ``E[score] = Σ k·p(k) / Σ p(k)`` over decimal tokens
     at the score position, filtered to score range and logprob ≥ log(0.01).
  3. Normalization: min/max scaling into the shared [0, 1] space.

Supported scoring modes:
  - ``likert_1_5`` (default)
  - ``binary_yes_no`` represented as ``score in {0,1}`` (yes=1, no=0)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from ng_core.constants import (
    AVG_DEEPEVAL_GEVAL_CRITERIA_STEP_TOKENS,
    AVG_DEEPEVAL_GEVAL_CRITERIA_STEPS,
    AVG_DEEPEVAL_OUTPUT_REASONING_TOKENS,
    AVG_DEEPEVAL_OUTPUT_VERDICT,
    AVG_DEEPEVAL_PROMPT_TOKENS,
    GEVAL_METRIC_PASS_THRESHOLD,
)
from ng_core.types import (
    CostEstimate,
    Geval,
    GevalMetrics,
    GevalScoringMode,
    GevalStepsResolved,
    Item,
    MetricCategory,
    MetricResult,
)
from ng_core.utils import _count_tokens, template_static_tokens
from ng_graph.llm.gateway import get_llm
from ng_graph.llm.pricing import cost_usd, get_node_pricing
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.geval.cache import (
    GEVAL_SCORE_PARSER_VERSION,
    GEVAL_SCORE_PROMPT_VERSION,
)
from ng_graph.nodes.metrics.geval.fields import FIELD_DISPLAY_NAMES, format_param_names
from ng_graph.nodes.metrics.geval.prompt import (
    build_score_system_prompt,
    score_response_model,
    scoring_spec,
)
from ng_graph.nodes.metrics.geval.weighted_score import calculate_weighted_summed_score
from ng_graph.nodes.metrics.verdicts import verdict_from_score
from pydantic import BaseModel

log = get_node_logger("geval")


@dataclass
class _TokenUsage:
    input_tokens: float = 0.0
    output_tokens: float = 0.0

    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens


class _EvaluationResult(BaseModel):
    metric: MetricResult
    usage: _TokenUsage
    cost: float

    model_config = {"arbitrary_types_allowed": True}


class GevalNode(BaseMetricNode):
    """Evaluate output quality against resolved GEval metrics."""

    node_name = "geval"
    prompt_version = GEVAL_SCORE_PROMPT_VERSION
    parser_version = GEVAL_SCORE_PARSER_VERSION

    USER_PROMPT = (
        "Evaluation steps:\n"
        "{steps_block}\n\n"
        "Parameters provided: {param_names}\n\n"
        "Test case:\n"
        "{rendered_fields}\n\n"
        "Produce your JSON score now."
    )
    static_prompt_tokens: int = _count_tokens(
        build_score_system_prompt(
            mode=GevalScoringMode.LIKERT_1_5,
            include_reasoning=True,
        )
    ) + template_static_tokens(USER_PROMPT)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _missing_required_fields(
        *,
        item_fields: list[str],
        output: str,
        input: Optional[str],
        reference: Optional[str],
        context: Optional[str],
    ) -> list[str]:
        missing: list[str] = []
        values = {
            "output": output,
            "input": input,
            "reference": reference,
            "context": context,
        }
        for field_name in item_fields:
            value = values.get(field_name)
            if not (value and str(value).strip()):
                missing.append(field_name)
        return missing

    @staticmethod
    def _build_messages(
        metric: GevalStepsResolved,
        output: str,
        input: Optional[str],
        reference: Optional[str],
        context: Optional[str],
    ) -> list[dict[str, str]]:
        steps_block = "\n".join(
            f"{i + 1}. {step.text.strip()}"
            for i, step in enumerate(metric.evaluation_steps)
            if step.text and step.text.strip()
        )

        values = {
            "output": output,
            "input": input,
            "reference": reference,
            "context": context,
        }
        rendered_fields = "\n\n".join(
            f"{FIELD_DISPLAY_NAMES.get(field_name, field_name)}:\n{values[field_name].strip()}"
            for field_name in metric.item_fields
            if values.get(field_name) and str(values[field_name]).strip()
        )

        user = GevalNode.USER_PROMPT.format(
            steps_block=steps_block,
            param_names=format_param_names(list(metric.item_fields)),
            rendered_fields=rendered_fields,
        )
        return [
            {
                "role": "system",
                "content": build_score_system_prompt(
                    mode=metric.scoring_mode,
                    include_reasoning=metric.include_reasoning,
                ),
            },
            {"role": "user", "content": user},
        ]

    def _cost_from_usage(self, usage: dict[str, Any], model: str) -> float:
        pricing = get_node_pricing(
            node_name=self.node_name,
            model=model or self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        prompt_tokens = float(usage.get("prompt_tokens", 0.0) or 0.0)
        completion_tokens = float(usage.get("completion_tokens", 0.0) or 0.0)
        return cost_usd(prompt_tokens, pricing, "input") + cost_usd(
            completion_tokens, pricing, "output"
        )

    @staticmethod
    def _token_usage_from(usage: dict[str, Any]) -> _TokenUsage:
        return _TokenUsage(
            input_tokens=float(usage.get("prompt_tokens", 0.0) or 0.0),
            output_tokens=float(usage.get("completion_tokens", 0.0) or 0.0),
        )

    # ── Core evaluation ───────────────────────────────────────────────────

    async def _evaluate_metric(
        self,
        *,
        metric: GevalStepsResolved,
        output: str,
        input: Optional[str],
        reference: Optional[str],
        context: Optional[str],
    ) -> _EvaluationResult:
        """Run one GEval scoring call for a single resolved metric."""
        evaluation_steps = [
            step.text.strip() for step in metric.evaluation_steps if step.text and step.text.strip()
        ]
        if not evaluation_steps:
            return _EvaluationResult(
                metric=MetricResult(
                    name=metric.name,
                    category=MetricCategory.ANSWER,
                    error="Skipped GEval metric due to empty evaluation steps.",
                ),
                usage=_TokenUsage(),
                cost=0.0,
            )

        missing = self._missing_required_fields(
            item_fields=list(metric.item_fields),
            output=output,
            input=input,
            reference=reference,
            context=context,
        )
        if missing:
            return _EvaluationResult(
                metric=MetricResult(
                    name=metric.name,
                    category=MetricCategory.ANSWER,
                    error=(
                        "Skipped GEval metric due to missing required item fields: "
                        f"{', '.join(sorted(missing))}."
                    ),
                ),
                usage=_TokenUsage(),
                cost=0.0,
            )

        spec = scoring_spec(metric.scoring_mode)
        response_model = score_response_model(metric.scoring_mode, metric.include_reasoning)
        llm = get_llm(
            "geval",
            response_model,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        messages = self._build_messages(metric, output, input, reference, context)

        response = await asyncio.to_thread(llm.invoke_with_logprobs, messages)
        self._record_model_response(response, primary_model=self.judge_model)

        used_model = response.get("model") or self.judge_model
        usage_payload = response.get("usage") or {}
        usage = self._token_usage_from(usage_payload)
        cost = self._cost_from_usage(usage_payload, used_model)

        parsed: BaseModel | None = response.get("parsed")
        if parsed is None:
            parsing_error = response.get("parsing_error")
            return _EvaluationResult(
                metric=MetricResult(
                    name=metric.name,
                    category=MetricCategory.ANSWER,
                    error=f"Failed to parse GEval response: {parsing_error}",
                ),
                usage=usage,
                cost=cost,
            )

        raw_score = int(getattr(parsed, "score"))
        logprobs_content = response.get("logprobs")
        if logprobs_content:
            weighted = calculate_weighted_summed_score(
                raw_score,
                logprobs_content,
                score_min=spec.score_min,
                score_max=spec.score_max,
            )
        else:
            weighted = float(raw_score)

        normalized = (weighted - spec.score_min) / (spec.score_max - spec.score_min)
        normalized = max(0.0, min(1.0, normalized))
        passed = normalized >= GEVAL_METRIC_PASS_THRESHOLD
        reasoning_text = str(getattr(parsed, "reason", "") or "")

        return _EvaluationResult(
            metric=MetricResult(
                name=metric.name,
                category=MetricCategory.ANSWER,
                score=normalized,
                verdict=verdict_from_score(normalized, GEVAL_METRIC_PASS_THRESHOLD),
                result=[
                    {
                        "passed": passed,
                        "reasoning": reasoning_text,
                        "tokens": _count_tokens(reasoning_text),
                    }
                ],
            ),
            usage=usage,
            cost=cost,
        )

    # ── Orchestration ─────────────────────────────────────────────────────

    def run(
        self,
        resolved_artifacts: list[GevalStepsResolved],
        output: Item,
        input: Optional[Item],
        reference: Optional[Item],
        context: Optional[Item],
    ) -> GevalMetrics:  # type: ignore[override]
        """Score resolved GEval metrics from case payload only."""
        self._reset_model_usage()

        if not resolved_artifacts:
            return GevalMetrics(metrics=[], cost=None)

        output_text = output.text if output else ""
        input_text = input.text if input else None
        reference_text = reference.text if reference else None
        context_text = context.text if context else None

        async def _run_all() -> list[_EvaluationResult]:
            tasks = [
                self._evaluate_metric(
                    metric=metric,
                    output=output_text,
                    input=input_text,
                    reference=reference_text,
                    context=context_text,
                )
                for metric in resolved_artifacts
            ]
            return list(await asyncio.gather(*tasks))

        evaluated = asyncio.run(_run_all())
        total_input_tokens = sum(result.usage.input_tokens for result in evaluated)
        total_output_tokens = sum(result.usage.output_tokens for result in evaluated)
        total_cost = sum(result.cost for result in evaluated)

        log.info(
            "GEval evaluated "
            f"metrics={len(evaluated)} "
            f"input_tokens={round(total_input_tokens, 4)} "
            f"output_tokens={round(total_output_tokens, 4)} "
            f"total_tokens={round(total_input_tokens + total_output_tokens, 4)}"
        )

        input_tokens = total_input_tokens if total_input_tokens > 0 else None
        output_tokens = total_output_tokens if total_output_tokens > 0 else None
        metrics = [result.metric for result in evaluated]
        return GevalMetrics(
            metrics=metrics,
            cost=CostEstimate(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=round(total_cost, 8),
            ),
        )

    # ── Estimation ────────────────────────────────────────────────────────

    def estimate(
        self,
        output: Item,
        input: Optional[Item],
        reference: Optional[Item],
        context: Optional[Item],
        geval: Optional[Geval],
    ) -> CostEstimate:
        self._reset_model_usage()
        if not geval or not geval.metrics:
            return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)

        metric_count = len(geval.metrics)
        steps_tokens = 0.0
        for metric in geval.metrics:
            if metric.evaluation_steps:
                steps_tokens += sum(float(step.tokens) for step in metric.evaluation_steps)
            elif metric.criteria:
                steps_tokens += float(metric.criteria.tokens)
                steps_tokens += (
                    AVG_DEEPEVAL_GEVAL_CRITERIA_STEPS * AVG_DEEPEVAL_GEVAL_CRITERIA_STEP_TOKENS
                )

        input_tokens = (
            metric_count * AVG_DEEPEVAL_PROMPT_TOKENS
            + steps_tokens
            + float(output.tokens)
            + (float(input.tokens) if input else 0.0)
            + (float(reference.tokens) if reference else 0.0)
            + (float(context.tokens) if context else 0.0)
        )
        output_tokens = 0.0
        for metric in geval.metrics:
            output_tokens += AVG_DEEPEVAL_OUTPUT_VERDICT
            if metric.include_reasoning:
                output_tokens += AVG_DEEPEVAL_OUTPUT_REASONING_TOKENS

        pricing = get_node_pricing(
            node_name=self.node_name,
            model=self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        estimated_cost = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimated_cost,
        )
