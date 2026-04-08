"""GEval scoring node.

Consumes resolved GEval metrics from `EvalCase.node_geval_steps`.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from numbers import Number
from typing import Any, Optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from pydantic import BaseModel

from lumiseval_core.constants import METRIC_PASS_THRESHOLD
from lumiseval_graph.nodes.metrics.geval.cache import (
    GEVAL_STEPS_PARSER_VERSION,
    GEVAL_STEPS_PROMPT_VERSION,
)
from lumiseval_core.types import (
    CostEstimate,
    GevalMetrics,
    GevalStepsResolved,
    Item,
    MetricCategory,
    MetricResult,
)

from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseMetricNode

log = get_node_logger("geval")


@dataclass
class _TokenUsage:
    input_tokens: float = 0.0
    output_tokens: float = 0.0

    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: float, output_tokens: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


class _EvaluationResult(BaseModel):
    metric: MetricResult
    usage: _TokenUsage
    cost: float


_FIELD_TO_PARAM = {
    "question": LLMTestCaseParams.INPUT,
    "generation": LLMTestCaseParams.ACTUAL_OUTPUT,
    "reference": LLMTestCaseParams.EXPECTED_OUTPUT,
    "context": LLMTestCaseParams.CONTEXT,
}


class GevalNode(BaseMetricNode):
    """Evaluate generation quality against resolved GEval metrics."""

    node_name = "geval"
    prompt_version = GEVAL_STEPS_PROMPT_VERSION
    parser_version = GEVAL_STEPS_PARSER_VERSION

    @staticmethod
    def _missing_required_fields(
        *,
        item_fields: list[str],
        generation: str,
        question: Optional[str],
        reference: Optional[str],
        context: Optional[str],
    ) -> list[str]:
        missing: list[str] = []
        for field_name in item_fields:
            if field_name == "generation" and not (generation and generation.strip()):
                missing.append("generation")
            if field_name == "question" and not (question and question.strip()):
                missing.append("question")
            if field_name == "reference" and not (reference and reference.strip()):
                missing.append("reference")
            if field_name == "context" and not (context and context.strip()):
                missing.append("context")
        return missing

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, Number):
            return float(value)
        return 0.0

    @classmethod
    def _extract_tokens_from_usage_payload(cls, payload: Any) -> tuple[float, float]:
        usage = getattr(payload, "usage", None)
        if usage is None and isinstance(payload, dict):
            usage = payload.get("usage")
        if usage is None:
            return 0.0, 0.0

        if isinstance(usage, dict):
            return (
                cls._as_float(usage.get("prompt_tokens")),
                cls._as_float(usage.get("completion_tokens")),
            )

        return (
            cls._as_float(getattr(usage, "prompt_tokens", 0.0)),
            cls._as_float(getattr(usage, "completion_tokens", 0.0)),
        )

    @classmethod
    def _extract_tokens_from_calculate_cost_call(
        cls, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[float, float]:
        if "input_tokens" in kwargs or "output_tokens" in kwargs:
            return (
                cls._as_float(kwargs.get("input_tokens")),
                cls._as_float(kwargs.get("output_tokens")),
            )

        if len(args) >= 2 and isinstance(args[0], Number) and isinstance(args[1], Number):
            return cls._as_float(args[0]), cls._as_float(args[1])

        if args:
            return cls._extract_tokens_from_usage_payload(args[0])

        return 0.0, 0.0

    @classmethod
    def _instrument_deepeval_model_usage(cls, model: Any, usage: _TokenUsage) -> None:
        calculate_cost = getattr(model, "calculate_cost", None)
        if not callable(calculate_cost):
            return

        if inspect.iscoroutinefunction(calculate_cost):

            async def _wrapped_calculate_cost(*args: Any, **kwargs: Any) -> Any:
                input_tokens, output_tokens = cls._extract_tokens_from_calculate_cost_call(args, kwargs)
                usage.add(input_tokens, output_tokens)
                return await calculate_cost(*args, **kwargs)

        else:

            def _wrapped_calculate_cost(*args: Any, **kwargs: Any) -> Any:
                input_tokens, output_tokens = cls._extract_tokens_from_calculate_cost_call(args, kwargs)
                usage.add(input_tokens, output_tokens)
                return calculate_cost(*args, **kwargs)

        setattr(model, "calculate_cost", _wrapped_calculate_cost)

    async def _evaluate_metric(
        self,
        *,
        metric: GevalStepsResolved,
        generation: str,
        question: Optional[str],
        reference: Optional[str],
        context: Optional[str],
    ) -> _EvaluationResult:
        """Run one GEval scoring call for a single resolved metric."""
        evaluation_steps = [step.text.strip() for step in metric.evaluation_steps if step.text.strip()]
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

        test_case = LLMTestCase(
            input=question or "",
            actual_output=generation,
            expected_output=reference or "",
            context=[context] if context else None,
        )
        g_eval = GEval(
            name=metric.name,
            criteria=f"Evaluate this response for '{metric.name}' using the provided evaluation steps.",
            evaluation_steps=evaluation_steps,
            evaluation_params=[_FIELD_TO_PARAM[field_name] for field_name in metric.item_fields],
            model=self.judge_model,
        )

        usage = _TokenUsage()
        self._instrument_deepeval_model_usage(g_eval.model, usage)
        await asyncio.get_event_loop().run_in_executor(None, g_eval.measure, test_case)

        score = g_eval.score or 0.0
        return _EvaluationResult(
            metric=MetricResult(
                name=metric.name,
                category=MetricCategory.ANSWER,
                score=score,
                result=[{"passed": score >= METRIC_PASS_THRESHOLD, "reasoning": g_eval.reason or ""}],
            ),
            usage=usage,
            cost=self._as_float(getattr(g_eval, "evaluation_cost", 0.0)),
        )

    def run(
        self,
        resolved_artifacts: list[GevalStepsResolved],
        generation: Item,
        question: Optional[Item],
        reference: Optional[Item],
        context: Optional[Item],
    ) -> GevalMetrics:  # type: ignore[override]
        """Score resolved GEval metrics from case payload only."""

        if not resolved_artifacts:
            return GevalMetrics(metrics=[], cost=None)

        resolved_by_index: dict[int, GevalStepsResolved] = {
            idx: metric for idx, metric in enumerate(resolved_artifacts)
        }
        indexed_results: dict[int, _EvaluationResult] = {}

        generation_text = generation.text if generation else ""
        question_text = question.text if question else None
        reference_text = reference.text if reference else None
        context_text = context.text if context else None

        async def _run_all() -> list[_EvaluationResult]:
            task_indices: list[int] = []
            tasks: list[asyncio.Future[_EvaluationResult]] = []

            for idx, metric in resolved_by_index.items():
                missing_fields = self._missing_required_fields(
                    item_fields=list(metric.item_fields),
                    generation=generation_text,
                    question=question_text,
                    reference=reference_text,
                    context=context_text,
                )
                if missing_fields:
                    indexed_results[idx] = _EvaluationResult(
                        metric=MetricResult(
                            name=metric.name,
                            category=MetricCategory.ANSWER,
                            error=(
                                "Skipped GEval metric due to missing required record fields: "
                                f"{', '.join(sorted(missing_fields))}."
                            ),
                        ),
                        usage=_TokenUsage(),
                        cost=0.0,
                    )
                    continue

                task_indices.append(idx)
                tasks.append(
                    asyncio.create_task(
                        self._evaluate_metric(
                            metric=metric,
                            generation=generation_text,
                            question=question_text,
                            reference=reference_text,
                            context=context_text,
                        )
                    )
                )

            if tasks:
                evaluated = await asyncio.gather(*tasks)
                for idx, result in zip(task_indices, evaluated):
                    indexed_results[idx] = result

            if not indexed_results:
                return []
            max_index = max(indexed_results.keys())
            return [indexed_results[i] for i in range(max_index + 1) if i in indexed_results]

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

    def cost_estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:
        pricing = get_model_pricing(self.judge_model)
        per_call_cost = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=per_call_cost,
        )

    def estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:  # type: ignore[override]
        return self.cost_estimate(input_tokens=input_tokens, output_tokens=output_tokens)
