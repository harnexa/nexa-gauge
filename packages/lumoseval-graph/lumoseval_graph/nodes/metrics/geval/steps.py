"""GEval steps node."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from lumoseval_core.types import (
    CostEstimate,
    GevalMetricInput,
    GevalStepsArtifacts,
    GevalStepsResolved,
    Item,
)
from lumoseval_core.utils import _count_tokens, template_static_tokens
from lumoseval_graph.llm.gateway import get_llm
from lumoseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumoseval_graph.log import get_node_logger
from lumoseval_graph.nodes.base import BaseMetricNode
from lumoseval_graph.nodes.metrics.geval.cache import (
    GEVAL_STEPS_PARSER_VERSION,
    GEVAL_STEPS_PROMPT_VERSION,
    GevalArtifactCache,
    compute_geval_signature,
)
from pydantic import BaseModel

log = get_node_logger("geval_steps")


class _GevalStepsResponse(BaseModel):
    evaluation_steps: list[str]


class GevalStepsNode(BaseMetricNode):
    node_name = "geval_steps"
    prompt_version = GEVAL_STEPS_PROMPT_VERSION
    parser_version = GEVAL_STEPS_PARSER_VERSION

    SYSTEM_PROMPT = "You are an expert evaluator that writes concrete evaluation steps."
    USER_PROMPT = (
        "Evaluation criteria:\n"
        "{criteria}\n\n"
        "Return 2-3 concrete, measurable evaluation steps as JSON:\n"
        '{{"evaluation_steps": ["step 1", "step 2", "..."]}}\n'
        "Each step must be specific, testable, and focused on the criterion above."
    )
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        llm_overrides: Optional[Mapping[str, Any]] = None,
        artifact_cache: Optional[GevalArtifactCache] = None,
    ) -> None:
        super().__init__(judge_model=judge_model, llm_overrides=llm_overrides)
        self._artifact_cache = artifact_cache or GevalArtifactCache()

    @staticmethod
    def _metric_key_sequence(metrics: list[GevalMetricInput]) -> list[str]:
        name_counts: dict[str, int] = {}
        for metric in metrics:
            name_counts[metric.name] = name_counts.get(metric.name, 0) + 1

        seen: dict[str, int] = {}
        keys: list[str] = []
        for metric in metrics:
            name = metric.name
            if name_counts[name] == 1:
                keys.append(name)
                continue
            seen[name] = seen.get(name, 0) + 1
            keys.append(f"{name}#{seen[name]}")
        return keys

    def _signature(self, criteria: str) -> str:
        return compute_geval_signature(
            criteria=criteria,
            model=self.judge_model,
            prompt_version=self.prompt_version,
            parser_version=self.parser_version,
        )

    def _generate_steps(self, criteria: str, metric_name: str) -> tuple[list[Item], CostEstimate]:
        llm = get_llm(
            "geval_steps",
            _GevalStepsResponse,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        response = llm.invoke(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT.format(criteria=criteria)},
            ]
        )
        self._record_model_response(response, primary_model=self.judge_model)

        pricing = get_model_pricing(self.judge_model)
        prompt_tokens = float(response["usage"]["prompt_tokens"])
        completion_tokens = float(response["usage"]["completion_tokens"])
        cost = CostEstimate(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=cost_usd(prompt_tokens, pricing, "input")
            + cost_usd(completion_tokens, pricing, "output"),
        )

        parsed: _GevalStepsResponse | None = response["parsed"]
        raw_steps = [
            s.strip() for s in (parsed.evaluation_steps if parsed else []) if s and s.strip()
        ]
        if not raw_steps:
            raise RuntimeError(f"GEval steps generation failed for metric '{metric_name}'.")

        steps = [
            Item(text=step, tokens=float(_count_tokens(step)), cached=False) for step in raw_steps
        ]
        return steps, cost

    def run(  # type: ignore[override]
        self,
        metrics: list[GevalMetricInput],
        enable_geval: bool = True,
    ) -> GevalStepsArtifacts:
        self._reset_model_usage()
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not enable_geval or not metrics:
            return GevalStepsArtifacts(resolved_steps=[], cost=zero_cost)

        resolved: list[GevalStepsResolved] = []
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        total_cost = 0.0

        for key, metric in zip(self._metric_key_sequence(metrics), metrics):
            if metric.evaluation_steps:
                resolved.append(
                    GevalStepsResolved(
                        key=key,
                        name=metric.name,
                        item_fields=list(metric.item_fields),
                        evaluation_steps=[
                            Item(**step.model_dump()) for step in metric.evaluation_steps
                        ],
                        steps_source="provided",
                        signature=None,
                    )
                )
                continue

            criteria_text = (metric.criteria.text if metric.criteria else "").strip()
            if not criteria_text:
                log.warning(f"Skipping GEval metric with no criteria/steps: {metric.name}")
                continue

            signature = self._signature(criteria_text)
            cached_steps = self._artifact_cache.get_steps(signature)

            if cached_steps is not None:
                resolved.append(
                    GevalStepsResolved(
                        key=key,
                        name=metric.name,
                        item_fields=list(metric.item_fields),
                        evaluation_steps=[Item(**step.model_dump()) for step in cached_steps],
                        steps_source="cache_used",
                        signature=signature,
                    )
                )
                continue

            generated_steps, generation_cost = self._generate_steps(criteria_text, metric.name)
            total_input_tokens += float(generation_cost.input_tokens or 0.0)
            total_output_tokens += float(generation_cost.output_tokens or 0.0)
            total_cost += float(generation_cost.cost or 0.0)

            self._artifact_cache.put_steps(
                signature=signature,
                model=self.judge_model,
                criteria=metric.criteria
                or Item(
                    text=criteria_text, tokens=float(_count_tokens(criteria_text)), cached=False
                ),
                evaluation_steps=generated_steps,
                prompt_version=self.prompt_version,
                parser_version=self.parser_version,
            )
            log.info(f"Generated GEval steps for metric={metric.name} signature={signature}")

            resolved.append(
                GevalStepsResolved(
                    key=key,
                    name=metric.name,
                    item_fields=list(metric.item_fields),
                    evaluation_steps=[Item(**step.model_dump()) for step in generated_steps],
                    steps_source="generated",
                    signature=signature,
                )
            )

        return GevalStepsArtifacts(
            resolved_steps=resolved,
            cost=CostEstimate(
                cost=total_cost,
                input_tokens=total_input_tokens if total_input_tokens > 0 else None,
                output_tokens=total_output_tokens if total_output_tokens > 0 else None,
            )
            if resolved
            else zero_cost,
        )

    def estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:  # type: ignore[override]
        self._reset_model_usage()
        pricing = get_model_pricing(self.judge_model)
        billable_input = self.static_prompt_tokens + input_tokens
        return CostEstimate(
            input_tokens=billable_input,
            output_tokens=output_tokens,
            cost=cost_usd(billable_input, pricing, "input")
            + cost_usd(output_tokens, pricing, "output"),
        )
