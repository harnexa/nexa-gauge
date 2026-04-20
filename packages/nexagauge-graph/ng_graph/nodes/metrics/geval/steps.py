"""GEval steps node — native (no DeepEval)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ng_core.cache import NodeCacheBackend
from ng_core.types import (
    CostEstimate,
    GevalCacheArtifact,
    GevalMetricInput,
    GevalStepsArtifacts,
    GevalStepsResolved,
    Item,
)
from ng_core.utils import _count_tokens, template_static_tokens
from ng_graph.llm.gateway import get_llm
from ng_graph.llm.pricing import cost_usd, get_model_pricing
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.geval.cache import (
    GEVAL_STEPS_PARSER_VERSION,
    GEVAL_STEPS_PROMPT_VERSION,
    build_geval_artifact_cache_key,
    compute_geval_signature,
)
from ng_graph.nodes.metrics.geval.fields import format_param_names
from pydantic import BaseModel, Field

log = get_node_logger("geval_steps")


class _GevalStepsResponse(BaseModel):
    evaluation_steps: list[str] = Field(..., min_length=2, max_length=3)


class GevalStepsNode(BaseMetricNode):
    """Generate G-Eval evaluation steps from a criterion — phase 1 of two.

    Pipeline recap:
        phase 1 (this node)  criteria  ──▶  reusable evaluation steps
        phase 2 (GevalNode)  steps + case values  ──▶  {score, reason}

    Why we pass *parameter names* but not *parameter values*
    --------------------------------------------------------
    The step-generator receives only the canonical field labels the scorer
    will later see — e.g. ``Parameters: Actual Output, Expected Output`` —
    never the actual text of a record. This is deliberate:

    1. **Caching.** Steps are generated once per
       ``(model, prompt_version, parser_version, item_fields, criteria)``
       signature and reused across every record that shares it. Injecting the
       record's values would make each signature unique and defeat the cache.
       In a typical run the step-generator fires a handful of times; the
       scorer fires thousands.

    2. **Reusability.** The paper frames G-Eval as *criterion-level*
       evaluation, not case-level. Steps like "Verify every claim in the
       Actual Output is supported by the Expected Output" apply to any
       record. "Verify that Paris is mentioned in the answer" would overfit.

    3. **Label alignment with phase 2.** Generated steps must reference the
       same labels the scorer renders values under. If phase 1 sees
       ``Parameters: Actual Output, Expected Output`` it will produce steps
       phrased against those labels. Phase 2 then renders::

           Actual Output:
           Paris is the capital of France.

           Expected Output:
           The capital of France is Paris.

       …and the judge can follow the step directly. Drop the parameter
       names from phase 1 and the generator hallucinates labels (e.g.
       "the summary", "the response") that phase 2 never emits.

    Worked example
    --------------
    Input to phase 1::

        SYSTEM: You write evaluation steps …  (rules, invariant)
        USER:   Parameters: Actual Output, Expected Output

                Criteria:
                The answer must be factually correct.

    Output from phase 1 (cached against the signature)::

        [
          "Verify that every factual claim in the Actual Output is supported "
          "by the Expected Output.",
          "Assess whether the Actual Output introduces claims absent from "
          "the Expected Output.",
        ]

    Input to phase 2 (per record, uses those steps verbatim)::

        SYSTEM: You are an evaluator …
        USER:   Evaluation steps:
                1. Verify that every factual claim …
                2. Assess whether …

                Parameters provided: Actual Output, Expected Output

                Test case:
                Actual Output:
                Paris is the capital of France.

                Expected Output:
                The capital of France is Paris.

    The display labels ("Actual Output", "Expected Output") come from the
    shared :mod:`fields.FIELD_DISPLAY_NAMES` map, so phase 1 and phase 2 are
    guaranteed to agree. Edit the map in one place; both prompts follow.

    Failure modes this avoids
    -------------------------
    - **Label drift.** Without `{param_names}` in phase 1, the generator may
      produce "Verify the summary …" against a criterion that has no summary
      field. Phase 2 has nothing to fill in.
    - **Cache collision.** Two metrics with the same criteria but different
      ``item_fields`` (one with ``["generation"]``, one with
      ``["generation", "reference"]``) used to collide on signature and reuse
      the wrong steps. ``item_fields`` is now part of the signature.
    """

    node_name = "geval_steps"
    prompt_version = GEVAL_STEPS_PROMPT_VERSION
    parser_version = GEVAL_STEPS_PARSER_VERSION

    SYSTEM_PROMPT = (
        "You write evaluation steps for a judge that scores outputs on a 1-10 integer scale.\n\n"
        "Generate 2-3 short steps. Each step must:\n"
        '- be an imperative check ("Verify that…", "Assess whether…")\n'
        "- reference at least one provided parameter by name\n"
        "- not reference parameters that are not provided\n"
        "- be self-contained (independent of other steps)\n\n"
        "Return JSON only:\n"
        '{"evaluation_steps": ["step 1", "step 2"]}'
    )
    USER_PROMPT = "Parameters: {param_names}\n\nCriteria:\n{criteria}"
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        llm_overrides: Optional[Mapping[str, Any]] = None,
        artifact_cache_store: Optional[NodeCacheBackend] = None,
    ) -> None:
        super().__init__(judge_model=judge_model, llm_overrides=llm_overrides)
        self._cache_store = artifact_cache_store

    def _signature(self, criteria: str, item_fields: list[str]) -> str:
        return compute_geval_signature(
            criteria=criteria,
            item_fields=item_fields,
            model=self.judge_model,
            prompt_version=self.prompt_version,
            parser_version=self.parser_version,
        )

    def _read_cached_steps(self, signature: str) -> Optional[list[Item]]:
        if self._cache_store is None:
            return None
        entry = self._cache_store.get_entry_by_key(build_geval_artifact_cache_key(signature))
        if entry is None:
            return None
        artifact = entry["node_output"].get("geval_artifact")
        if artifact is None or not artifact.evaluation_steps:
            return None
        return [Item(**step.model_dump()) for step in artifact.evaluation_steps]

    def _write_cached_steps(
        self,
        *,
        signature: str,
        item_fields: list[str],
        criteria: Item,
        evaluation_steps: list[Item],
    ) -> None:
        if self._cache_store is None:
            return
        artifact = GevalCacheArtifact(
            signature=signature,
            model=self.judge_model,
            prompt_version=self.prompt_version,
            parser_version=self.parser_version,
            item_fields=list(item_fields),
            criteria=criteria,
            evaluation_steps=evaluation_steps,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._cache_store.put_by_key(
            build_geval_artifact_cache_key(signature),
            "geval_steps_artifact",
            {"geval_artifact": artifact},
            metadata={
                "signature": signature,
                "model": self.judge_model,
                "prompt_version": self.prompt_version,
                "parser_version": self.parser_version,
                "cache_schema": "v2",
            },
        )

    def _generate_steps(
        self, criteria: str, item_fields: list[str], metric_name: str
    ) -> tuple[list[Item], CostEstimate]:
        llm = get_llm(
            "geval_steps",
            _GevalStepsResponse,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        response = llm.invoke(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.USER_PROMPT.format(
                        param_names=format_param_names(item_fields),
                        criteria=criteria,
                    ),
                },
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
        if len(raw_steps) < 2:
            raise RuntimeError(
                f"GEval steps generation for metric '{metric_name}' produced "
                f"{len(raw_steps)} steps (need at least 2)."
            )

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

        for metric in metrics:
            item_fields = list(metric.item_fields)

            if metric.evaluation_steps:
                resolved.append(
                    GevalStepsResolved(
                        key=metric.name,
                        name=metric.name,
                        item_fields=item_fields,
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

            signature = self._signature(criteria_text, item_fields)
            cached_steps = self._read_cached_steps(signature)

            if cached_steps is not None:
                resolved.append(
                    GevalStepsResolved(
                        key=metric.name,
                        name=metric.name,
                        item_fields=item_fields,
                        evaluation_steps=[Item(**step.model_dump()) for step in cached_steps],
                        steps_source="cache_used",
                        signature=signature,
                    )
                )
                continue

            generated_steps, generation_cost = self._generate_steps(
                criteria_text, item_fields, metric.name
            )
            total_input_tokens += float(generation_cost.input_tokens or 0.0)
            total_output_tokens += float(generation_cost.output_tokens or 0.0)
            total_cost += float(generation_cost.cost or 0.0)

            self._write_cached_steps(
                signature=signature,
                item_fields=item_fields,
                criteria=metric.criteria
                or Item(
                    text=criteria_text, tokens=float(_count_tokens(criteria_text)), cached=False
                ),
                evaluation_steps=generated_steps,
            )
            log.info(f"Generated GEval steps for metric={metric.name} signature={signature}")

            resolved.append(
                GevalStepsResolved(
                    key=metric.name,
                    name=metric.name,
                    item_fields=item_fields,
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
