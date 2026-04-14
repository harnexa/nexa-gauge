"""Grounding metric node."""

from typing import Tuple

from lumoseval_core.constants import (
    AVG_CLAIM_INPUT_TOKENS,
    AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT,
    AVG_CLAIMS_PER_CHUNK,
)
from lumoseval_core.types import (
    Claim,
    CostEstimate,
    Faithfulness,
    GroundingMetrics,
    Item,
    MetricCategory,
    MetricResult,
)
from lumoseval_core.utils import _count_tokens, template_static_tokens
from lumoseval_graph.llm.gateway import get_llm
from lumoseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumoseval_graph.log import get_node_logger
from lumoseval_graph.nodes.base import BaseMetricNode
from pydantic import BaseModel

log = get_node_logger("grounding")


class _GroundingResult(BaseModel):
    verdicts: list[bool]


class GroundingNode(BaseMetricNode):
    node_name = "grounding"
    SYSTEM_PROMPT = "You are a factual verification judge."
    USER_PROMPT = (
        "Context passages:\n{context}\n\n"
        "Claims to verify (one per line):\n{claims}\n\n"
        "For each claim determine whether it is fully supported by the context passages above. "
        "Return a JSON object with key 'verdicts' containing a list of booleans in the same order."
    )
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def _grounding(self, claims: list[Claim], context: str) -> Tuple[MetricResult, CostEstimate]:
        numbered = "\n".join(f"{i + 1}. {c.item.text}" for i, c in enumerate(claims))
        response = get_llm(
            "grounding",
            _GroundingResult,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        ).invoke(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.USER_PROMPT.format(context=context, claims=numbered),
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

        result: _GroundingResult = response["parsed"]
        if result is None or not result.verdicts:
            return (
                MetricResult(
                    name=self.node_name,
                    category=MetricCategory.ANSWER,
                    error="No verdicts returned",
                ),
                cost,
            )

        verdicts = result.verdicts[: len(claims)]
        score = sum(verdicts) / len(verdicts)
        claim_verdicts = [
            Faithfulness(**claim.model_dump(), verdict="ACCEPTED" if verdict else "REJECTED")
            for claim, verdict in zip(claims, verdicts)
        ]
        return (
            MetricResult(
                name=self.node_name,
                category=MetricCategory.ANSWER,
                score=score,
                result=claim_verdicts,
            ),
            cost,
        )

    def run(  # type: ignore[override]
        self,
        claims: list[Claim],
        context: Item | str | list[str] | None,
        enable_grounding: bool = True,
    ) -> GroundingMetrics:
        self._reset_model_usage()
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not claims or not enable_grounding:
            return GroundingMetrics(metrics=[], cost=zero_cost)

        if isinstance(context, Item):
            context_text = context.text
        elif isinstance(context, list):
            context_text = "\n\n".join([c for c in context if c])
        else:
            context_text = context or ""

        if not context_text.strip():
            log.info("No context passages — skipping grounding")
            return GroundingMetrics(metrics=[], cost=zero_cost)

        result, cost = self._grounding(claims=claims, context=context_text)
        return GroundingMetrics(metrics=[result], cost=cost)

    def estimate(self, context: Item | str | None) -> CostEstimate:
        self._reset_model_usage()
        context_tokens = (
            float(context.tokens)
            if isinstance(context, Item)
            else float(_count_tokens(context or ""))
        )
        input_tokens = (
            self.static_prompt_tokens
            + context_tokens
            + AVG_CLAIM_INPUT_TOKENS * AVG_CLAIMS_PER_CHUNK
        )
        output_tokens = AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT + (AVG_CLAIMS_PER_CHUNK - 1)
        pricing = get_model_pricing(self.judge_model)
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost_usd(input_tokens, pricing, "input")
            + cost_usd(output_tokens, pricing, "output"),
        )
