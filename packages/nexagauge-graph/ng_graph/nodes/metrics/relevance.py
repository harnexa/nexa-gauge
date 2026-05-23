"""Claim-level answer relevancy metric node."""

from typing import Tuple

from ng_core.constants import (
    AVG_CLAIM_INPUT_TOKENS,
    AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT,
    AVG_CLAIMS_PER_CHUNK,
    RELEVANCE_METRIC_PASS_THRESHOLD,
)
from ng_core.types import (
    Claim,
    CostEstimate,
    Item,
    MetricCategory,
    MetricResult,
    RelevanceMetrics,
    Relevancy,
)
from ng_core.utils import _count_tokens, template_static_tokens
from ng_graph.llm.gateway import get_llm
from ng_graph.llm.pricing import cost_usd, get_node_pricing
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.verdicts import verdict_from_score
from pydantic import BaseModel

log = get_node_logger("relevance")


class _RelevancyResult(BaseModel):
    verdicts: list[bool]


class RelevanceNode(BaseMetricNode):
    node_name = "relevance"
    SYSTEM_PROMPT = (
        "You are an answer relevance judge. Evaluate only whether each statement "
        "addresses the user's input. Do not judge factual correctness, truth, "
        "or whether the statement is supported by evidence."
    )
    USER_PROMPT = (
        "Input: {input}\n\n"
        "Statements extracted from an answer (one per line):\n{claims}\n\n"
        "For each statement, return true if it is on-topic and responsive to the "
        "input, even if the statement may be factually wrong. Return false only "
        "if the statement is unrelated, off-topic, or does not help answer the "
        "input.\n\n"
        "Return a JSON object with key 'verdicts' containing a list of booleans "
        "(true = relevant, false = not relevant) in the same order."
    )
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def _answer_relevancy(
        self, claims: list[Claim], input: str
    ) -> Tuple[MetricResult, CostEstimate]:
        numbered = "\n".join(f"{i + 1}. {c.item.text}" for i, c in enumerate(claims))
        response = get_llm(
            "relevance",
            _RelevancyResult,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        ).invoke(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.USER_PROMPT.format(input=input, claims=numbered),
                },
            ]
        )
        self._record_model_response(response, primary_model=self.judge_model)

        pricing = get_node_pricing(
            node_name=self.node_name,
            model=self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        prompt_tokens = float(response["usage"]["prompt_tokens"])
        completion_tokens = float(response["usage"]["completion_tokens"])
        cost = CostEstimate(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=cost_usd(prompt_tokens, pricing, "input")
            + cost_usd(completion_tokens, pricing, "output"),
        )

        result: _RelevancyResult = response["parsed"]
        if result is None or not result.verdicts:
            log.warning("Answer relevancy LLM call returned no verdicts")
            return (
                MetricResult(
                    name="answer_relevancy",
                    category=MetricCategory.ANSWER,
                    error="No verdicts returned",
                ),
                cost,
            )

        verdicts = result.verdicts[: len(claims)]
        score = sum(verdicts) / len(verdicts)

        per_claim = [
            Relevancy(**c.model_dump(), verdict="ACCEPTED" if v else "REJECTED")
            for c, v in zip(claims, verdicts)
        ]

        return (
            MetricResult(
                name="answer_relevancy",
                category=MetricCategory.ANSWER,
                score=score,
                verdict=verdict_from_score(score, RELEVANCE_METRIC_PASS_THRESHOLD),
                result=per_claim,
            ),
            cost,
        )

    def run(  # type: ignore[override]
        self,
        claims: list[Claim],
        input: Item | str | None,
        enable_relevance: bool = True,
    ) -> RelevanceMetrics:
        self._reset_model_usage()
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not claims or not enable_relevance:
            return RelevanceMetrics(
                metrics=[],
                cost=zero_cost,
            )

        if isinstance(input, Item):
            input_text = input.text
        else:
            input_text = input or ""

        if not input_text.strip():
            log.info("No input provided — skipping answer relevancy")
            return RelevanceMetrics(metrics=[], cost=zero_cost)

        result, cost = self._answer_relevancy(claims=claims, input=input_text)
        return RelevanceMetrics(metrics=[result], cost=cost)

    def estimate(self, input: Item | str | None) -> CostEstimate:
        self._reset_model_usage()
        question_tokens = (
            float(input.tokens) if isinstance(input, Item) else float(_count_tokens(input or ""))
        )
        input_tokens = (
            self.static_prompt_tokens
            + question_tokens
            + AVG_CLAIM_INPUT_TOKENS * AVG_CLAIMS_PER_CHUNK
        )
        output_tokens = AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT + (AVG_CLAIMS_PER_CHUNK - 1)
        pricing = get_node_pricing(
            node_name=self.node_name,
            model=self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost_usd(input_tokens, pricing, "input")
            + cost_usd(output_tokens, pricing, "output"),
        )
