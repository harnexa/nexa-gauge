"""Claim-level answer relevancy metric node."""

from typing import Literal, Tuple

from pydantic import BaseModel

from lumiseval_core.types import (
    Claim,
    CostEstimate,
    Item,
    MetricCategory,
    MetricResult,
    Relevancy,
    RelevanceMetrics,
)
from lumiseval_core.utils import _count_tokens, template_static_tokens
from lumiseval_graph.llm.gateway import get_llm
from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseMetricNode

log = get_node_logger("relevance")

_Verdict = Literal["relevant", "irrelevant", "idk"]


class _VerdictItem(BaseModel):
    verdict: _Verdict


class _RelevancyResult(BaseModel):
    verdicts: list[_VerdictItem]


class RelevanceNode(BaseMetricNode):
    node_name = "relevance"
    SYSTEM_PROMPT = "You are a relevancy judge."
    USER_PROMPT = (
        "Question: {question}\n\n"
        "Statements extracted from an answer (one per line):\n{claims}\n\n"
        "For each statement classify its relevance to the question above:\n"
        "  - 'relevant'   : directly addresses the question\n"
        "  - 'irrelevant' : does not relate to the question\n"
        "  - 'idk'        : cannot determine relevance\n\n"
        "Return JSON with key 'verdicts' containing objects: "
        '{{"verdict":"relevant"|"irrelevant"|"idk"}} in the same order.'
    )
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def _answer_relevancy(self, claims: list[Claim], question: str) -> Tuple[MetricResult, CostEstimate]:
        numbered = "\n".join(f"{i + 1}. {c.item.text}" for i, c in enumerate(claims))
        response = get_llm(
            "relevance",
            _RelevancyResult,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        ).invoke(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT.format(question=question, claims=numbered)},
            ]
        )

        pricing = get_model_pricing(self.judge_model)
        prompt_tokens = float(response["usage"]["prompt_tokens"])
        completion_tokens = float(response["usage"]["completion_tokens"])
        cost = CostEstimate(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=cost_usd(prompt_tokens, pricing, "input") + cost_usd(completion_tokens, pricing, "output"),
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
        relevant_count = sum(1 for v in verdicts if v.verdict == "relevant")
        score = relevant_count / len(verdicts)

        per_claim = [Relevancy(**c.model_dump(), verdict=v.verdict) for c, v in zip(claims, verdicts)]

        return (
            MetricResult(
                name="answer_relevancy",
                category=MetricCategory.ANSWER,
                score=score,
                result=per_claim,
            ),
            cost,
        )

    def run(  # type: ignore[override]
        self,
        claims: list[Claim],
        question: Item | str | None,
        enable_relevance: bool = True,
    ) -> RelevanceMetrics:
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not claims or not enable_relevance:
            return RelevanceMetrics(
                metrics=[],
                cost=zero_cost,
            )

        if isinstance(question, Item):
            question_text = question.text
        else:
            question_text = question or ""

        if not question_text.strip():
            log.info("No question provided — skipping answer relevancy")
            return RelevanceMetrics(metrics=[], cost=zero_cost)

        result, cost = self._answer_relevancy(claims=claims, question=question_text)
        return RelevanceMetrics(metrics=[result], cost=cost)

    def estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:  # type: ignore[override]
        pricing = get_model_pricing(self.judge_model)
        billable_input = self.static_prompt_tokens + input_tokens
        return CostEstimate(
            input_tokens=billable_input,
            output_tokens=output_tokens,
            cost=cost_usd(billable_input, pricing, "input") + cost_usd(output_tokens, pricing, "output"),
        )
