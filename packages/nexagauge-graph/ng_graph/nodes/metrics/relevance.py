"""Answer-relevance metric node — claim-level on-topic judgment vs the user input.

Supports the shared per-node knobs (see :mod:`ng_graph.nodes.metrics.scoring`):

- ``scoring_mode``:
    - ``binary_yes_no`` (default): judge returns ``scores: list[int]`` in ``{0,1}``;
      per-claim score is 0 or 1.
    - ``scale_1_5``: judge returns ``scores: list[int]`` (1-5); per-claim
      score is normalized via shared min/max scaling, then averaged.
- ``include_reasoning``: when ``True``, the judge also returns a single
  ``reasoning`` string summarising the batch decision.
"""

from __future__ import annotations

from functools import lru_cache
from statistics import mean
from typing import Any, Tuple

from ng_core.constants import (
    AVG_CLAIM_INPUT_TOKENS,
    AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT,
    AVG_CLAIMS_PER_CHUNK,
    AVG_JUDGE_REASONING_TOKENS,
    RELEVANCE_METRIC_PASS_THRESHOLD,
)
from ng_core.types import (
    Claim,
    CostEstimate,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    Relevance,
    RelevancyClaim,
    RelevanceMetrics,
    ScoringMode,
)
from ng_core.utils import _count_tokens, template_static_tokens
from ng_graph.llm.gateway import get_llm
from ng_graph.llm.pricing import cost_usd, get_node_pricing
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.commons import (
    build_scores_response_model,
    normalize_score_value,
    raw_int_from_score,
)
from ng_graph.nodes.metrics.scoring import (
    build_score_output_contract,
    build_metric_system_prompt,
)
from ng_graph.nodes.metrics.verdicts import verdict_from_score
from pydantic import BaseModel

log = get_node_logger("relevance")

_BASE_SYSTEM_PROMPT = (
    "You are an answer relevance judge. Given user **Input** and **Answer Statements** "
    "evaluate only whether each statement is relevant to and responsive to the "
    "input. Do not judge factual correctness, truth, or whether the statement "
    "is supported by evidence."
)

_USER_TEMPLATE = (
    "## Input: \n{input}\n\n"
    "## Answer Statements (one per line):\n{claims}"
)



@lru_cache(maxsize=4)
def _static_prompt_tokens_for(mode: ScoringMode, include_reasoning: bool) -> int:
    """Cached static prompt token count per (mode, reasoning) configuration."""
    system = build_metric_system_prompt(_BASE_SYSTEM_PROMPT, mode, include_reasoning)
    output_contract = build_score_output_contract(mode, include_reasoning)
    user_template = _USER_TEMPLATE.format(input="{input}", claims= "{claims}")
    return (
        _count_tokens(system)
        + _count_tokens(output_contract)
        + template_static_tokens(user_template)
    )

class RelevanceNode(BaseMetricNode):
    node_name = "relevance"

    def _answer_relevancy(
        self,
        claims: list[Claim],
        input: str,
        scoring_mode: ScoringMode,
        include_reasoning: bool,
    ) -> Tuple[MetricResult, CostEstimate]:
        numbered = "\n".join(f"{i + 1}. {c.item.text}" for i, c in enumerate(claims))
        response_model = build_scores_response_model(
            "RelevanceResult",
            scoring_mode,
            include_reasoning,
        )
        system_prompt = build_metric_system_prompt(_BASE_SYSTEM_PROMPT, scoring_mode, include_reasoning)
        output_contract = build_score_output_contract(scoring_mode, include_reasoning)
        user_prompt = _USER_TEMPLATE.format(input=input, claims=numbered)

        response = get_llm(
            "relevance",
            response_model,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        ).invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": output_contract},
                {"role": "user", "content": user_prompt},
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

        parsed: BaseModel | None = response["parsed"]
        raw_scores = list(getattr(parsed, "scores", []) or []) if parsed else []
        if not raw_scores:
            log.warning("Answer relevancy LLM call returned no scores")
            return (
                MetricResult(
                    name="answer_relevancy",
                    category=MetricCategory.ANSWER,
                    error="No scores returned",
                ),
                cost,
            )

        raw_scores = raw_scores[: len(claims)]
        per_claim_scores = [normalize_score_value(v, scoring_mode) for v in raw_scores]
        score = mean(per_claim_scores)

        claim_verdicts = [
            RelevancyClaim(
                **claim.model_dump(),
                verdict="ACCEPTED" if per_score >= RELEVANCE_METRIC_PASS_THRESHOLD else "REJECTED",
                raw_score=raw_int_from_score(raw),
            )
            for claim, per_score, raw in zip(claims, per_claim_scores, raw_scores)
        ]
        result_payload: list[Any] = list(claim_verdicts)
        if include_reasoning:
            reasoning_text = str(getattr(parsed, "reasoning", "") or "")
            if reasoning_text:
                result_payload.append({"reasoning": reasoning_text})

        return (
            MetricResult(
                name="answer_relevancy",
                category=MetricCategory.ANSWER,
                score=score,
                verdict=verdict_from_score(score, RELEVANCE_METRIC_PASS_THRESHOLD),
                result=result_payload,
            ),
            cost,
        )

    def run(  # type: ignore[override]
        self,
        claims: list[Claim],
        input: Item | str | None,
        enable_relevance: bool = True,
        scoring_mode: ScoringMode = ScoringMode.BINARY_YES_NO,
        include_reasoning: bool = False,
    ) -> RelevanceMetrics:
        self._reset_model_usage()
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not claims or not enable_relevance:
            return RelevanceMetrics(metrics=[], cost=zero_cost)

        if isinstance(input, Item):
            input_text = input.text
        else:
            input_text = input or ""

        if not input_text.strip():
            log.info("No input provided — skipping answer relevancy")
            return RelevanceMetrics(metrics=[], cost=zero_cost)

        result, cost = self._answer_relevancy(
            claims=claims,
            input=input_text,
            scoring_mode=scoring_mode,
            include_reasoning=include_reasoning,
        )
        return RelevanceMetrics(metrics=[result], cost=cost)

    def estimate(self, inputs: Inputs) -> CostEstimate:  # type: ignore[override]
        """Cost estimate sized for the actual mode/reasoning the case will use."""
        self._reset_model_usage()
        relevance_cfg = inputs.relevance or Relevance()
        mode = relevance_cfg.scoring_mode
        include_reasoning = bool(relevance_cfg.include_reasoning)

        input_item = inputs.input
        input_token_count = (
            float(input_item.tokens) if isinstance(input_item, Item) else 0.0
        )
        input_tokens = (
            _static_prompt_tokens_for(mode, include_reasoning)
            + input_token_count
            + AVG_CLAIM_INPUT_TOKENS * AVG_CLAIMS_PER_CHUNK
        )
        output_tokens = AVG_CLAIM_OUTPUT_TOKENS_BOOLEAN_VERDICT + (AVG_CLAIMS_PER_CHUNK - 1)
        if include_reasoning:
            output_tokens += AVG_JUDGE_REASONING_TOKENS
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
