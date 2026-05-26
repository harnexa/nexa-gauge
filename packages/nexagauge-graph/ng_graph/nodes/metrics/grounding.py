"""Grounding metric node — claim-level support verdict against retrieved context.

Supports the shared per-node knobs (see :mod:`ng_graph.nodes.metrics.scoring`):

- ``scoring_mode``:
    - ``binary_yes_no`` (default): judge returns ``scores: list[int]`` in ``{0,1}``;
      per-claim score is 0 or 1.
    - ``scale_1_5``: judge returns ``scores: list[int]`` (1-5); per-claim
      score is normalized via shared min/max scaling, then averaged.
- ``include_reasoning``: when ``True``, the judge also returns a single
  ``reasoning`` string summarising the batch decision; the node surfaces it
  on the ``MetricResult.result`` payload alongside the per-claim scores.
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
    GROUNDING_METRIC_PASS_THRESHOLD,
)
from ng_core.types import (
    Claim,
    CostEstimate,
    Grounding,
    GroundingClaim,
    GroundingMetrics,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    ScoringMode,
)
from ng_core.utils import _count_tokens, template_static_tokens
from ng_graph.llm.gateway import get_llm
from ng_graph.llm.pricing import cost_usd, get_node_pricing
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.scoring import (
    ScoringSpec,
    build_scores_response_model,
    normalize_raw_score,
    verdict_from_score
)
from pydantic import BaseModel

log = get_node_logger("grounding")

_BASE_SYSTEM_PROMPT = (
    "You are a factual verification judge. Given **Context Passages** and **Claims**, "
    "decide whether each claim is supported by the provided context. "
    "Do not use external knowledge."
)

# Mode-specific user-prompt fragments. Kept separate so the per-mode wording is
# explicit; the rest of the template is identical across modes.
_USER_TEMPLATE = (
    "## Context passages:\n{context}\n\n"
    "## Claims to verify (one per line):\n{claims}"
)



@lru_cache(maxsize=4)
def _static_prompt_tokens_for(mode: ScoringMode, include_reasoning: bool) -> int:
    """Cached token count of the static (non-placeholder) prompt for one mode/reasoning combo.

    There are only four combinations (2 modes × 2 reasoning settings), so the
    cache pays for itself after the second case in a batch and replaces the
    previous over-estimating upper-bound constant. The estimator can ask for
    the exact static cost of the configuration the case will actually use.
    """
    system = _BASE_SYSTEM_PROMPT
    scoring_spec = ScoringSpec(model=mode, include_reasoning=include_reasoning)
    user_template = _USER_TEMPLATE.format(context="{context}", claims="{claims}")
    return (
        _count_tokens(system)
        + _count_tokens(scoring_spec.contract)
        + template_static_tokens(user_template)
    )

class GroundingNode(BaseMetricNode):
    node_name = "grounding"

    def _grounding(
        self,
        claims: list[Claim],
        context: str,
        scoring_mode: ScoringMode,
        include_reasoning: bool,
    ) -> Tuple[MetricResult, CostEstimate]:
        scoring_spec = ScoringSpec(mode=scoring_mode, include_reasoning=include_reasoning)
        numbered = "\n".join(f"{i + 1}. {c.item.text}" for i, c in enumerate(claims))

        response_model = build_scores_response_model(
            model_prefix="GroundingResult", 
            mode_value=scoring_spec.mode.value, 
            min_score=scoring_spec.score_min, 
            max_score=scoring_spec.score_max,
            include_reasoning=scoring_spec.include_reasoning
        )

        system_prompt = _BASE_SYSTEM_PROMPT
        output_contract = scoring_spec.contract
        user_prompt =_USER_TEMPLATE.format(context=context, claims=numbered)

        response = get_llm(
            "grounding",
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
            return (
                MetricResult(
                    name=self.node_name,
                    category=MetricCategory.ANSWER,
                    error="No scores returned",
                ),
                cost,
            )

        raw_scores = raw_scores[: len(claims)]
        per_claim_scores = [normalize_raw_score(v, scoring_spec) for v in raw_scores]
        score = mean(per_claim_scores)

        claim_verdicts = [
            GroundingClaim(
                **claim.model_dump(),
                verdict=verdict_from_score(per_score, GROUNDING_METRIC_PASS_THRESHOLD),
                raw_score=raw,
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
                name=self.node_name,
                category=MetricCategory.ANSWER,
                score=score,
                verdict=verdict_from_score(score, GROUNDING_METRIC_PASS_THRESHOLD),
                result=result_payload,
            ),
            cost,
        )

    def run(  # type: ignore[override]
        self,
        claims: list[Claim],
        context: Item | str | list[str] | None,
        enable_grounding: bool = True,
        scoring_mode: ScoringMode = ScoringMode.BINARY_YES_NO,
        include_reasoning: bool = False,
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

        result, cost = self._grounding(
            claims=claims,
            context=context_text,
            scoring_mode=scoring_mode,
            include_reasoning=include_reasoning,
        )

        return GroundingMetrics(metrics=[result], cost=cost)

    def estimate(self, inputs: Inputs) -> CostEstimate:  # type: ignore[override]
        """Cost estimate for the grounding judge call on this case.

        Sources the scoring knobs from ``inputs.grounding`` (falling back to
        defaults when the case omits the block) so the static prompt cost is
        sized for the exact mode/reasoning configuration the case will use,
        not an over-estimating likert+reasoning upper bound.
        """
        self._reset_model_usage()
        grounding_cfg = inputs.grounding or Grounding()
        mode = grounding_cfg.scoring_mode
        include_reasoning = bool(grounding_cfg.include_reasoning)

        context = inputs.context
        context_tokens = (
            float(context.tokens) if isinstance(context, Item) else 0.0
        )
        input_tokens = (
            _static_prompt_tokens_for(mode, include_reasoning)
            + context_tokens
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
