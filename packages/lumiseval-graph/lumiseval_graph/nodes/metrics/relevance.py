# Run smoke test:
#   python -m lumiseval_graph.nodes.metrics.relevance
"""
Claim-level answer relevancy metric.

Mirrors the DeepEval AnswerRelevancy pattern:
  1. Statements are extracted from the response — we skip this step and use the
     pre-extracted claims from claim_extractor → mmr_deduplicator instead.
  2. Each claim is classified as "relevant", "irrelevant", or "idk" relative to
     the original question.
  3. score = count("relevant") / total_claims

Using claims (not chunks) as statements because:
  - Claims are atomic, deduplicated factual propositions — exactly what DeepEval
    would extract internally with its own LLM call.
  - Chunks are raw text segments (~512 tokens) that may span multiple statements
    and include surrounding context that dilutes the relevancy signal.
"""

from typing import Literal, Optional

from lumiseval_core.constants import (
    COST_AVG_CLAIM_TOKENS,
    COST_AVG_OUTPUT_TOKENS_JSON_VERDICT,
    COST_AVG_QUESTION_TOKENS,
)
from lumiseval_core.types import (
    Claim,
    MetricCategory,
    MetricResult,
    NodeCostBreakdown,
    Relevancy,
)
from pydantic import BaseModel

from lumiseval_graph.llm.gateway import get_llm
from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.metrics.base import BaseMetricNode
from lumiseval_graph.nodes.metrics.token_utils import count_tokens, template_static_tokens

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
        "  - 'relevant'   : the statement directly addresses or helps answer the question\n"
        "  - 'irrelevant' : the statement does not relate to the question\n"
        "  - 'idk'        : cannot determine relevance (ambiguous or incomplete)\n\n"
        "Return a JSON object with a single key 'verdicts' containing a list of objects "
        '{"verdict": "relevant"|"irrelevant"|"idk"} in the same order as the statements.'
    )

    def _answer_relevancy(self, claims: list[Claim], question: str) -> MetricResult:
        """Classify each claim as relevant / irrelevant / idk; score = relevant / total."""
        numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(claims))
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_PROMPT.format(question=question, claims=numbered),
            },
        ]
        llm = get_llm("relevance_answer", _RelevancyResult, self.judge_model)
        response = llm.invoke(messages)
        result: _RelevancyResult = response["parsed"]

        if result is None or not result.verdicts:
            log.warning("Answer relevancy LLM call returned no verdicts")
            return MetricResult(
                name="answer_relevancy",
                category=MetricCategory.ANSWER,
                error="No verdicts returned",
            )

        verdicts = result.verdicts[: len(claims)]
        relevant_count = sum(1 for v in verdicts if v.verdict == "relevant")
        score = relevant_count / len(verdicts)

        per_claim = [
            Relevancy(**c.model_dump(), verdict=v.verdict) for c, v in zip(claims, verdicts)
        ]

        log.success(
            f"answer_relevancy={score:.3f}  "
            f"(relevant={relevant_count}  irrelevant={sum(1 for v in verdicts if v.verdict == 'irrelevant')}  "
            f"idk={sum(1 for v in verdicts if v.verdict == 'idk')}  total={len(verdicts)})"
        )
        return MetricResult(
            name="answer_relevancy",
            category=MetricCategory.ANSWER,
            score=score,
            result=per_claim,
        )

    def run(  # type: ignore[override]
        self,
        *,
        claims: list[Claim],
        question: Optional[str] = None,
        enable_answer_relevancy: bool = True,
    ) -> list[MetricResult]:
        """Compute answer relevancy using pre-extracted claims.

        Args:
            claims:                  Deduplicated claims from the mmr_deduplicator node.
            question:                Original query — required for answer relevancy.
            enable_answer_relevancy: Whether to compute answer relevancy.

        Returns:
            list[MetricResult] — one entry per enabled metric.
        """
        if not claims:
            log.warning("No claims provided — skipping answer relevancy")
            return []

        results: list[MetricResult] = []

        if enable_answer_relevancy:
            if not question:
                log.info("No question provided — skipping answer relevancy")
            else:
                results.append(self._answer_relevancy(claims, question))

        return results

    def cost_estimate(
        self,
        *,
        eligible_records: int = 0,
        avg_claims_per_record: float = 0.0,
        avg_question_tokens: int = COST_AVG_QUESTION_TOKENS,
        **_ignored,
    ) -> NodeCostBreakdown:
        if eligible_records == 0:
            return NodeCostBreakdown(judge_calls=0, cost_usd=0.0)

        pricing = get_model_pricing(self.judge_model)
        claims = max(1.0, avg_claims_per_record)

        input_tokens = (
            _RELEVANCE_STATIC_TOKENS + avg_question_tokens + round(claims * COST_AVG_CLAIM_TOKENS)
        )
        output_tokens = round(claims * COST_AVG_OUTPUT_TOKENS_JSON_VERDICT)

        cost_per_record = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return NodeCostBreakdown(
            judge_calls=eligible_records,
            cost_usd=round(eligible_records * cost_per_record, 6),
        )


# Pre-computed once at module load — static (non-placeholder) token overhead
# for RelevanceNode's prompts.
_RELEVANCE_STATIC_TOKENS: int = count_tokens(RelevanceNode.SYSTEM_PROMPT) + template_static_tokens(
    RelevanceNode.USER_PROMPT
)

# ── Manual smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Real answer relevancy test — mirrors DeepEval's verdict-based approach.

    Question: "What causes type 2 diabetes?"

    Claims:
      - claim_1 → relevant   (insulin resistance is a direct cause)
      - claim_2 → relevant   (obesity is a known contributing factor)
      - claim_3 → irrelevant (peripheral neuropathy is a complication, not a cause)
      - claim_4 → irrelevant (transformer architecture is off-topic)

    Expected score: 2/4 = 0.5
    """
    claims = [
        Claim(
            text="Insulin resistance in muscle and fat cells causes blood glucose to remain elevated.",
            source_chunk_index=0,
            confidence=0.95,
        ),
        Claim(
            text="Excess body weight, particularly visceral fat, is a major contributing factor to type 2 diabetes.",
            source_chunk_index=0,
            confidence=0.90,
        ),
        Claim(
            text="Poorly controlled type 2 diabetes can lead to peripheral neuropathy over time.",
            source_chunk_index=1,
            confidence=0.85,
        ),
        Claim(
            text="The Transformer model uses multi-head self-attention to process token sequences in parallel.",
            source_chunk_index=2,
            confidence=0.80,
        ),
    ]

    question = "What causes type 2 diabetes?"
    node = RelevanceNode(judge_model="gpt-4o-mini")
    print(repr(node))
    results = node.run(claims=claims, question=question)
    for r in results:
        print("result:", r)
