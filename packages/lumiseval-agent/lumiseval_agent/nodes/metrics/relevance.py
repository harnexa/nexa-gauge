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

from lumiseval_core.constants import DEFAULT_JUDGE_MODEL
from lumiseval_core.types import (
    Claim,
    MetricCategory,
    MetricResult,
    Relevancy,
)
from pydantic import BaseModel

from lumiseval_agent.llm.gateway import get_llm
from lumiseval_agent.log import get_node_logger

log = get_node_logger("relevance")

_Verdict = Literal["relevant", "irrelevant", "idk"]


class _VerdictItem(BaseModel):
    verdict: _Verdict


class _RelevancyResult(BaseModel):
    verdicts: list[_VerdictItem]


# ── Metric implementation ─────────────────────────────────────────────────────


def _answer_relevancy(
    claims: list[Claim],
    question: str,
    judge_model: str,
) -> MetricResult:
    """Classify each claim as relevant / irrelevant / idk; score = relevant / total."""
    numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(claims))
    messages = [
        {
            "role": "system",
            "content": "You are a relevancy judge.",
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Statements extracted from an answer (one per line):\n{numbered}\n\n"
                "For each statement classify its relevance to the question above:\n"
                "  - 'relevant'   : the statement directly addresses or helps answer the question\n"
                "  - 'irrelevant' : the statement does not relate to the question\n"
                "  - 'idk'        : cannot determine relevance (ambiguous or incomplete)\n\n"
                "Return a JSON object with a single key 'verdicts' containing a list of objects "
                '{"verdict": "relevant"|"irrelevant"|"idk"} in the same order as the statements.'
            ),
        },
    ]
    llm = get_llm("relevance_answer", _RelevancyResult, judge_model)
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

    per_claim = [Relevancy(**c.model_dump(), verdict=v.verdict) for c, v in zip(claims, verdicts)]

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


# ── Public entry point ────────────────────────────────────────────────────────


def run(
    claims: list[Claim],
    question: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    enable_answer_relevancy: bool = True,
) -> list[MetricResult]:
    """Compute answer relevancy using pre-extracted claims.

    Args:
        claims:                  Deduplicated claims from the mmr_deduplicator node.
        question:                Original query — required for answer relevancy.
        judge_model:             LiteLLM model string.
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
            results.append(_answer_relevancy(claims, question, judge_model))

    return results


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
    result = _answer_relevancy(claims, question, judge_model="gpt-4o-mini")

    print(result)
