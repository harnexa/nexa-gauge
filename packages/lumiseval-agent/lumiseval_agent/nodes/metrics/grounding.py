"""
Grounding Node — checks each extracted claim against context passages (faithfulness).

Uses a single batched LLM call to verify all claims at once.
Returns a MetricResult with score = fraction of claims supported by context (1.0 = all supported).
"""

from lumiseval_core.constants import DEFAULT_JUDGE_MODEL
from lumiseval_core.types import Claim, Faithfulness, MetricCategory, MetricResult
from pydantic import BaseModel

from lumiseval_agent.llm.gateway import get_llm
from lumiseval_agent.log import get_node_logger

log = get_node_logger("grounding")


class _FaithfulnessResult(BaseModel):
    verdicts: list[bool]


def _faithfulness(claims: list[Claim], context: list[str], judge_model: str) -> MetricResult:
    """Check each claim against context passages; return fraction supported."""
    context_text = "\n\n".join(context)
    numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(claims))
    messages = [
        {
            "role": "system",
            "content": "You are a factual verification judge.",
        },
        {
            "role": "user",
            "content": (
                f"Context passages:\n{context_text}\n\n"
                f"Claims to verify (one per line):\n{numbered}\n\n"
                "For each claim determine whether it is fully supported by the context "
                "passages above. Return a JSON object with a single key 'verdicts' containing "
                "a list of booleans — true if supported, false if not — in the same order as "
                "the claims."
            ),
        },
    ]
    llm = get_llm("relevance_faithfulness", _FaithfulnessResult, judge_model)
    response = llm.invoke(messages)
    result: _FaithfulnessResult = response["parsed"]

    if result is None or not result.verdicts:
        log.warning("Faithfulness LLM call returned no verdicts")
        return MetricResult(
            name="faithfulness", category=MetricCategory.ANSWER, error="No verdicts returned"
        )

    # Align length in case LLM returns fewer/more verdicts than claims
    verdicts = result.verdicts[: len(claims)]
    score = sum(verdicts) / len(verdicts)

    claim_verdicts = [
        Faithfulness(**c.model_dump(), verdict="ACCEPTED" if v else "REJECTED")
        for c, v in zip(claims, verdicts)
    ]

    return MetricResult(
        name="faithfulness",
        category=MetricCategory.ANSWER,
        score=score,
        result=claim_verdicts,
    )


def run(
    claims: list[Claim],
    context: list[str],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    enable_faithfulness: bool = True,
) -> list[MetricResult]:

    if not claims:
        log.warning("No claims provided — skipping faithfulness")
        return []

    results: list[MetricResult] = []

    if enable_faithfulness:
        if not context:
            log.info("No context passages — skipping faithfulness")
        else:
            results.append(_faithfulness(claims, context, judge_model))
    return results


# ── Manual smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Real faithfulness test.

    Context: three paragraphs about transformer architecture.
    Claims:
      - claim_1 (SUPPORTED)  — self-attention is directly stated in the context
      - claim_2 (SUPPORTED)  — positional encodings are mentioned explicitly
      - claim_3 (NOT SUPPORTED) — "transformers require a minimum of 12 layers"
                                   is not stated anywhere in the context

    Expected: 2 of 3 verdicts true → faithfulness ≈ 0.667
    """
    from lumiseval_core.types import Claim

    context = [
        (
            "The Transformer model introduced in 'Attention Is All You Need' (Vaswani et al., 2017) "
            "relies entirely on a self-attention mechanism to draw global dependencies between input "
            "and output, discarding recurrence and convolutions entirely."
        ),
        (
            "Because the model contains no recurrence and no convolution, positional encodings are "
            "added to the input embeddings to inject information about the relative or absolute "
            "position of tokens in the sequence."
        ),
        (
            "Transformers have become the dominant architecture for NLP tasks and are increasingly "
            "applied to vision, audio, and multi-modal domains."
        ),
    ]

    claims = [
        Claim(
            text="The Transformer architecture uses self-attention to capture dependencies between tokens.",
            source_chunk_index=0,
            confidence=0.95,
        ),
        Claim(
            text="Positional encodings are added to embeddings so the model knows token order.",
            source_chunk_index=0,
            confidence=0.90,
        ),
        Claim(
            text="Transformer models require a minimum of 12 layers to function effectively.",
            source_chunk_index=0,
            confidence=0.80,
        ),
    ]

    result = _faithfulness(claims, context, judge_model="gpt-4o-mini")
    print("result: ", result)

    # assert result.error is None, f"Unexpected error: {result.error}"
    # assert result.score is not None and 0.5 <= result.score <= 0.8, (
    #     f"Score {result.score:.3f} outside expected range [0.5, 0.8]"
    # )
    # print("PASSED")
