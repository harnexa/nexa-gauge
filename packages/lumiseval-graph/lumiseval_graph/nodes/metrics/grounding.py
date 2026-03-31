# Run smoke test:
#   python -m lumiseval_graph.nodes.metrics.grounding
"""
Grounding Node — checks each extracted claim against context passages (faithfulness).

Uses a single batched LLM call to verify all claims at once.
Returns a MetricResult with score = fraction of claims supported by context (1.0 = all supported).
"""

from lumiseval_core.constants import (
    COST_AVG_CLAIM_TOKENS,
    COST_AVG_CONTEXT_TOKENS,
    COST_AVG_OUTPUT_TOKENS_BOOLEAN_VERDICT,
)
from lumiseval_core.types import (
    Claim,
    Faithfulness,
    MetricCategory,
    MetricResult,
    NodeCostBreakdown,
)
from pydantic import BaseModel

from lumiseval_graph.llm.gateway import get_llm
from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.metrics.base import BaseMetricNode
from lumiseval_graph.nodes.metrics.token_utils import count_tokens, template_static_tokens

log = get_node_logger("grounding")


class _FaithfulnessResult(BaseModel):
    verdicts: list[bool]


class GroundingNode(BaseMetricNode):
    node_name = "grounding"
    SYSTEM_PROMPT = "You are a factual verification judge."
    USER_PROMPT = (
        "Context passages:\n{context}\n\n"
        "Claims to verify (one per line):\n{claims}\n\n"
        "For each claim determine whether it is fully supported by the context "
        "passages above. Return a JSON object with a single key 'verdicts' containing "
        "a list of booleans — true if supported, false if not — in the same order as "
        "the claims."
    )

    def _faithfulness(self, claims: list[Claim], context: list[str]) -> MetricResult:
        """Check each claim against context passages; return fraction supported."""
        context_text = "\n\n".join(context)
        numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(claims))
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_PROMPT.format(context=context_text, claims=numbered),
            },
        ]
        llm = get_llm("relevance_faithfulness", _FaithfulnessResult, self.judge_model)
        response = llm.invoke(messages)
        result: _FaithfulnessResult = response["parsed"]

        if result is None or not result.verdicts:
            log.warning("Faithfulness LLM call returned no verdicts")
            return MetricResult(
                name="faithfulness", category=MetricCategory.ANSWER, error="No verdicts returned"
            )

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

    def run(  # type: ignore[override]
        self,
        *,
        claims: list[Claim],
        context: list[str],
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
                results.append(self._faithfulness(claims, context))
        return results

    def cost_estimate(
        self,
        *,
        eligible_records: int = 0,
        avg_claims_per_record: float = 0.0,
        avg_context_tokens: int = COST_AVG_CONTEXT_TOKENS,
        **_ignored,
    ) -> NodeCostBreakdown:
        if eligible_records == 0:
            return NodeCostBreakdown(judge_calls=0, cost_usd=0.0)

        pricing = get_model_pricing(self.judge_model)
        claims = max(1.0, avg_claims_per_record)

        input_tokens = (
            _GROUNDING_STATIC_TOKENS + avg_context_tokens + round(claims * COST_AVG_CLAIM_TOKENS)
        )
        output_tokens = round(claims * COST_AVG_OUTPUT_TOKENS_BOOLEAN_VERDICT)

        cost_per_record = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return NodeCostBreakdown(
            judge_calls=eligible_records,
            cost_usd=round(eligible_records * cost_per_record, 6),
        )


# Pre-computed once at module load — static (non-placeholder) token overhead
# for GroundingNode's prompts. Every call pays this regardless of content size.
_GROUNDING_STATIC_TOKENS: int = count_tokens(GroundingNode.SYSTEM_PROMPT) + template_static_tokens(
    GroundingNode.USER_PROMPT
)

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

    MetricResult(
    name='faithfulness',
    category=<MetricCategory.ANSWER: 'answer'>,
    score=0.6666666666666666,
    passed=None,
    reasoning=None,
    error=None,
    result=[
        Faithfulness(
            text='The Transformer architecture uses self-attention to capture dependencies between tokens.', s
            ource_chunk_index=0,
            confidence=0.95,
            extraction_failed=False,
            verdict='ACCEPTED'
        ),
        Faithfulness(
            text='Positional encodings are added to embeddings so the model knows token order.',
            source_chunk_index=0,
            confidence=0.9,
            extraction_failed=False,
            verdict='ACCEPTED'
        ),
        Faithfulness(
            text='Transformer models require a minimum of 12 layers to function effectively.',
            source_chunk_index=0,
            confidence=0.8,
            extraction_failed=False,
            verdict='REJECTED'
            )
        ]
    )

    """
    from pprint import pprint

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

    node = GroundingNode(judge_model="gpt-4o-mini")
    print(repr(node))
    results = node.run(claims=claims, context=context)
    for r in results:
        print("result:")
        pprint(r)
