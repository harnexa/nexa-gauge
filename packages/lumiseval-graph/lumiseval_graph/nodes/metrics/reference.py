# Run smoke test:
#   python -m lumiseval_graph.nodes.metrics.reference
"""
Reference Metrics Node — ROUGE, BLEU, METEOR scores against reference.

No LLM calls. Activates only when reference is present.
Returns one MetricResult per metric: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR.
All scores are in the 0.0–1.0 range where 1.0 is best.
"""

from typing import Optional

import nltk
from lumiseval_core.types import (
    MetricCategory,
    MetricResult,
    NodeCostBreakdown,
    ReferenceCostMeta,
)
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.metrics.base import BaseMetricNode

log = get_node_logger("reference")

# Download required NLTK data (no-op if already present)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class ReferenceMetricsNode(BaseMetricNode):
    node_name = "reference"
    SYSTEM_PROMPT = ""  # No LLM — these are reference-based lexical metrics
    USER_PROMPT = ""

    def _compute_rouge(self, generation: str, reference: str) -> list[MetricResult]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores against reference."""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generation)
        return [
            MetricResult(
                name="rouge1",
                category=MetricCategory.ANSWER,
                score=round(scores["rouge1"].fmeasure, 4),
            ),
            MetricResult(
                name="rouge2",
                category=MetricCategory.ANSWER,
                score=round(scores["rouge2"].fmeasure, 4),
            ),
            MetricResult(
                name="rougeL",
                category=MetricCategory.ANSWER,
                score=round(scores["rougeL"].fmeasure, 4),
            ),
        ]

    def _compute_bleu(self, generation: str, reference: str) -> MetricResult:
        """Compute sentence BLEU score (smoothed) against reference."""
        hypothesis = generation.split()
        ref_tokens = reference.split()
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hypothesis, smoothing_function=smoothing)
        return MetricResult(
            name="bleu",
            category=MetricCategory.ANSWER,
            score=round(score, 4),
        )

    def _compute_meteor(self, generation: str, reference: str) -> MetricResult:
        """Compute METEOR score against reference."""
        hypothesis = generation.split()
        ref_tokens = reference.split()
        score = meteor_score([ref_tokens], hypothesis)
        return MetricResult(
            name="meteor",
            category=MetricCategory.ANSWER,
            score=round(score, 4),
        )

    def run(  # type: ignore[override]
        self,
        *,
        generation: str,
        reference: Optional[str],
        enable_generation_metrics: bool = True,
    ) -> list[MetricResult]:
        """Compute ROUGE, BLEU, and METEOR scores; return empty list when reference is absent."""
        if not enable_generation_metrics:
            return []
        if not reference:
            log.info("No reference provided — skipping reference metrics")
            return []

        results: list[MetricResult] = []
        results.extend(self._compute_rouge(generation, reference))
        results.append(self._compute_bleu(generation, reference))
        results.append(self._compute_meteor(generation, reference))
        return results

    def cost_estimate(
        self,
        *,
        cost_meta: ReferenceCostMeta,
        **_ignored,
    ) -> NodeCostBreakdown:
        """No LLM calls — cost is always $0."""
        return NodeCostBreakdown(model_calls=0, cost_usd=0.0)


# ── Manual smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test with a near-perfect match and a partial match.

    Expected: near-perfect case → high scores; partial case → lower scores.
    """
    from pprint import pprint

    node = ReferenceMetricsNode()
    print(repr(node))

    generation = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris, France, built between 1887 and 1889."
    )
    reference = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris, France, built between 1887 and 1889 as the entrance arch for "
        "the 1889 World's Fair."
    )

    print("\n── Near-match case ──")
    results = node.run(generation=generation, reference=reference)
    for r in results:
        pprint(r)

    print("\n── No reference case ──")
    results = node.run(generation=generation, reference=None)
    print(f"results: {results!r}  (expected [])")
