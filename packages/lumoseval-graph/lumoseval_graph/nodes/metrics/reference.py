# Run smoke test:
#   python -m lumos_graph.nodes.metrics.reference
"""
Reference Metrics Node — ROUGE, BLEU, METEOR scores against reference.

No LLM calls. Activates only when reference is present.
Returns one MetricResult per metric: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR.
All scores are in the 0.0–1.0 range where 1.0 is best.
"""

import nltk
from lumoseval_core.types import (
    CostEstimate,
    Item,
    MetricCategory,
    MetricResult,
    ReferenceMetrics,
)
from lumoseval_graph.log import get_node_logger
from lumoseval_graph.nodes.base import BaseMetricNode
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

log = get_node_logger("reference")

# Download required NLTK data (no-op if already present)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class ReferenceNode(BaseMetricNode):
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
        generation: Item | str,
        reference: Item | str | None,
        enable_generation_metrics: bool = True,
    ) -> ReferenceMetrics:
        """Compute ROUGE, BLEU, and METEOR scores against reference text."""
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not enable_generation_metrics:
            return ReferenceMetrics(metrics=[], cost=zero_cost)

        generation_text = generation.text if isinstance(generation, Item) else (generation or "")
        if isinstance(reference, Item):
            reference_text = reference.text
        else:
            reference_text = reference or ""

        if not reference_text.strip():
            log.info("No reference provided — skipping reference metrics")
            return ReferenceMetrics(metrics=[], cost=zero_cost)

        results: list[MetricResult] = []
        results.extend(self._compute_rouge(generation_text, reference_text))
        results.append(self._compute_bleu(generation_text, reference_text))
        results.append(self._compute_meteor(generation_text, reference_text))
        return ReferenceMetrics(
            metrics=results,
            cost=zero_cost,
        )

    def estimate(
        self,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
    ) -> CostEstimate:
        """No LLM calls — cost is always $0."""
        return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
