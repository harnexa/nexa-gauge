# Run smoke test:
#   python -m ng_graph.nodes.metrics.refmatch
"""
Reference Metrics Node — ROUGE, BLEU, METEOR scores against reference.

No LLM calls. Activates only when reference is present.
Returns one MetricResult per metric: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR.
All scores are in the 0.0–1.0 range where 1.0 is best.
"""

from threading import Lock

import nltk
from ng_core.constants import (
    REFERENCE_BLEU_METRIC_PASS_THRESHOLD,
    REFERENCE_METEOR_METRIC_PASS_THRESHOLD,
    REFERENCE_ROUGE1_METRIC_PASS_THRESHOLD,
    REFERENCE_ROUGE2_METRIC_PASS_THRESHOLD,
    REFERENCE_ROUGEL_METRIC_PASS_THRESHOLD,
)
from ng_core.types import (
    CostEstimate,
    Item,
    MetricCategory,
    MetricResult,
    RefmatchMetrics,
)
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.scoring import verdict_from_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

log = get_node_logger("refmatch")
_METEOR_LOCK = Lock()
_REFERENCE_THRESHOLDS: dict[str, float] = {
    "rouge1": REFERENCE_ROUGE1_METRIC_PASS_THRESHOLD,
    "rouge2": REFERENCE_ROUGE2_METRIC_PASS_THRESHOLD,
    "rougeL": REFERENCE_ROUGEL_METRIC_PASS_THRESHOLD,
    "bleu": REFERENCE_BLEU_METRIC_PASS_THRESHOLD,
    "meteor": REFERENCE_METEOR_METRIC_PASS_THRESHOLD,
}


class _NoWordNet:
    """Fallback shim for METEOR when WordNet access is unavailable/unstable."""

    def synsets(self, _word: str) -> list[object]:
        return []


# Download required NLTK data (no-op if already present)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class RefmatchNode(BaseMetricNode):
    node_name = "refmatch"
    SYSTEM_PROMPT = ""  # No LLM — these are reference-based lexical metrics
    USER_PROMPT = ""

    @staticmethod
    def _metric_result(name: str, score: float) -> MetricResult:
        threshold = _REFERENCE_THRESHOLDS[name]
        rounded_score = round(score, 4)
        return MetricResult(
            name=name,
            category=MetricCategory.ANSWER,
            score=rounded_score,
            verdict=verdict_from_score(rounded_score, threshold),
        )

    def _compute_rouge(self, output: str, reference: str) -> list[MetricResult]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores against reference."""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, output)
        return [
            self._metric_result("rouge1", scores["rouge1"].fmeasure),
            self._metric_result("rouge2", scores["rouge2"].fmeasure),
            self._metric_result("rougeL", scores["rougeL"].fmeasure),
        ]

    def _compute_bleu(self, output: str, reference: str) -> MetricResult:
        """Compute sentence BLEU score (smoothed) against reference."""
        hypothesis = output.split()
        ref_tokens = reference.split()
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hypothesis, smoothing_function=smoothing)
        return self._metric_result("bleu", score)

    def _compute_meteor(self, output: str, reference: str) -> MetricResult:
        """Compute METEOR score against reference."""
        hypothesis = output.split()
        ref_tokens = reference.split()
        try:
            # NLTK WordNet access inside METEOR is not reliably thread-safe in
            # all environments; serialize calls to avoid intermittent crashes.
            with _METEOR_LOCK:
                score = meteor_score([ref_tokens], hypothesis)
        except Exception as exc:
            log.warning(f"METEOR WordNet path failed ({exc}); falling back to token-only METEOR")
            try:
                with _METEOR_LOCK:
                    score = meteor_score([ref_tokens], hypothesis, wordnet=_NoWordNet())
            except Exception as fallback_exc:
                log.warning(f"METEOR fallback failed ({fallback_exc}); defaulting score to 0.0")
                score = 0.0
        return self._metric_result("meteor", score)

    def run(  # type: ignore[override]
        self,
        output: Item | str,
        reference: Item | str | None,
        enable_output_metrics: bool = True,
    ) -> RefmatchMetrics:
        """Compute ROUGE, BLEU, and METEOR scores against reference text."""
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not enable_output_metrics:
            return RefmatchMetrics(metrics=[], cost=zero_cost)

        output_text = output.text if isinstance(output, Item) else (output or "")
        if isinstance(reference, Item):
            reference_text = reference.text
        else:
            reference_text = reference or ""

        if not reference_text.strip():
            log.info("No reference provided — skipping reference metrics")
            return RefmatchMetrics(metrics=[], cost=zero_cost)

        results: list[MetricResult] = []
        results.extend(self._compute_rouge(output_text, reference_text))
        results.append(self._compute_bleu(output_text, reference_text))
        results.append(self._compute_meteor(output_text, reference_text))
        return RefmatchMetrics(
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
