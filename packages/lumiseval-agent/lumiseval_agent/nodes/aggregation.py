"""
Aggregation & Scoring Node — rolls up all metric node outputs into a single EvalReport.

Composite score formula (weights configurable via EvalJobConfig.score_weights):
  composite = 0.4 × faithfulness
            + 0.3 × (1 - hallucination_rate)
            + 0.2 × rubric_adherence
            + 0.1 × safety_score

Confidence band is the standard deviation of per-claim support scores.
"""

import statistics
from typing import Optional

from lumiseval_core.types import (
    ClaimVerdict,
    CostEstimate,
    DeepEvalMetricResult,
    EvalJobConfig,
    EvalReport,
    EvidenceResult,
    GiskardScanResult,
    RAGASMetricResult,
    RubricEvalResult,
)


def aggregate(
    job_id: str,
    claim_verdicts: list[EvidenceResult],
    ragas: Optional[RAGASMetricResult],
    deepeval: Optional[DeepEvalMetricResult],
    giskard: Optional[GiskardScanResult],
    rubric: Optional[RubricEvalResult],
    cost_estimate: Optional[CostEstimate],
    cost_actual_usd: float,
    job_config: EvalJobConfig,
) -> EvalReport:
    """Assemble all metric results into a final EvalReport."""
    weights = job_config.score_weights
    warnings: list[str] = []

    # Faithfulness
    faithfulness = ragas.faithfulness if (ragas and ragas.faithfulness is not None) else None

    # Hallucination rate (fraction of claims that are CONTRADICTED or UNVERIFIABLE)
    if claim_verdicts:
        non_supported = sum(
            1 for v in claim_verdicts if v.verdict != ClaimVerdict.SUPPORTED
        )
        hallucination_rate = non_supported / len(claim_verdicts)
    else:
        hallucination_rate = None

    # Rubric adherence
    rubric_score = rubric.composite_adherence_score if rubric else None

    # Safety score (average of privacy + bias, if available)
    safety_parts = []
    if deepeval:
        if deepeval.privacy_score is not None:
            safety_parts.append(deepeval.privacy_score)
        if deepeval.bias_score is not None:
            safety_parts.append(deepeval.bias_score)
    safety_score = sum(safety_parts) / len(safety_parts) if safety_parts else None

    # Composite score
    components: list[tuple[float, float]] = []
    if faithfulness is not None:
        components.append((weights.get("faithfulness", 0.4), faithfulness))
    if hallucination_rate is not None:
        components.append((weights.get("hallucination", 0.3), 1.0 - hallucination_rate))
    if rubric_score is not None:
        components.append((weights.get("rubric", 0.2), rubric_score))
    if safety_score is not None:
        components.append((weights.get("safety", 0.1), safety_score))

    composite_score: Optional[float] = None
    if components:
        total_weight = sum(w for w, _ in components)
        composite_score = sum(w * s for w, s in components) / total_weight

    # Confidence band — std dev of per-claim retrieval scores
    retrieval_scores = [
        p.retrieval_score
        for v in claim_verdicts
        for p in v.passages
    ]
    confidence_band: Optional[float] = None
    if len(retrieval_scores) >= 2:
        confidence_band = statistics.stdev(retrieval_scores)

    # Warnings
    if ragas and ragas.error:
        warnings.append(f"RAGAS error: {ragas.error}")
    if deepeval and deepeval.error:
        warnings.append(f"DeepEval error: {deepeval.error}")
    if giskard and not giskard.giskard_available:
        warnings.append("GISKARD_NOT_AVAILABLE: adversarial probe coverage reduced.")
    if cost_estimate and cost_estimate.approximate_warning:
        warnings.append(cost_estimate.approximate_warning)

    evaluation_incomplete = (
        ragas is None and deepeval is None and not claim_verdicts
    )

    return EvalReport(
        job_id=job_id,
        composite_score=round(composite_score, 4) if composite_score is not None else None,
        confidence_band=round(confidence_band, 4) if confidence_band is not None else None,
        claim_verdicts=claim_verdicts,
        ragas=ragas,
        deepeval=deepeval,
        giskard=giskard,
        rubric=rubric,
        cost_estimate=cost_estimate,
        cost_actual_usd=round(cost_actual_usd, 6),
        evaluation_incomplete=evaluation_incomplete,
        warnings=warnings,
    )
