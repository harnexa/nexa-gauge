"""Report aggregation helpers.

Builds a flat JSON-style list of metrics from whichever metric nodes ran.
"""

from __future__ import annotations

from typing import Any

from lumiseval_core.types import CostEstimate, MetricResult


def _to_json_metric(metric: MetricResult) -> dict[str, Any]:
    if hasattr(metric, "model_dump"):
        return metric.model_dump()
    return {
        "name": getattr(metric, "name", None),
        "category": getattr(metric, "category", None),
        "score": getattr(metric, "score", None),
        "result": getattr(metric, "result", None),
        "error": getattr(metric, "error", None),
    }


def aggregate(
    job_id: str,
    grounding_metrics: list[MetricResult],
    relevance_metrics: list[MetricResult],
    redteam_metrics: list[MetricResult],
    geval_metrics: list[MetricResult],
    reference_metrics: list[MetricResult],
    cost_estimate: CostEstimate | None,
    cost_actual_usd: float,
) -> list[dict[str, Any]]:
    """Return a flat JSON list of available metric objects.

    This intentionally does no score combining or confidence computation.
    """
    del job_id, cost_estimate, cost_actual_usd

    all_metrics: list[MetricResult] = list(
        grounding_metrics + relevance_metrics + redteam_metrics + geval_metrics + reference_metrics
    )
    return [_to_json_metric(metric) for metric in all_metrics]
