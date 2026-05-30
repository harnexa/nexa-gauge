"""Default-redteam metric resolution helpers."""

from __future__ import annotations

from ng_core.types import RedteamMetricInput

from .bias import build_default_bias_metric
from .toxicity import build_default_toxicity_metric


def resolve_redteam_metrics(
    metrics: list[RedteamMetricInput] | None,
) -> list[RedteamMetricInput]:
    """Merge user metrics with defaults, replacing by metric name when duplicated."""
    merged: list[RedteamMetricInput] = [
        build_default_bias_metric(),
        build_default_toxicity_metric(),
    ]
    if not metrics:
        return merged

    for candidate in metrics:
        replaced = False
        for idx, existing in enumerate(merged):
            if existing.name == candidate.name:
                merged[idx] = candidate
                replaced = True
                break
        if not replaced:
            merged.append(candidate)
    return merged
