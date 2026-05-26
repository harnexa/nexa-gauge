"""Shared helpers for score-based metric nodes.

These are intentionally small and explicit so metric modules can stay readable
without repeating the same score/normalization code.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Type

from ng_core.types import ScoringMode
from ng_graph.nodes.metrics.scoring import normalize_raw_score, scoring_spec
from pydantic import BaseModel, Field, create_model


@lru_cache(maxsize=8)
def build_scores_response_model(
    model_prefix: str, mode: ScoringMode, include_reasoning: bool
) -> "Type[BaseModel]":
    """Create shared `{scores, reasoning?}` schema for judge outputs."""
    spec = scoring_spec(mode)
    scores_field = (
        list[int],
        Field(
            default_factory=list,
            description=f"Integer {spec.score_min}-{spec.score_max} per item",
        ),
    )
    fields: dict[str, tuple[object, object]] = {"scores": scores_field}
    if include_reasoning:
        fields["reasoning"] = (str, ...)

    suffix = "with_reason" if include_reasoning else "scores_only"
    return create_model(f"{model_prefix}_{mode.value}_{suffix}", **fields)


def normalize_score_value(raw_value: Any, mode: ScoringMode) -> float:
    """Map a raw claim score into shared [0, 1] score space."""
    try:
        raw_int = int(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return normalize_raw_score(raw_int, mode)


def raw_int_from_score(raw_value: Any) -> int:
    """Coerce a raw score into integer form."""
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0
