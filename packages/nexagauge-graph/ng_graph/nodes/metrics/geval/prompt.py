"""Prompt composition helpers for GEval scoring.

Keeps prompt text DRY across scoring combinations:
1. scoring mode (`likert_1_5` vs `binary_yes_no`)
2. reasoning on/off.

Example SYSPrompts:

"You are an evaluator that rates a model's output against the given evaluation steps.
You receive (a) a numbered list of evaluation steps, and (b) named test-case fields.
Score on the integer scale 1-5: 1 = fails every step, 5 = perfectly satisfies every step.
Return JSON matching the schema exactly: {\"score\": int, \"reason\": str}.
The \"reason\" must cite specific steps that drove the score; do not quote the numeric score in the reason text.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from ng_core.types import GevalScoringMode
from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from typing import Type


@dataclass(frozen=True)
class GevalScoringSpec:
    mode: GevalScoringMode
    score_min: int
    score_max: int
    score_instruction: str


_BASE_SCORE_PROMPT = (
    "You are an evaluator that rates a model's output against the given evaluation steps. "
    "You receive (a) a numbered list of evaluation steps, and (b) named test-case fields."
)

_BASE_REASONING_RULE = (
    'The "reason" must cite specific steps that drove the score; '
    "do not quote the numeric score in the reason text."
)


def scoring_spec(mode: GevalScoringMode) -> GevalScoringSpec:
    if mode == GevalScoringMode.BINARY_YES_NO:
        return GevalScoringSpec(
            mode=mode,
            score_min=0,
            score_max=1,
            score_instruction=(
                "Make a binary decision: yes=1 if the output satisfies the evaluation steps, "
                "no=0 otherwise. Return the numeric score only as 0 or 1."
            ),
        )
    return GevalScoringSpec(
        mode=GevalScoringMode.LIKERT_1_5,
        score_min=1,
        score_max=5,
        score_instruction=(
            "Score on the integer scale 1-5: 1 = fails every step, "
            "5 = perfectly satisfies every step."
        ),
    )


def build_score_system_prompt(*, mode: GevalScoringMode, include_reasoning: bool) -> str:
    spec = scoring_spec(mode)
    if include_reasoning:
        schema_clause = (
            'Return JSON matching the schema exactly: {"score": int, "reason": str}. '
            + _BASE_REASONING_RULE
        )
    else:
        schema_clause = (
            'Return JSON matching the schema exactly: {"score": int}. '
            "Do not include explanation, analysis, or any field other than score."
        )
    return f"{_BASE_SCORE_PROMPT} {spec.score_instruction} {schema_clause}"


@lru_cache(maxsize=16)
def score_response_model(mode: GevalScoringMode, include_reasoning: bool) -> "Type[BaseModel]":
    """Return a cached dynamic schema with mode-specific score bounds."""
    spec = scoring_spec(mode)

    model_name = f"GevalScore_{mode.value}_{'with_reason' if include_reasoning else 'score_only'}"
    fields: dict[str, tuple[object, object]] = {
        "score": (int, Field(..., ge=spec.score_min, le=spec.score_max)),
    }
    if include_reasoning:
        fields["reason"] = (str, ...)
    return create_model(model_name, **fields)
