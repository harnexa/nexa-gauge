"""Shared prompt + scoring helpers for LLM-as-a-judge metric nodes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Type

from ng_core.types import ScoringMode
from pydantic import BaseModel, Field, create_model, model_validator



PASSED_VERDICT = "PASSED"
FAILED_VERDICT = "FAILED"


def verdict_from_passed(passed: bool) -> str:
    """Map a boolean metric outcome to the standardized verdict label."""
    return PASSED_VERDICT if passed else FAILED_VERDICT


def verdict_from_score(score: float | None, threshold: float) -> str | None:
    """Derive standardized verdict from a numeric score using a pass threshold."""
    if score is None:
        return None
    return verdict_from_passed(score >= threshold)


class ScoringSpec(BaseModel):
    """Rendered output-contract block for a scoring configuration."""

    mode: ScoringMode
    include_reasoning: bool
    reasoning_field: str = "reasoning"

    score_min: int = 0
    score_max: int = 0
    contract: str = ""

    @model_validator(mode="after")
    def _build_contract(self) -> "ScoringSpec":
        if self.mode == ScoringMode.BINARY_YES_NO:
            score_min, score_max = 0, 1
            contract = (
                "## Output contract (binary_1_0): \nreturn JSON with key `scores` as a list of "
                "scores in the same order (1 = supported/relevant, 0 = unsupported/not relevant)."
            )
            example = (
                '{\n'
                '  "scores": [1, 0 ..]\n'
                '}'
                if not self.include_reasoning
                else (
                    '{\n'
                    '  "scores": [1, 0 ..],\n'
                    f'  "{self.reasoning_field}": "Brief evidence-based explanation."\n'
                    '}'
                )
            )
        else:
            score_min, score_max = 1, 5
            contract = (
                "## Output contract (scale_1_5): \nreturn JSON with key `scores` as a list of integers "
                "in the same order. Use 1 for not supported/not relevant at all, 5 for fully "
                "supported/relevant, and 2-4 for intermediate strength."
            )
            example = (
                '{\n'
                '  "scores": [1, 3, 5 ..]\n'
                '}'
                if not self.include_reasoning
                else (
                    '{\n'
                    '  "scores": [1, 3, 5 ..],\n'
                    f'  "{self.reasoning_field}": "Brief evidence-based explanation."\n'
                    '}'
                )
            )

        if self.include_reasoning:
            contract += " Additionally, provide a brief overall evidence-based explanation for your scores."

        contract = (
            f"{contract}\nDo not include any extra keys.\n"
            f"Example:\n```json\n{example}\n```"
        )

        self.score_min = score_min
        self.score_max = score_max
        self.contract = contract
        return self


def normalize_raw_score(raw: float | int, score_contract: ScoringSpec) -> float:
    """Map a raw judge score into shared ``[0, 1]`` space with min/max scaling.
    
    Formula:
    normalized = (raw - score_min) / (score_max - score_min)
    then clamped to [0.0, 1.0].

    For binary_yes_no (score_min=0, score_max=1):

    - raw=0 -> (0-0)/(1-0)=0.0
    - raw=1 -> (1-0)/(1-0)=1.0
    - raw=2 -> 2.0 -> clamped to 1.0

    For scale_1_5 (score_min=1, score_max=5):

    - raw=1 -> (1-1)/4=0.0
    - raw=3 -> (3-1)/4=0.5
    - raw=5 -> (5-1)/4=1.0
    - raw=0 -> -0.25 -> clamped to 0.0
    """
    span = score_contract.score_max - score_contract.score_min
    if span <= 0:
        return 0.0
    normalized = (float(raw) - score_contract.score_min) / span
    return max(0.0, min(1.0, normalized))



@lru_cache(maxsize=8)
def build_scores_response_model(
    model_prefix: str, mode_value: str, min_score: float, max_score: float, include_reasoning: bool
) -> "Type[BaseModel]":
    """Create shared `{scores, reasoning?}` schema for judge outputs."""
    fields: dict[str, tuple[object, object]] = {
        "scores": (
            list[int],
            Field(
                default_factory=list,
                description=f"Integer {min_score}-{max_score} per item",
            ),
        ),
    }
    if include_reasoning:
        fields["reasoning"] = (str, ...)

    suffix = "with_reason" if include_reasoning else "scores_only"
    return create_model(f"{model_prefix}_{mode_value}_{suffix}", **fields)

