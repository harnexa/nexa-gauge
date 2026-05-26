"""Shared prompt + normalization helpers for all LLM-as-a-judge metric nodes.

Each metric node (geval, grounding, relevance, redteam) supports the same two
knobs:

1. ``scoring_mode`` — ``scale_1_5`` (judge returns integer 1-5) or
   ``binary_yes_no`` (judge returns 0/1).
2. ``include_reasoning`` — whether the judge also emits a short rationale.

Per-node SYSTEM/USER prompts stay where they are. This module supplies the
two prompt fragments the nodes append (score-instruction + reasoning-clause)
and the one normalization function that maps any raw integer score into the
shared ``[0, 1]`` reporting space.
"""

from __future__ import annotations

from dataclasses import dataclass

from ng_core.types import ScoringMode


@dataclass(frozen=True)
class ScoringSpec:
    """Numeric bounds and prompt instruction for one scoring mode."""

    mode: ScoringMode
    score_min: int
    score_max: int
    score_instruction: str


_SCALE_SPEC = ScoringSpec(
    mode=ScoringMode.SCALE_1_5,
    score_min=1,
    score_max=5,
    score_instruction=(
        "Score on the integer scale 1-5: 1 = fails every criterion, "
        "5 = perfectly satisfies every criterion."
    ),
)

_BINARY_SPEC = ScoringSpec(
    mode=ScoringMode.BINARY_YES_NO,
    score_min=0,
    score_max=1,
    score_instruction=(
        "Make a binary decision: 1 if the output satisfies the criterion, "
        "0 otherwise. Return the numeric score only as 0 or 1."
    ),
)


def scoring_spec(mode: ScoringMode) -> ScoringSpec:
    """Return the spec (bounds + instruction text) for the given mode."""
    if mode == ScoringMode.BINARY_YES_NO:
        return _BINARY_SPEC
    return _SCALE_SPEC


def normalize_raw_score(raw: float | int, mode: ScoringMode) -> float:
    """Map a raw judge integer into the shared ``[0, 1]`` space.

    Uses min/max scaling: ``(raw - score_min) / (score_max - score_min)``.
    Clamped to ``[0, 1]`` so an out-of-range judge response can't poison
    aggregations downstream.
    """
    spec = scoring_spec(mode)
    span = spec.score_max - spec.score_min
    if span <= 0:
        return 0.0
    normalized = (float(raw) - spec.score_min) / span
    return max(0.0, min(1.0, normalized))


def build_reasoning_clause(include_reasoning: bool) -> str:
    """Return the prompt fragment that either asks for or suppresses reasoning."""
    if include_reasoning:
        return (
            "Include a short `reasoning` field (one to two sentences) explaining "
            "the decision. Cite concrete evidence; do not restate the numeric score."
        )
    return (
        "Do not include any rationale, explanation, or extra fields beyond the "
        "required schema."
    )


def build_metric_system_prompt(
    base_system_prompt: str, mode: ScoringMode, include_reasoning: bool
) -> str:
    """Compose a metric system prompt from base intent only.

    Scoring + reasoning instructions are intentionally emitted via the shared
    output-contract block in user prompts.
    """
    del mode, include_reasoning
    return base_system_prompt


def build_score_output_contract(
    mode: ScoringMode, include_reasoning: bool, *, reasoning_field: str = "reasoning"
) -> str:
    """Return a shared output-format contract for claim-level scores lists."""
    if mode == ScoringMode.BINARY_YES_NO:
        contract = (
            "## Output contract (binary_1_0): \nreturn JSON with key `scores` as a list of "
            "scores in the same order (1 = supported/relevant, 0 = unsupported/not relevant)."
        )
        if include_reasoning:
            contract += " Additionally, provide a brief overall evidence-based explanation for your scores."
        example = (
            '{\n'
            '  "scores": [1, 0 ..]\n'
            '}'
            if not include_reasoning
            else (
                '{\n'
                '  "scores": [1, 0 ..],\n'
                f'  "{reasoning_field}": "Brief evidence-based explanation."\n'
                '}'
            )
        )
    else:
        contract = (
            "## Output contract (scale_1_5): \nreturn JSON with key `scores` as a list of integers "
            "in the same order. Use 1 for not supported/not relevant at all, 5 for fully "
            "supported/relevant, and 2-4 for intermediate strength."
        )
        if include_reasoning:
            contract += " Additionally, provide a brief overall evidence-based explanation for your scores."
        example = (
            '{\n'
            '  "scores": [1, 3, 5 ..]\n'
            '}'
            if not include_reasoning
            else (
                '{\n'
                '  "scores": [1, 3, 5 ..],\n'
                f'  "{reasoning_field}": "Brief evidence-based explanation."\n'
                '}'
            )
        )
    return f"{contract}\nDo not include any extra keys.\nExample:\n```json\n{example}\n```"
