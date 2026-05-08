"""Single-source-of-truth resolver for raw input field aliases.

All callers that need to read a logical input field (``generation``,
``question``, ``reference``, ``context``, ``geval``, ``redteam``,
``case_id``) from a raw record must go through :func:`resolve_alias` so
that adding a new accepted vocabulary in
:data:`ng_core.constants.INPUT_FIELD_ALIASES` is sufficient — no other
file edits required.
"""

from __future__ import annotations

from typing import Any, Mapping

# ── Input Field Aliases ───────────────────────────────────────────────────────
# Accepted record field names per logical input. The scanner picks the first
# matching key from each tuple, so order = priority. Update here to extend the
# accepted vocabulary across all adapters/datasets.
INPUT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "case_id": ("case_id", "id"),
    "generation": ("generation", "response", "answer", "output", "completion"),
    "question": ("question", "query", "prompt"),
    "reference": ("reference", "ground_truth", "gold_answer", "label"),
    "context": ("context", "contexts", "documents"),
    "geval": ("geval",),
    "redteam": ("redteam",),
}


def resolve_alias(record: Any, logical_key: str, default: Any = None) -> Any:
    """Read ``logical_key`` from ``record`` using ``INPUT_FIELD_ALIASES`` order.

    Returns the first non-None value found across the alias tuple. Works on
    both dicts and attribute-bearing objects (Pydantic models). Falls back
    to ``default`` when no alias matches.
    """
    aliases = INPUT_FIELD_ALIASES.get(logical_key, (logical_key,))
    is_mapping = isinstance(record, Mapping)
    for key in aliases:
        value = record.get(key) if is_mapping else getattr(record, key, None)
        if value is not None:
            return value
    return default
