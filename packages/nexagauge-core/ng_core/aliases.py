"""Single-source-of-truth resolver for raw input field aliases.

All callers that need to read a logical input field (``output``, ``input``,
``reference``, ``context``, ``geval``, ``redteam``, ``case_id``) from a
raw record must go through :func:`resolve_alias` so that adding a new
accepted vocabulary in :data:`INPUT_FIELD_ALIASES` is sufficient — no
other file edits required.

The canonical names ``output`` and ``input`` are the first-class field
names. Legacy keys ``generation`` and ``question`` (and their long-time
aliases ``response``, ``answer``, ``completion``, ``query``, ``prompt``)
remain accepted as aliases for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Mapping

# ── Input Field Aliases ───────────────────────────────────────────────────────
# Accepted record field names per logical input. The scanner picks the first
# matching key from each tuple, so order = priority. Update here to extend the
# accepted vocabulary across all adapters/datasets.
INPUT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "case_id": ("case_id", "id"),
    "output": ("output", "generation", "response", "answer", "completion"),
    "input": ("input", "question", "query", "prompt"),
    "reference": ("reference", "ground_truth", "gold_answer", "label"),
    "context": ("context", "contexts", "documents"),
    "geval": ("geval",),
    "redteam": ("redteam",),
}


def extend_aliases(user_field_map: Mapping[str, str]) -> None:
    """Prepend user-supplied source columns to ``INPUT_FIELD_ALIASES`` in place.

    Each ``(canonical, source)`` pair makes ``source`` the highest-priority
    alias for ``canonical`` so that :func:`resolve_alias` picks it before the
    built-in vocabulary. Idempotent: re-adding the same pair is a no-op.

    Raises ``ValueError`` if ``canonical`` is not a known logical key, or if
    ``source`` is empty.
    """
    for canonical, source in user_field_map.items():
        if canonical not in INPUT_FIELD_ALIASES:
            valid = ", ".join(sorted(INPUT_FIELD_ALIASES))
            raise ValueError(
                f"Unknown logical key '{canonical}' in field mapping. Allowed: {valid}."
            )
        if not source:
            raise ValueError(f"Field mapping for '{canonical}' has empty source column name.")
        existing = INPUT_FIELD_ALIASES[canonical]
        if source not in existing:
            INPUT_FIELD_ALIASES[canonical] = (source,) + existing


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
