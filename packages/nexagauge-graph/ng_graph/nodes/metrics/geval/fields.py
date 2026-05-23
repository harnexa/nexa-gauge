"""Shared display-name mapping for G-Eval item fields.

The step-output and scoring prompts both refer to case fields by
human-readable names. Keeping the map in one module guarantees the two prompts
stay consistent — if scoring renames ``"Output"`` to ``"Generation"``,
step output follows automatically.
"""

from __future__ import annotations

FIELD_DISPLAY_NAMES: dict[str, str] = {
    "input": "Input",
    "output": "Output",
    "reference": "Reference",
    "context": "Context",
}


def format_param_names(item_fields: list[str]) -> str:
    """Render item_fields as a human-readable, order-preserving comma list."""
    return ", ".join(FIELD_DISPLAY_NAMES.get(f, f) for f in item_fields)
