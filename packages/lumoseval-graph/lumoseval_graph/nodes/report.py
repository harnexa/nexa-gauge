"""Declarative per-record report aggregation.

REPORT_VISIBILITY defines the output shape — nested dict keys become output keys,
string leaves are dot-path expressions resolved from the EvalCase state root.
SECTION_GATES controls which sections are omitted when their artifact is None.
"""

from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Declarative config: defines exactly what appears in the report output.
#
# - Keys = output field names (the structure of the report)
# - String leaf values = dot-path expressions resolved from EvalCase state
# - [*] in a path = iterate over a list and extract the remaining path per item
# - List spec [base_path, {field: sub_path}] = iterate base_path items,
#   project sub_path fields from each → produces a list of dicts
# ---------------------------------------------------------------------------

REPORT_VISIBILITY: dict[str, Any] = {
    "target_node": "target_node",
    "input": {
        "case_id": "inputs.case_id",
        "generation": "inputs.generation.text",
        "question": "inputs.question.text",
        "context": "inputs.context.text",
        "reference": "inputs.reference.text",
    },
    "chunks": {
        "text": "generation_chunk.chunks[*].item.text",
        "cost": {
            "cost": "generation_chunk.cost.cost",
            "input_tokens": "generation_chunk.cost.input_tokens",
            "output_tokens": "generation_chunk.cost.output_tokens",
        },
    },
    "claims": {
        "text": "generation_claims.claims[*].item.text",
        "cost": {
            "cost": "generation_claims.cost.cost",
            "input_tokens": "generation_claims.cost.input_tokens",
            "output_tokens": "generation_claims.cost.output_tokens",
        },
    },
    "claims_unique": {
        "text": "generation_dedup_claims.claims[*].item.text",
        "cost": {
            "cost": "generation_dedup_claims.cost.cost",
            "input_tokens": "generation_dedup_claims.cost.input_tokens",
            "output_tokens": "generation_dedup_claims.cost.output_tokens",
        },
    },
    "geval_steps": {
        "names": "geval_steps.resolved_steps[*].name",
        "steps_source": "geval_steps.resolved_steps[*].steps_source",
        "evaluation_steps": "geval_steps.resolved_steps[*].evaluation_steps.text",
        "cost": {
            "cost": "geval_steps.cost.cost",
            "input_tokens": "geval_steps.cost.input_tokens",
            "output_tokens": "geval_steps.cost.output_tokens",
        },
    },
    "grounding": {
        "metrics": ["grounding_metrics.metrics[*]", {"name": "name", "score": "score", "result": "result"}],
        "cost": {
            "cost": "grounding_metrics.cost.cost",
            "input_tokens": "grounding_metrics.cost.input_tokens",
            "output_tokens": "grounding_metrics.cost.output_tokens",
        },
    },
    "relevance": {
        "metrics": ["relevance_metrics.metrics[*]", {"name": "name", "score": "score", "result": "result"}],
        "cost": {
            "cost": "relevance_metrics.cost.cost",
            "input_tokens": "relevance_metrics.cost.input_tokens",
            "output_tokens": "relevance_metrics.cost.output_tokens",
        },
    },
    "redteam": {
        "metrics": ["redteam_metrics.metrics[*]", {"name": "name", "score": "score", "result": "result"}],
        "cost": {
            "cost": "redteam_metrics.cost.cost",
            "input_tokens": "redteam_metrics.cost.input_tokens",
            "output_tokens": "redteam_metrics.cost.output_tokens",
        },
    },
    "geval": {
        "metrics": ["geval_metrics.metrics[*]", {"name": "name", "score": "score", "result": "result"}],
        "cost": {
            "cost": "geval_metrics.cost.cost",
            "input_tokens": "geval_metrics.cost.input_tokens",
            "output_tokens": "geval_metrics.cost.output_tokens",
        },
    },
    "reference": {
        "metrics": ["reference_metrics.metrics[*]", {"name": "name", "score": "score", "result": "result"}],
        "cost": {
            "cost": "reference_metrics.cost.cost",
            "input_tokens": "reference_metrics.cost.input_tokens",
            "output_tokens": "reference_metrics.cost.output_tokens",
        },
    },
}

# ---------------------------------------------------------------------------
# Section gates: if the mapped state key is None, the section is omitted.
# Sections not listed here (target_node, input) are always included.
# ---------------------------------------------------------------------------

SECTION_GATES: dict[str, str] = {
    "chunks": "generation_chunk",
    "claims": "generation_claims",
    "claims_unique": "generation_dedup_claims",
    "geval_steps": "geval_steps",
    "grounding": "grounding_metrics",
    "relevance": "relevance_metrics",
    "redteam": "redteam_metrics",
    "geval": "geval_metrics",
    "reference": "reference_metrics",
}


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _extract_path(value: Any, path: str) -> Any:
    """Walk a dot-separated path through nested dicts, with [*] list wildcards.

    Examples:
        _extract_path({"a": {"b": 1}}, "a.b")           -> 1
        _extract_path({"items": [{"x": 1}, {"x": 2}]},
                       "items[*].x")                     -> [1, 2]
        _extract_path({"a": None}, "a.b")                -> None
        _extract_path({"a": None}, "a[*].b")             -> []
    """
    segments = [seg for seg in path.split(".") if seg]
    if not segments:
        return value

    def _walk(current: Any, index: int) -> Any:
        if index >= len(segments):
            return current

        seg = segments[index]
        wildcard = seg.endswith("[*]")
        key = seg[:-3] if wildcard else seg

        if key:
            if not isinstance(current, Mapping):
                return [] if wildcard else None
            current = current.get(key)

        if wildcard:
            if not isinstance(current, list):
                return []
            return [_walk(item, index + 1) for item in current]

        return _walk(current, index + 1)

    return _walk(value, 0)


def resolve_path(state: Mapping[str, Any], path: str) -> Any:
    """Extract a value from state using a dot-path with [*] wildcards.

    First segment is the state key (dict lookup). The value is converted to a
    plain dict via model_dump() if it's a Pydantic model. Remaining segments
    are resolved via _extract_path.
    """
    dot = path.find(".")
    if dot == -1:
        return state.get(path)
    top_key, rest = path[:dot], path[dot + 1:]
    value = state.get(top_key)
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return _extract_path(value, rest)


def resolve_section(state: Mapping[str, Any], spec: str | dict | list) -> Any:
    """Resolve a REPORT_VISIBILITY section against state.

    String specs are resolved as dot-paths. Dict specs are recursed —
    each key becomes an output key, each value is resolved recursively.
    List specs [base_path, projection] iterate items from base_path and
    project sub-fields from each, producing a list of dicts.
    """
    if isinstance(spec, str):
        return resolve_path(state, spec)
    if isinstance(spec, list):
        base_path, projection = spec
        items = resolve_path(state, base_path)
        if not isinstance(items, list):
            return []
        return [
            {key: _extract_path(item, path) for key, path in projection.items()}
            for item in items
        ]
    return {key: resolve_section(state, child) for key, child in spec.items()}


def aggregate(*, state: Mapping[str, Any]) -> dict[str, Any]:
    """Build report by resolving REPORT_VISIBILITY against state.

    Sections gated on a None state key (per SECTION_GATES) are omitted.
    """
    result: dict[str, Any] = {}
    for section, spec in REPORT_VISIBILITY.items():
        gate_key = SECTION_GATES.get(section)
        if gate_key is not None and state.get(gate_key) is None:
            continue
        result[section] = resolve_section(state, spec)
    return result
