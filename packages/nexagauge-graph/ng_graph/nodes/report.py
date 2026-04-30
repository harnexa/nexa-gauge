"""Topology-driven per-record report aggregation.

Report contract
---------------
1. ``target_node`` and normalized ``input`` are always included.
2. For every node in ``topology.PIPELINE``:
   - if ``state_key`` is defined and ``state[state_key]`` is not ``None``,
     the section is included under that exact ``state_key``.
   - if the value is ``None``, the section is omitted.
3. Projection strategy is selected from ``NodeSpec``:
   - ``artifact_out_kind == "chunks"``  -> ``{"text": [...], "cost": ...}``
   - ``artifact_out_kind == "claims"``  -> ``{"text": [...], "cost": ...}``
   - ``is_metric == True``              -> ``{"metrics": [...], "cost": ...}``
   - otherwise                          -> generic ``model_dump``/dict conversion.

When adding a new node
----------------------
Usually no report-code change is needed if the node has a proper ``state_key``
and one of the supported projection contracts above.

You only need to edit this file when:
- the node needs a custom presentation shape not covered by existing projection
  rules, or
- you introduce a new artifact kind that should be rendered specially.
"""

from __future__ import annotations

from typing import Any, Mapping

from ng_graph.topology import PIPELINE, NodeSpec


def _to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dict(v) for v in value]
    return value


def _input_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    inputs = state.get("inputs")
    return {
        "case_id": getattr(inputs, "case_id", None),
        "generation": getattr(getattr(inputs, "generation", None), "text", None),
        "question": getattr(getattr(inputs, "question", None), "text", None),
        "context": getattr(getattr(inputs, "context", None), "text", None),
        "reference": getattr(getattr(inputs, "reference", None), "text", None),
    }


def _project_chunk_artifact(artifact: Any) -> dict[str, Any]:
    chunks = getattr(artifact, "chunks", None) or []
    cost = getattr(artifact, "cost", None)
    return {
        "text": [getattr(getattr(chunk, "item", None), "text", None) for chunk in chunks],
        "cost": _to_dict(cost),
    }


def _project_claim_artifact(artifact: Any) -> dict[str, Any]:
    claims = getattr(artifact, "claims", None) or []
    cost = getattr(artifact, "cost", None)
    return {
        "text": [getattr(getattr(claim, "item", None), "text", None) for claim in claims],
        "cost": _to_dict(cost),
    }


def _project_metric_wrapper(wrapper: Any) -> dict[str, Any]:
    metrics = getattr(wrapper, "metrics", None) or []
    cost = getattr(wrapper, "cost", None)
    rows = []
    for metric in metrics:
        rows.append(
            {
                "name": getattr(metric, "name", None),
                "score": getattr(metric, "score", None),
                "verdict": getattr(metric, "verdict", None),
                "result": _to_dict(getattr(metric, "result", None)),
                "error": getattr(metric, "error", None),
            }
        )
    return {"metrics": rows, "cost": _to_dict(cost)}


def _project_by_spec(spec: NodeSpec, value: Any) -> Any:
    if spec.artifact_out_kind == "chunks":
        return _project_chunk_artifact(value)
    if spec.artifact_out_kind == "claims":
        return _project_claim_artifact(value)
    if spec.is_metric:
        return _project_metric_wrapper(value)
    return _to_dict(value)


def aggregate(*, state: Mapping[str, Any]) -> dict[str, Any]:
    """Build one per-case report payload from topology + state.

    This function is intentionally declarative: report visibility is driven by
    ``PIPELINE`` and section presence is driven by non-``None`` state values.
    """
    result: dict[str, Any] = {
        "target_node": state.get("target_node"),
        "input": _input_projection(state),
    }

    for spec in PIPELINE:
        if not spec.state_key:
            continue
        value = state.get(spec.state_key)
        if value is None:
            continue
        result[spec.state_key] = _project_by_spec(spec, value)

    return result
