"""Simple per-record scanner for graph-side EvalCase input hydration.

This scanner is intentionally minimal: it takes one raw input record (like one item
from sample.json) and fills the ``inputs`` field expected by the graph-side EvalCase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypedDict

from lumiseval_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from lumiseval_core.types import (
    Geval,
    GevalMetricInput,
    Item,
    Inputs,
)
from lumiseval_core.utils import _count_tokens

_GEVAL_ITEM_FIELDS = {"question", "generation", "reference", "context"}


class GraphEvalCase(TypedDict, total=False):
    """Lightweight graph-side EvalCase shape used by this scanner."""

    case_id: str
    dataset: str
    split: str
    reference_files: list[str]
    inputs: Inputs


def _pick_first(record: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_context_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return "\n\n".join(parts)
    return str(value).strip()


def _build_geval(raw_geval: Any) -> Geval | None:
    if not isinstance(raw_geval, dict):
        return None

    metrics_raw = raw_geval.get("metrics")
    if not isinstance(metrics_raw, list):
        return None

    metrics: list[GevalMetricInput] = []
    for metric_raw in metrics_raw:
        if not isinstance(metric_raw, dict):
            continue

        name = _normalize_text(metric_raw.get("name"))
        if not name:
            continue

        raw_item_fields = metric_raw.get("item_fields")
        item_fields: list[str] = []
        if isinstance(raw_item_fields, list):
            for field in raw_item_fields:
                normalized_field = _normalize_text(field)
                if normalized_field in _GEVAL_ITEM_FIELDS:
                    item_fields.append(normalized_field)
        if not item_fields:
            item_fields = ["generation"]

        criteria_text = _normalize_text(metric_raw.get("criteria"))
        criteria = (
            Item(
                text=criteria_text,
                tokens=float(_count_tokens(criteria_text)),
            )
            if criteria_text
            else None
        )

        steps_raw = metric_raw.get("evaluation_steps")
        steps: list[Item] = []
        if isinstance(steps_raw, list):
            for step in steps_raw:
                step_text = _normalize_text(step)
                if not step_text:
                    continue
                steps.append(
                    Item(
                        text=step_text,
                        tokens=float(_count_tokens(step_text)),
                    )
                )

        metrics.append(
            GevalMetricInput(
                name=name,
                item_fields=item_fields,
                criteria=criteria,
                evaluation_steps=steps,
            )
        )

    if not metrics:
        return None
    return Geval(metrics=metrics)


def _build_inputs(record: Mapping[str, Any]) -> Inputs:
    generation_text = _normalize_text(
        _pick_first(record, ["generation", "response", "answer", "output", "completion"])
    )
    question_text = _normalize_text(_pick_first(record, ["question", "query", "prompt"]))
    reference_text = _normalize_text(
        _pick_first(record, ["reference", "ground_truth", "gold_answer", "label"])
    )
    context_text = _normalize_context_text(_pick_first(record, ["context", "contexts", "documents"]))
    geval = _build_geval(_pick_first(record, ["geval"]))

    return Inputs(
        generation=Item(text=generation_text, tokens=float(_count_tokens(generation_text))),
        question=(
            Item(text=question_text, tokens=float(_count_tokens(question_text)))
            if question_text
            else None
        ),
        reference=(
            Item(text=reference_text, tokens=float(_count_tokens(reference_text)))
            if reference_text
            else None
        ),
        context=(
            Item(text=context_text, tokens=float(_count_tokens(context_text)))
            if context_text
            else None
        ),
        geval=geval,
        has_generation=bool(generation_text),
        has_question=bool(question_text),
        has_reference=bool(reference_text),
        has_context=bool(context_text),
        has_geval=geval is not None,
    )


def scan(
    record: Mapping[str, Any],
    idx: int = 0,
    case: GraphEvalCase | None = None,
) -> GraphEvalCase:
    """Fill and return graph EvalCase.inputs from one raw record."""

    result: GraphEvalCase = dict(case) if case is not None else {}
    result.setdefault("case_id", _normalize_text(_pick_first(record, ["case_id", "id"], f"record-{idx}")))
    result.setdefault("dataset", DEFAULT_DATASET_NAME)
    result.setdefault("split", DEFAULT_SPLIT)
    result.setdefault("reference_files", [])
    result["inputs"] = _build_inputs(record)
    return result


def scan_file_record(path: str | Path, idx: int = 0) -> GraphEvalCase:
    """Load one record from a JSON file and run ``scan`` on it."""

    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        if not data:
            raise ValueError("Input JSON list is empty")
        return scan(data[idx], idx=idx)
    if isinstance(data, dict):
        return scan(data, idx=idx)
    raise ValueError("Input JSON must be an object or a list of objects")


if __name__ == "__main__":
    from lumiseval_core.utils import pprint_model
    result = scan_file_record(path="sample.json", idx=5)
    pprint_model(result)
