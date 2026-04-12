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
    Redteam,
    RedteamMetricInput,
    RedteamRubric,
)
from lumiseval_core.utils import _count_tokens

_GEVAL_ITEM_FIELDS = {"question", "generation", "reference", "context"}
_REDTEAM_ITEM_FIELDS = {"question", "generation", "reference", "context"}


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


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if text:
            normalized.append(text)
    return normalized


def _build_redteam_rubric(raw_rubric: Any) -> RedteamRubric | None:
    if not isinstance(raw_rubric, dict):
        return None

    normalized_keys: dict[str, Any] = {}
    for key, value in raw_rubric.items():
        normalized_key = _normalize_text(key).lower().replace("-", "_").replace(" ", "_")
        if normalized_key:
            normalized_keys[normalized_key] = value

    goal = _normalize_text(normalized_keys.get("goal"))
    violations = _normalize_text_list(normalized_keys.get("violations"))
    non_violations = _normalize_text_list(normalized_keys.get("non_violations"))

    if not goal or not violations:
        return None

    return RedteamRubric(
        goal=goal,
        violations=violations,
        non_violations=non_violations,
    )


def _build_geval(raw_geval: Any) -> Geval | None:
    accept_legacy_record_fields = hasattr(raw_geval, "model_dump")
    if hasattr(raw_geval, "model_dump"):
        raw_geval = raw_geval.model_dump()
    if not isinstance(raw_geval, dict):
        return None

    metrics_raw = raw_geval.get("metrics")
    if not isinstance(metrics_raw, list):
        return None

    metrics: list[GevalMetricInput] = []
    for metric_raw in metrics_raw:
        metric_is_model = hasattr(metric_raw, "model_dump")
        if hasattr(metric_raw, "model_dump"):
            metric_raw = metric_raw.model_dump()
        if not isinstance(metric_raw, dict):
            continue

        name = _normalize_text(metric_raw.get("name"))
        if not name:
            continue

        raw_item_fields = metric_raw.get("item_fields")
        if not isinstance(raw_item_fields, list) and (
            accept_legacy_record_fields or metric_is_model
        ):
            raw_item_fields = metric_raw.get("record_fields")
        item_fields: list[str] = []
        if isinstance(raw_item_fields, list):
            for field in raw_item_fields:
                normalized_field = _normalize_text(field)
                if normalized_field in _GEVAL_ITEM_FIELDS:
                    item_fields.append(normalized_field)
        if not item_fields:
            item_fields = ["generation"]

        raw_criteria = metric_raw.get("criteria")
        if hasattr(raw_criteria, "model_dump"):
            raw_criteria = raw_criteria.model_dump()
        if isinstance(raw_criteria, dict):
            criteria_text = _normalize_text(raw_criteria.get("text"))
        elif hasattr(raw_criteria, "text"):
            criteria_text = _normalize_text(getattr(raw_criteria, "text"))
        else:
            criteria_text = _normalize_text(raw_criteria)
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
                if hasattr(step, "model_dump"):
                    step = step.model_dump()
                if isinstance(step, dict):
                    step_text = _normalize_text(step.get("text"))
                elif hasattr(step, "text"):
                    step_text = _normalize_text(getattr(step, "text"))
                else:
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


def _build_redteam(raw_redteam: Any) -> Redteam | None:
    if hasattr(raw_redteam, "model_dump"):
        raw_redteam = raw_redteam.model_dump()
    if not isinstance(raw_redteam, dict):
        return None

    metrics_raw = raw_redteam.get("metrics")
    if not isinstance(metrics_raw, list):
        return None

    metrics: list[RedteamMetricInput] = []
    for metric_raw in metrics_raw:
        if hasattr(metric_raw, "model_dump"):
            metric_raw = metric_raw.model_dump()
        if not isinstance(metric_raw, dict):
            continue

        name = _normalize_text(metric_raw.get("name"))
        if not name:
            continue

        rubric = _build_redteam_rubric(metric_raw.get("rubric"))
        if rubric is None:
            continue

        raw_item_fields = metric_raw.get("item_fields")
        item_fields: list[str] = []
        if isinstance(raw_item_fields, list):
            for field in raw_item_fields:
                normalized_field = _normalize_text(field)
                if normalized_field in _REDTEAM_ITEM_FIELDS:
                    item_fields.append(normalized_field)
        if not item_fields:
            item_fields = ["generation"]

        metrics.append(
            RedteamMetricInput(
                name=name,
                rubric=rubric,
                item_fields=item_fields,
            )
        )

    if not metrics:
        return None
    return Redteam(metrics=metrics)


def _build_inputs(record: Mapping[str, Any], *, idx: int = 0) -> Inputs:
    case_id = _normalize_text(_pick_first(record, ["case_id", "id"], f"record-{idx}"))
    generation_text = _normalize_text(
        _pick_first(record, ["generation", "response", "answer", "output", "completion"])
    )
    question_text = _normalize_text(_pick_first(record, ["question", "query", "prompt"]))
    reference_text = _normalize_text(
        _pick_first(record, ["reference", "ground_truth", "gold_answer", "label"])
    )
    context_text = _normalize_context_text(_pick_first(record, ["context", "contexts", "documents"]))
    geval = _build_geval(_pick_first(record, ["geval"]))
    redteam = _build_redteam(_pick_first(record, ["redteam"]))

    return Inputs(
        case_id=case_id,
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
        redteam=redteam,
        has_generation=bool(generation_text),
        has_question=bool(question_text),
        has_reference=bool(reference_text),
        has_context=bool(context_text),
        has_geval=geval is not None,
        has_redteam=redteam is not None,
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
    result["inputs"] = _build_inputs(record, idx=idx)
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
