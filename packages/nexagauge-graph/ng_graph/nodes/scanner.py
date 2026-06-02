"""Simple per-record scanner for graph-side EvalCase input hydration.

This scanner is intentionally minimal: it takes one raw input record (like one item
from sample.json) and fills the ``inputs`` field expected by the graph-side EvalCase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypedDict

from ng_core.aliases import resolve_alias
from ng_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from ng_core.types import (
    Geval,
    GevalMetricInput,
    Grounding,
    Inputs,
    Item,
    Redteam,
    RedteamMetricInput,
    RedteamRubric,
    Refalign,
    Relevance,
    ScoringMode,
)
from ng_core.utils import _count_tokens
from ng_graph.nodes.metrics.redteam.defaults import resolve_redteam_metrics

_GEVAL_ITEM_FIELDS = {"input", "output", "reference", "context"}
_REDTEAM_ITEM_FIELDS = {"input", "output", "reference", "context"}


class GraphEvalCase(TypedDict, total=False):
    """Lightweight graph-side EvalCase shape used by this scanner."""

    case_id: str
    dataset: str
    split: str
    reference_files: list[str]
    inputs: Inputs


def build_scan_record(record: Mapping[str, Any], *, idx: int = 0) -> dict[str, Any]:
    """Build the canonical raw-record shape consumed by scanner parsing."""
    return {
        "case_id": _normalize_text(resolve_alias(record, "case_id", f"record-{idx}")),
        "output": resolve_alias(record, "output"),
        "input": resolve_alias(record, "input"),
        "reference": resolve_alias(record, "reference"),
        "context": resolve_alias(record, "context"),
        "geval": resolve_alias(record, "geval"),
        "grounding": resolve_alias(record, "grounding"),
        "relevance": resolve_alias(record, "relevance"),
        "redteam": resolve_alias(record, "redteam"),
        "refalign": resolve_alias(record, "refalign"),
    }


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


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = _normalize_text(value).lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _parse_scoring_knobs(raw: Mapping[str, Any]) -> tuple[ScoringMode, bool]:
    """Lenient parser for the shared `scoring_mode` / `include_reasoning` knobs.

    Returns the per-node defaults (binary + reasoning off) when either key is
    absent or has an unrecognised value, matching the conservative defaults on
    the typed config models.
    """
    raw_mode = raw.get("scoring_mode")
    mode = ScoringMode.BINARY_YES_NO
    if raw_mode:
        normalized_mode = _normalize_text(raw_mode).lower()
        mode_aliases = {
            "likert": ScoringMode.SCALE_1_5,
            "binary": ScoringMode.BINARY_YES_NO,
            "yes_no": ScoringMode.BINARY_YES_NO,
            "yesno": ScoringMode.BINARY_YES_NO,
        }
        if normalized_mode in mode_aliases:
            mode = mode_aliases[normalized_mode]
        else:
            try:
                mode = ScoringMode(normalized_mode)
            except ValueError:
                mode = ScoringMode.BINARY_YES_NO

    include_reasoning = _normalize_bool(raw.get("include_reasoning"), default=False)

    return mode, include_reasoning


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
    accept_legacy_item_fields = hasattr(raw_geval, "model_dump")
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
        if not isinstance(raw_item_fields, list) and (accept_legacy_item_fields or metric_is_model):
            raw_item_fields = metric_raw.get("item_fields")
        item_fields: list[str] = []
        if isinstance(raw_item_fields, list):
            for field in raw_item_fields:
                normalized_field = _normalize_text(field)
                if normalized_field in _GEVAL_ITEM_FIELDS:
                    item_fields.append(normalized_field)
        if not item_fields:
            item_fields = ["output"]

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
    scoring_mode, include_reasoning = _parse_scoring_knobs(raw_geval)
    return Geval(
        metrics=metrics,
        scoring_mode=scoring_mode,
        include_reasoning=include_reasoning,
    )


def _build_redteam(raw_redteam: Any) -> Redteam | None:
    if hasattr(raw_redteam, "model_dump"):
        raw_redteam = raw_redteam.model_dump()
    if not isinstance(raw_redteam, dict):
        return None

    scoring_mode, include_reasoning = _parse_scoring_knobs(raw_redteam)
    metrics_raw = raw_redteam.get("metrics")
    if metrics_raw is not None and not isinstance(metrics_raw, list):
        return None

    parsed_user_metrics: list[RedteamMetricInput] = []
    for metric_raw in metrics_raw or []:
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
            item_fields = ["output"]

        parsed_user_metrics.append(
            RedteamMetricInput(
                name=name,
                rubric=rubric,
                item_fields=item_fields,
            )
        )
    return Redteam(
        metrics=resolve_redteam_metrics(parsed_user_metrics),
        scoring_mode=scoring_mode,
        include_reasoning=include_reasoning,
    )


def _build_judge_only_config(raw: Any, factory):
    """Build a `Grounding` / `Relevance` config from a knobs-only record block.

    Returns the typed config when the record explicitly supplied the block
    (even an empty dict ``{}``), so the user's intent to opt into per-case
    config is preserved. Returns ``None`` when the key is absent so node
    defaults apply at run-time.
    """

    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return None
    scoring_mode, include_reasoning = _parse_scoring_knobs(raw)
    return factory(scoring_mode=scoring_mode, include_reasoning=include_reasoning)


def _build_refalign(raw_refalign: Any) -> Refalign | None:
    if raw_refalign is None:
        return None
    if hasattr(raw_refalign, "model_dump"):
        raw_refalign = raw_refalign.model_dump()
    if not isinstance(raw_refalign, dict):
        return None
    defaults = Refalign()
    atomic_chunks = _normalize_bool(
        raw_refalign.get("atomic_chunks"),
        default=defaults.atomic_chunks,
    )
    raw_threshold = raw_refalign.get("similarity_threshold")
    similarity_threshold = defaults.similarity_threshold
    if raw_threshold is not None:
        try:
            similarity_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            similarity_threshold = defaults.similarity_threshold
    similarity_threshold = max(0.0, min(1.0, similarity_threshold))
    raw_refine_top_k = raw_refalign.get("refine_top_k")
    refine_top_k = defaults.refine_top_k
    if raw_refine_top_k is not None:
        try:
            parsed = int(raw_refine_top_k)
            if parsed > 0:
                refine_top_k = parsed
        except (TypeError, ValueError):
            refine_top_k = None
    return Refalign(
        atomic_chunks=atomic_chunks,
        similarity_threshold=similarity_threshold,
        refine_top_k=refine_top_k,
    )


def _build_inputs(record: Mapping[str, Any], *, idx: int = 0) -> Inputs:
    del idx
    case_id = _normalize_text(record.get("case_id"))
    output_text = _normalize_text(record.get("output"))
    input_text = _normalize_text(record.get("input"))
    reference_text = _normalize_text(record.get("reference"))
    context_text = _normalize_context_text(record.get("context"))

    grounding = _build_judge_only_config(record.get("grounding"), Grounding)
    relevance = _build_judge_only_config(record.get("relevance"), Relevance)
    geval = _build_geval(record.get("geval"))
    redteam = _build_redteam(record.get("redteam"))
    refalign = _build_refalign(record.get("refalign"))

    return Inputs(
        case_id=case_id,
        output=Item(text=output_text, tokens=float(_count_tokens(output_text))),
        input=(
            Item(text=input_text, tokens=float(_count_tokens(input_text))) if input_text else None
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
        grounding=grounding,
        relevance=relevance,
        redteam=redteam,
        refalign=refalign,
        has_output=bool(output_text),
        has_input=bool(input_text),
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

    canonical = build_scan_record(record, idx=idx)
    result: GraphEvalCase = dict(case) if case is not None else {}
    result.setdefault("case_id", _normalize_text(canonical.get("case_id")))
    result.setdefault("dataset", DEFAULT_DATASET_NAME)
    result.setdefault("split", DEFAULT_SPLIT)
    result.setdefault("reference_files", [])
    result["inputs"] = _build_inputs(canonical, idx=idx)
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
    from ng_core.utils import pprint_model

    result = scan_file_record(path="sample.json", idx=5)
    pprint_model(result)
