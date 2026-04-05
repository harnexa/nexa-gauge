"""Shared raw-record canonicalization helpers for ingest adapters and scanner."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from lumiseval_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from lumiseval_core.errors import InputParseError
from lumiseval_core.types import EvalCase, GevalConfig, GevalMetricSpec

FieldCandidates = Mapping[str, Sequence[str]]

_DEFAULT_FIELD_CANDIDATES: dict[str, list[str]] = {
    "case_id": ["case_id", "id", "uuid", "prompt_id"],
    "generation": ["generation", "response", "answer", "output", "completion"],
    "question": ["question", "query", "prompt"],
    "reference": ["ground_truth", "reference", "gold_answer", "label", "answer"],
    "context": ["context", "contexts", "documents"],
    "reference_files": ["reference_files", "reference_paths"],
    "geval": ["geval"],
}


def pick_first(record: Mapping[str, Any], candidates: Sequence[str], default: Any = None) -> Any:
    """Return the first non-null field value present in the record."""
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_context(raw: Any) -> list[str]:
    """Normalize any context payload to a clean list[str]."""
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    value = str(raw).strip()
    return [value] if value else []


def normalize_reference_files(raw: Any) -> list[str]:
    """Normalize optional reference file paths to a clean list[str]."""
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    value = str(raw).strip()
    return [value] if value else []


def normalize_geval(raw: Any, *, record_index: int | None = None) -> GevalConfig | None:
    """Normalize strict GEval payload into typed GevalConfig."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputParseError(
            "geval must be an object with a 'metrics' list.",
            record_index=record_index,
        )

    metrics_raw = raw.get("metrics")
    if metrics_raw is None:
        raise InputParseError(
            "geval.metrics is required when 'geval' is provided.",
            record_index=record_index,
        )
    if not isinstance(metrics_raw, list):
        raise InputParseError(
            "geval.metrics must be a list.",
            record_index=record_index,
        )

    metrics: list[GevalMetricSpec] = []
    for idx, item in enumerate(metrics_raw):
        if not isinstance(item, dict):
            raise InputParseError(
                f"geval.metrics[{idx}] must be an object, got {type(item).__name__}.",
                record_index=record_index,
            )
        try:
            metrics.append(GevalMetricSpec.model_validate(item))
        except Exception as exc:
            raise InputParseError(
                f"Invalid geval.metrics[{idx}]: {exc}",
                record_index=record_index,
            ) from exc

    return GevalConfig(metrics=metrics)


def canonical_case_from_raw(
    raw: Mapping[str, Any],
    idx: int,
    *,
    dataset: str | None = None,
    split: str | None = None,
    field_candidates: FieldCandidates | None = None,
    metadata_mode: Literal["none", "extras", "full"] = "none",
) -> EvalCase:
    """Canonicalize one raw record into an EvalCase.

    Args:
        raw: Raw source row from JSON/JSONL/CSV/HF datasets.
        idx: Row index used for fallback case_id and error annotations.
        dataset: Optional dataset name override.
        split: Optional split override.
        field_candidates: Optional alias override for canonical fields.
        metadata_mode:
            - "none": do not attach metadata
            - "extras": attach non-consumed raw fields
            - "full": attach full raw row as metadata
    """
    row = dict(raw)
    candidates_by_field: dict[str, list[str]] = {
        key: list(values) for key, values in _DEFAULT_FIELD_CANDIDATES.items()
    }
    if field_candidates is not None:
        for field_name, candidates in field_candidates.items():
            candidates_by_field[field_name] = [str(name) for name in candidates]

    generation = pick_first(row, candidates_by_field["generation"])
    if generation is None or not str(generation).strip():
        raise InputParseError(
            "Record is missing required generation/response/answer/output/completion field.",
            record_index=idx,
        )

    case_id = str(
        pick_first(
            row,
            candidates_by_field["case_id"],
            default=f"record-{idx}",
        )
    )
    question = pick_first(row, candidates_by_field["question"])
    reference = pick_first(row, candidates_by_field["reference"])
    context = normalize_context(pick_first(row, candidates_by_field["context"]))
    reference_files = normalize_reference_files(
        pick_first(row, candidates_by_field["reference_files"])
    )
    geval = normalize_geval(
        pick_first(row, candidates_by_field["geval"]),
        record_index=idx,
    )

    metadata: dict[str, Any]
    if metadata_mode == "none":
        metadata = {}
    elif metadata_mode == "full":
        metadata = row
    else:
        consumed_keys = set().union(*candidates_by_field.values())
        metadata = {k: v for k, v in row.items() if k not in consumed_keys}

    return EvalCase(
        case_id=case_id,
        generation=str(generation),
        dataset=dataset or DEFAULT_DATASET_NAME,
        split=split or DEFAULT_SPLIT,
        question=str(question) if question is not None else None,
        reference=str(reference) if reference is not None else None,
        context=context,
        reference_files=reference_files,
        geval=geval,
        metadata=metadata,
    )
