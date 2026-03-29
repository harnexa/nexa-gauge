"""Local file dataset adapter (json/jsonl/csv/txt)."""

import csv
import json
from pathlib import Path
from typing import Any

from lumiseval_core.errors import InputParseError
from lumiseval_core.types import EvalCase, RubricRule

from .base import DatasetAdapter


def _normalize_context(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item is not None]
    return [str(raw)]


def _normalize_reference_files(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item is not None]
    return [str(raw)]


def _normalize_rubric_rules(raw: Any) -> list[RubricRule]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InputParseError("rubric_rules must be a list if provided.")

    rules: list[RubricRule] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            rules.append(
                RubricRule(
                    id=f"R-{idx + 1:03d}",
                    statement=item,
                    pass_condition="Rule must be satisfied.",
                )
            )
            continue
        if isinstance(item, dict):
            rules.append(RubricRule.model_validate(item))
            continue
        raise InputParseError(f"Unsupported rubric rule type at index {idx}: {type(item).__name__}")
    return rules


def _pick_first(record: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _canonical_case(record: dict[str, Any], idx: int, dataset_name: str, split: str) -> EvalCase:
    generation = _pick_first(record, ["generation", "response", "answer", "output", "completion"])
    if generation is None or str(generation).strip() == "":
        raise InputParseError(
            "Record is missing required generation/response/answer/output/completion field.",
            record_index=idx,
        )

    case_id = str(_pick_first(record, ["case_id", "id", "uuid", "prompt_id"], default=idx))
    question = _pick_first(record, ["question", "query", "prompt"])
    ground_truth = _pick_first(record, ["ground_truth", "reference", "gold_answer"])
    context = _normalize_context(_pick_first(record, ["context", "contexts", "documents"]))
    reference_files = _normalize_reference_files(
        _pick_first(record, ["reference_files", "reference_paths"])
    )
    rubric_rules = _normalize_rubric_rules(_pick_first(record, ["rubric_rules", "rubric"]))

    reserved = {
        "case_id",
        "id",
        "uuid",
        "prompt_id",
        "generation",
        "response",
        "answer",
        "output",
        "completion",
        "question",
        "query",
        "prompt",
        "ground_truth",
        "reference",
        "gold_answer",
        "context",
        "contexts",
        "documents",
        "reference_files",
        "reference_paths",
        "rubric_rules",
        "rubric",
    }
    metadata = {k: v for k, v in record.items() if k not in reserved}

    return EvalCase(
        case_id=case_id,
        generation=str(generation),
        dataset=dataset_name,
        split=split,
        question=str(question) if question is not None else None,
        ground_truth=str(ground_truth) if ground_truth is not None else None,
        context=context,
        reference_files=reference_files,
        rubric_rules=rubric_rules,
        metadata=metadata,
    )


class LocalFileDatasetAdapter(DatasetAdapter):
    """Dataset adapter for local files with flexible schema mapping."""

    def __init__(self, path: str | Path, dataset_name: str | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise InputParseError(f"Dataset file not found: {self.path}")
        self.dataset_name = dataset_name or self.path.stem

    @property
    def name(self) -> str:
        return "local_file"

    def _load_records(self) -> list[dict[str, Any]]:
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            records: list[dict[str, Any]] = []
            with self.path.open() as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise InputParseError(f"Invalid JSONL at line {idx + 1}: {exc}") from exc
                    if not isinstance(item, dict):
                        raise InputParseError(f"JSONL line {idx + 1} must decode to an object.")
                    records.append(item)
            return records

        if suffix == ".json":
            with self.path.open() as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                if not all(isinstance(item, dict) for item in payload):
                    raise InputParseError("JSON array dataset must contain only objects.")
                return payload
            raise InputParseError("JSON dataset must be an object or array of objects.")

        if suffix == ".csv":
            with self.path.open(newline="") as f:
                return list(csv.DictReader(f))

        # Fallback: treat as raw text single case
        return [{"generation": self.path.read_text()}]

    def iter_cases(
        self,
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
    ):
        del seed  # deterministic by file order
        records = self._load_records()
        for idx, record in enumerate(records):
            if limit is not None and idx >= limit:
                break
            try:
                yield _canonical_case(record, idx=idx, dataset_name=self.dataset_name, split=split)
            except InputParseError as exc:
                if exc.record_index is None:
                    raise InputParseError(str(exc), record_index=idx) from exc
                raise
