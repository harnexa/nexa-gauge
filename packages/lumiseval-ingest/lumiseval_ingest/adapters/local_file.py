"""Local file dataset adapter (json/jsonl/csv/txt)."""

import csv
import json
from pathlib import Path
from typing import Any

from lumiseval_core.errors import InputParseError

from ..canonical import canonical_case_from_raw
from .base import DatasetAdapter


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
                yield canonical_case_from_raw(
                    record,
                    idx=idx,
                    dataset=self.dataset_name,
                    split=split,
                    metadata_mode="extras",
                )
            except InputParseError as exc:
                if exc.record_index is None:
                    raise InputParseError(str(exc), record_index=idx) from exc
                raise
