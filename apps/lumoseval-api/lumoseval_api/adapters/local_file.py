"""Local file dataset adapter (json/jsonl/csv/txt)."""

import csv
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from lumoseval_core.errors import InputParseError

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

    def _load_records_eager(self) -> list[dict[str, Any]]:
        """Eager loaders used for JSON arrays and plain text fallback."""
        suffix = self.path.suffix.lower()
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

        # Fallback: treat as raw text single case
        return [{"generation": self.path.read_text()}]

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        """Stream records when possible; fallback to eager paths when required."""
        suffix = self.path.suffix.lower()

        if suffix == ".jsonl":
            with self.path.open() as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise InputParseError(
                            f"Invalid JSONL at line {line_idx + 1}: {exc}"
                        ) from exc
                    if not isinstance(item, dict):
                        raise InputParseError(
                            f"JSONL line {line_idx + 1} must decode to an object."
                        )
                    yield item
            return

        if suffix == ".csv":
            with self.path.open(newline="") as f:
                for row in csv.DictReader(f):
                    yield dict(row)
            return

        for record in self._load_records_eager():
            yield record

    def iter_cases(
        self,
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
    ) -> Iterator[dict[str, Any]]:
        del seed  # deterministic by file order
        records = self._iter_records()
        if limit is not None:
            records = islice(records, limit)

        for idx, record in enumerate(records):
            try:
                yield record
            except InputParseError as exc:
                if exc.record_index is None:
                    raise InputParseError(str(exc), record_index=idx) from exc
                raise
