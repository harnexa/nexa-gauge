"""
Metadata Scanner — scans input generation(s) and returns token/chunk/claim estimates.

Reads raw text or JSON/JSONL/CSV dataset files. Makes no LLM calls.
TODO: Implement full dataset scanning logic.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import tiktoken

from lumiseval_core.errors import InputParseError
from lumiseval_core.types import InputMetadata

# cl100k_base is compatible with GPT-4 and approximates Anthropic token counts.
_ENCODING = tiktoken.get_encoding("cl100k_base")

TOKENS_PER_CHUNK = 512
CLAIMS_PER_CHUNK = 3  # heuristic


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _record_metadata(idx: int, generation: str) -> dict[str, Any]:
    tokens = _count_tokens(generation)
    chunks = max(1, math.ceil(tokens / TOKENS_PER_CHUNK))
    return {
        "record_index": idx,
        "tokens": tokens,
        "chars": len(generation),
        "estimated_chunks": chunks,
        "estimated_claims": chunks * CLAIMS_PER_CHUNK,
    }


def scan_text(generation: str) -> InputMetadata:
    """Scan a single generation string and return metadata."""
    rec = _record_metadata(0, generation)
    return InputMetadata(
        record_count=1,
        total_tokens=rec["tokens"],
        total_chars=rec["chars"],
        estimated_chunk_count=rec["estimated_chunks"],
        estimated_claim_count=rec["estimated_claims"],
        per_record=[rec],
    )


def scan_file(path: str | Path) -> InputMetadata:
    """Scan a JSON, JSONL, or CSV dataset file.

    Each record must have a ``generation`` field.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    try:
        if path.suffix == ".jsonl":
            with path.open() as f:
                for i, line in enumerate(f):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise InputParseError(str(exc), record_index=i) from exc
        elif path.suffix == ".json":
            with path.open() as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
        elif path.suffix == ".csv":
            with path.open(newline="") as f:
                records = list(csv.DictReader(f))
        else:
            # Treat as plain text — single record
            generation = path.read_text()
            return scan_text(generation)
    except (OSError, UnicodeDecodeError) as exc:
        raise InputParseError(f"Cannot read {path}: {exc}") from exc

    per_record = []
    for i, rec in enumerate(records):
        if "generation" not in rec:
            raise InputParseError(
                f"Record at index {i} is missing required 'generation' field.", record_index=i
            )
        per_record.append(_record_metadata(i, rec["generation"]))

    totals: dict[str, int] = {
        k: sum(r[k] for r in per_record)
        for k in ("tokens", "chars", "estimated_chunks", "estimated_claims")
    }
    return InputMetadata(
        record_count=len(per_record),
        total_tokens=totals["tokens"],
        total_chars=totals["chars"],
        estimated_chunk_count=totals["estimated_chunks"],
        estimated_claim_count=totals["estimated_claims"],
        per_record=per_record,
    )
