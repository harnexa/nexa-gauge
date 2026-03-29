"""
Metadata Scanner — scans input generation(s) and returns token/chunk/claim estimates.

Reads raw text or JSON/JSONL/CSV dataset files. Makes no LLM calls.
TODO: Implement full dataset scanning logic.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import tiktoken
from lumiseval_core.constants import CHUNK_SIZE_TOKENS, CLAIMS_PER_CHUNK, TIKTOKEN_ENCODING
from lumiseval_core.errors import InputParseError
from lumiseval_core.types import EvalCase, InputMetadata
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

_ENCODING = tiktoken.get_encoding(TIKTOKEN_ENCODING)

TOKENS_PER_CHUNK = CHUNK_SIZE_TOKENS
ProgressCallback = Callable[[int, int], None]
_NODE_ORDER = [
    "scan",
    "estimate",
    "approve",
    "chunk",
    "claims",
    "dedupe",
    "relevance",
    "grounding",
    "redteam",
    "rubric",
    "eval",
]
_CONTEXT_NODES = {"chunk", "claims", "dedupe", "relevance", "grounding"}


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


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_context(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(
            _is_nonempty_text(item) or (item is not None and str(item).strip()) for item in value
        )
    return bool(str(value).strip())


def _has_rubric_rules(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


def _node_eligibility(
    *,
    has_generation: bool,
    has_context: bool,
    has_rubric_rules: bool,
) -> dict[str, bool]:
    flags = {node: has_generation for node in _NODE_ORDER}
    for node in _CONTEXT_NODES:
        flags[node] = has_generation and has_context
    flags["rubric"] = has_generation and has_rubric_rules
    return flags


def _accumulate_node_eligibility(
    rec: dict[str, Any],
    flags: dict[str, bool],
    *,
    eligible_record_count: dict[str, int],
    eligible_chunk_count: dict[str, int],
    eligible_claim_count: dict[str, int],
) -> None:
    for node, is_eligible in flags.items():
        if not is_eligible:
            continue
        eligible_record_count[node] += 1
        eligible_chunk_count[node] += int(rec["estimated_chunks"])
        eligible_claim_count[node] += int(rec["estimated_claims"])


def _make_rich_progress() -> Progress:
    return Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


def scan_text(
    generation: str,
    *,
    has_context: bool = False,
    has_rubric_rules: bool = False,
) -> InputMetadata:
    """Scan a single generation string and return metadata.

    Optional eligibility hints can be provided so downstream cost estimation can
    reflect which nodes are runnable for this record.
    """
    rec = _record_metadata(0, generation)
    flags = _node_eligibility(
        has_generation=_is_nonempty_text(generation),
        has_context=has_context,
        has_rubric_rules=has_rubric_rules,
    )
    rec["has_context"] = has_context
    rec["has_rubric_rules"] = has_rubric_rules
    rec["eligible_nodes"] = [node for node, ok in flags.items() if ok]

    eligible_record_count = {node: 0 for node in _NODE_ORDER}
    eligible_chunk_count = {node: 0 for node in _NODE_ORDER}
    eligible_claim_count = {node: 0 for node in _NODE_ORDER}
    _accumulate_node_eligibility(
        rec,
        flags,
        eligible_record_count=eligible_record_count,
        eligible_chunk_count=eligible_chunk_count,
        eligible_claim_count=eligible_claim_count,
    )

    return InputMetadata(
        record_count=1,
        total_tokens=rec["tokens"],
        total_chars=rec["chars"],
        estimated_chunk_count=rec["estimated_chunks"],
        estimated_claim_count=rec["estimated_claims"],
        per_record=[rec],
        eligible_record_count=eligible_record_count,
        eligible_chunk_count=eligible_chunk_count,
        eligible_claim_count=eligible_claim_count,
    )


def scan_cases(
    cases: Sequence[EvalCase],
    show_progress: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> InputMetadata:
    """Scan canonical EvalCase rows and return aggregate metadata.

    This is the preferred scanner for adapter-backed dataset flows because it
    works after schema normalization (aliases already resolved into EvalCase).
    """
    total = len(cases)
    per_record: list[dict[str, Any]] = []
    eligible_record_count = {node: 0 for node in _NODE_ORDER}
    eligible_chunk_count = {node: 0 for node in _NODE_ORDER}
    eligible_claim_count = {node: 0 for node in _NODE_ORDER}

    def _append_case(i: int, case: EvalCase) -> None:
        rec = _record_metadata(i, case.generation)
        rec["case_id"] = case.case_id

        has_context = _has_context(case.context)
        has_rubric_rules = len(case.rubric_rules) > 0
        flags = _node_eligibility(
            has_generation=_is_nonempty_text(case.generation),
            has_context=has_context,
            has_rubric_rules=has_rubric_rules,
        )
        rec["has_context"] = has_context
        rec["has_rubric_rules"] = has_rubric_rules
        rec["eligible_nodes"] = [node for node, ok in flags.items() if ok]

        _accumulate_node_eligibility(
            rec,
            flags,
            eligible_record_count=eligible_record_count,
            eligible_chunk_count=eligible_chunk_count,
            eligible_claim_count=eligible_claim_count,
        )
        per_record.append(rec)

    if show_progress and progress_callback is None:
        with _make_rich_progress() as progress:
            task = progress.add_task("Scanning records", total=total)
            for i, case in enumerate(cases):
                _append_case(i, case)
                progress.advance(task)
    else:
        for i, case in enumerate(cases):
            if show_progress and progress_callback is not None:
                progress_callback(i + 1, total)
            _append_case(i, case)

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
        eligible_record_count=eligible_record_count,
        eligible_chunk_count=eligible_chunk_count,
        eligible_claim_count=eligible_claim_count,
    )


def scan_file(
    path: str | Path,
    show_progress: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> InputMetadata:
    """Scan a local input file and return aggregate metadata used by cost estimation.

    Supported input types:
    - ``.jsonl``: one JSON object per line
    - ``.json``: either a single object or an array of objects
    - ``.csv``: parsed with ``csv.DictReader``
    - any other suffix: treated as a raw text generation (single-record scan)

    For structured dataset formats (JSON/JSONL/CSV), each record must contain a
    ``generation`` field. The scanner computes per-record token count, character
    count, estimated chunk count, and estimated claim count, then aggregates totals.

    Args:
        path: File path to scan.
        show_progress: If ``True``, emits per-record scan progress in the format
            ``[current/total] scanning ...``.
        progress_callback: Optional callback invoked as ``callback(current, total)``
            during record scanning. If omitted and ``show_progress=True``, a default
            console renderer is used.

    Returns:
        InputMetadata with:
        - ``record_count``
        - ``total_tokens``
        - ``total_chars``
        - ``estimated_chunk_count``
        - ``estimated_claim_count``
        - ``per_record`` breakdown for each scanned row
        - ``eligible_record_count`` per node (how many records can run that node)
        - ``eligible_chunk_count`` per node (chunk estimates across eligible records)
        - ``eligible_claim_count`` per node (claim estimates across eligible records)

    Raises:
        InputParseError: If any structured record is missing ``generation``.
        json.JSONDecodeError: If a JSON/JSONL payload cannot be decoded.
        OSError / UnicodeDecodeError: If the file cannot be read.

    Notes:
        - This function does not call any external LLM APIs.
        - Token counting uses ``tiktoken`` (``cl100k_base`` encoding).
        - Chunk and claim counts are heuristic estimates intended for pre-run budgeting.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    # try:
    if path.suffix == ".jsonl":
        with path.open() as f:
            for i, line in enumerate(f):
                # try:
                records.append(json.loads(line))
                # except json.JSONDecodeError as exc:
                #     raise InputParseError(str(exc), record_index=i) from exc
    elif path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    elif path.suffix == ".csv":
        with path.open(newline="") as f:
            records = list(csv.DictReader(f))
    else:
        # Treat as plain text — single record
        return scan_text(path.read_text())
    # except (OSError, UnicodeDecodeError) as exc:
    #     raise InputParseError(f"Cannot read {path}: {exc}") from exc

    total = len(records)
    per_record = []
    eligible_record_count = {node: 0 for node in _NODE_ORDER}
    eligible_chunk_count = {node: 0 for node in _NODE_ORDER}
    eligible_claim_count = {node: 0 for node in _NODE_ORDER}

    def _append_record(i: int, rec: dict[str, Any]) -> None:
        generation = rec.get("generation")
        if generation is None:
            raise InputParseError(
                f"Record at index {i} is missing required 'generation' field.",
                record_index=i,
            )
        meta = _record_metadata(i, str(generation))

        has_context = _has_context(rec.get("context", rec.get("contexts", rec.get("documents"))))
        has_rubric_rules = _has_rubric_rules(rec.get("rubric_rules", rec.get("rubric")))
        flags = _node_eligibility(
            has_generation=_is_nonempty_text(str(generation)),
            has_context=has_context,
            has_rubric_rules=has_rubric_rules,
        )
        meta["has_context"] = has_context
        meta["has_rubric_rules"] = has_rubric_rules
        meta["eligible_nodes"] = [node for node, ok in flags.items() if ok]

        _accumulate_node_eligibility(
            meta,
            flags,
            eligible_record_count=eligible_record_count,
            eligible_chunk_count=eligible_chunk_count,
            eligible_claim_count=eligible_claim_count,
        )
        per_record.append(meta)

    if show_progress and progress_callback is None:
        # Use Rich's live-rendering progress bar — updates regardless of loop speed
        with _make_rich_progress() as progress:
            task = progress.add_task("Scanning records", total=total)
            for i, rec in enumerate(records):
                _append_record(i, rec)
                progress.advance(task)
    else:
        for i, rec in enumerate(records):
            if show_progress and progress_callback is not None:
                progress_callback(i + 1, total)
            _append_record(i, rec)

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
        eligible_record_count=eligible_record_count,
        eligible_chunk_count=eligible_chunk_count,
        eligible_claim_count=eligible_claim_count,
    )
