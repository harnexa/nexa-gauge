"""
Metadata Scanner — scans input cases and returns token/chunk/claim counts.

Tokens are counted with tiktoken (cl100k_base) and chunks are produced by the
actual chunk_text() function so counts are precise rather than heuristic.

Token breakdown per record:
  generation_tokens  — tokens in the model's generated response
  context_tokens     — tokens across all context passages for that record
  rubric_tokens      — tokens in rubric rule statements + pass conditions
  total_tokens       — sum of the three above

Chunk count is derived by running chunk_text() on every context passage.
Claim count remains a heuristic: chunk_count × CLAIMS_PER_CHUNK.
"""

import csv
import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import tiktoken
from lumiseval_core.constants import CLAIMS_PER_CHUNK, TIKTOKEN_ENCODING, CONTEXT_CHUNK_SIZE_TOKENS, GENERATION_CHUNK_SIZE_TOKENS
from lumiseval_core.errors import InputParseError
from lumiseval_core.pipeline import CONTEXT_REQUIRED_NODES as _CONTEXT_NODES
from lumiseval_core.pipeline import NODE_ORDER as _NODE_ORDER
from lumiseval_core.pipeline import RUBRIC_REQUIRED_NODES as _RUBRIC_NODES
from lumiseval_core.types import EvalCase, InputMetadata, Record, RecordMeta, Rubric
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from lumiseval_ingest.chunker import chunk_text

_ENCODING = tiktoken.get_encoding(TIKTOKEN_ENCODING)

ProgressCallback = Callable[[int, int], None]

_TOTALS_KEYS = (
    "tokens",
    "chars",
    "context_chunks",
    "generation_chunks",
    "estimated_claims",
    "generation_tokens",
    "context_tokens",
    "rubric_tokens",
)


# ── Token counting helpers ────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


# def _rubric_tokens_typed(rules: list[Rubric]) -> int:
#     """Sum tokens across a typed list of Rubric objects."""
#     return sum(_count_tokens(r.statement + " " + r.pass_condition) for r in rules)


def _rubric_tokens_raw(raw: Any) -> int:
    """Sum tokens across a raw rubric field (list of str or dict)."""
    if not raw or not isinstance(raw, list):
        return 0
    total = 0
    for rule in raw:
        if isinstance(rule, str):
            total += _count_tokens(rule)
        elif isinstance(rule, dict):
            total += _count_tokens(rule.get("statement", "") + " " + rule.get("pass_condition", ""))
    return total


# ── Context normalisation ─────────────────────────────────────────────────────


def _normalize_context(raw: Any) -> list[str]:
    """Normalise any context field value to a clean list[str]."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item) for item in raw if item and str(item).strip()]
    return [str(raw)] if str(raw).strip() else []


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_context(value: Any) -> bool:
    return bool(_normalize_context(value))


def _has_rubric(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


# ── Per-record metadata ───────────────────────────────────────────────────────


def _record_metadata(
    idx: int,
    case_id: str,
    question: Optional[str],
    generation: Optional[str],
    context: list[str]=[],
    rubric: list[str]=[],
) -> RecordMeta:
    """Compute token counts and actual chunk count for one record.

    Args:
        idx:               Record index in the dataset.
        generation:        Model generation text.
        context_passages:  Normalised list of context passage strings.
        rubric_tok:        Pre-computed rubric token count for this record.
    """
    question_token_count = _count_tokens(question) if question else 0
    generation_token_count = _count_tokens(generation) if generation else 0
    context_token_count = sum([_count_tokens(p) for p in context]) if context else 0
    rubric_token_count = sum([_count_tokens(r) for r in rubric]) if rubric else 0

    # Chunk generation text (used by the claims extraction node)
    generation_chunks = chunk_text(generation, GENERATION_CHUNK_SIZE_TOKENS) if generation else []
    generation_chunk_count = len(generation_chunks)

    # Claims are extracted from generation chunks
    estimated_claim_count = generation_chunk_count * CLAIMS_PER_CHUNK
    total_tokens = (
        question_token_count + 
        generation_token_count + 
        context_token_count + 
        rubric_token_count
    )

    return Record(
        record_index=idx,
        case_id=case_id,
        question=question,
        context=context,
        generation=generation,
        generation_chunks=generation_chunks,
        rubric=rubric,
        record_metadata=RecordMeta(
            context_token_count=context_token_count,
            generation_chunk_count=generation_chunk_count,
            generation_token_count=generation_token_count,
            rubric_token_count=rubric_token_count,
            total_token_count=total_tokens,
            estimated_claim_count=estimated_claim_count,
            has_context=True if context else False,
            has_rubric=True if rubric else False,
            eligible_nodes=
        )
    )



# ── Node eligibility ──────────────────────────────────────────────────────────


def _node_eligibility(
    *,
    has_generation: bool,
    has_context: bool,
    has_rubric: bool,
) -> dict[str, bool]:
    flags = {node: has_generation for node in _NODE_ORDER}
    for node in _CONTEXT_NODES:
        flags[node] = has_generation and has_context
    for node in _RUBRIC_NODES:
        flags[node] = has_generation and has_rubric
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
        # "chunk" node processes context passages; all other nodes work from
        # generation chunks (claims are extracted from generation text).
        if node == "chunk":
            eligible_chunk_count[node] += int(rec["context_chunks"])
        else:
            eligible_chunk_count[node] += int(rec["generation_chunks"])
        eligible_claim_count[node] += int(rec["estimated_claims"])


# ── Progress bar ──────────────────────────────────────────────────────────────


def _make_rich_progress() -> Progress:
    return Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def _build_metadata(per_record: list[dict[str, Any]], **kwargs: dict[str, int]) -> InputMetadata:
    """Aggregate per-record dicts into an InputMetadata object."""
    totals: dict[str, int] = {k: sum(r[k] for r in per_record) for k in _TOTALS_KEYS}
    record_meta = [
        Record(
            case_id=r.get("case_id", f"record-{r['record_index']}"),
            record_index=r["record_index"],
            question=r.get("question"),
            generation=r.get("generation"),
            rubric=r.get("rubric"),
            context=r.get("context"),
            record_metadata=RecordMeta(
                chars=r["chars"],
                context_chunks=r["context_chunks"],
                context_tokens=r["context_tokens"],
                estimated_chunks=r["estimated_chunks"],
                estimated_claims=r["estimated_claims"],
                generation_chunks=r["generation_chunks"],
                generation_tokens=r["generation_tokens"],
                has_context=r.get("has_context", False),
                has_rubric_rules=r.get("has_rubric", False),
                rubric_tokens=r["rubric_tokens"],
                tokens=r["tokens"],
                eligible_nodes=r.get("eligible_nodes", []),
            ),
        )
        for r in per_record
    ]

    return InputMetadata(
        record_count=len(per_record),
        total_tokens=totals["tokens"],
        total_chars=totals["chars"],
        estimated_claim_count=totals["estimated_claims"],
        generation_tokens=totals["generation_tokens"],
        context_tokens=totals["context_tokens"],
        rubric_tokens=totals["rubric_tokens"],
        context_chunk_count=totals["context_chunks"],
        generation_chunk_count=totals["generation_chunks"],
        record_meta=record_meta,
        **kwargs,
    )


def scan_text(
    generation: str,
    *,
    has_context: bool = False,
    has_rubric: bool = False,
) -> InputMetadata:
    """Scan a single generation string and return metadata.

    Optional eligibility hints can be provided so downstream cost estimation can
    reflect which nodes are runnable for this record.
    """
    rec = _record_metadata(0, generation, [], 0)
    rec["generation"] = generation
    flags = _node_eligibility(
        has_generation=_is_nonempty_text(generation),
        has_context=has_context,
        has_rubric=has_rubric,
    )
    rec["has_context"] = has_context
    rec["has_rubric"] = has_rubric
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

    return _build_metadata(
        [rec],
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
        context_passages = _normalize_context(case.context)
        rubric_tok = _rubric_tokens_typed(case.rubric)
        rec = _record_metadata(i, case.generation, context_passages, rubric_tok)
        rec["case_id"] = case.case_id
        rec["question"] = case.question
        rec["generation"] = case.generation
        rec["context"] = context_passages
        rec["rubric"] = [r.statement for r in case.rubric]

        has_ctx = bool(context_passages)
        has_rubric = len(case.rubric) > 0
        flags = _node_eligibility(
            has_generation=_is_nonempty_text(case.generation),
            has_context=has_ctx,
            has_rubric=has_rubric,
        )
        rec["has_context"] = has_ctx
        rec["has_rubric"] = has_rubric
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

    return _build_metadata(
        per_record,
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
    ``generation`` field. The scanner computes per-record token counts (generation,
    context, rubric), actual chunk count (via chunk_text), and estimated claim count.

    Args:
        path: File path to scan.
        show_progress: If ``True``, renders a Rich progress bar during scanning.
        progress_callback: Optional callback invoked as ``callback(current, total)``
            during record scanning. Ignored when show_progress is False.

    Returns:
        InputMetadata with token breakdowns, chunk counts, and node eligibility
        data for every scanned record.

    Raises:
        InputParseError: If any structured record is missing ``generation``.
        json.JSONDecodeError: If a JSON/JSONL payload cannot be decoded.
        OSError / UnicodeDecodeError: If the file cannot be read.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                records.append(json.loads(line))
    elif path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    elif path.suffix == ".csv":
        with path.open(newline="") as f:
            records = list(csv.DictReader(f))
    else:
        return scan_text(path.read_text())

    total = len(records)
    per_record = []
    eligible_record_count = {node: 0 for node in _NODE_ORDER}
    eligible_chunk_count = {node: 0 for node in _NODE_ORDER}
    eligible_claim_count = {node: 0 for node in _NODE_ORDER}

    def _append_record(i: int, raw: dict[str, Any]) -> None:
        generation = raw.get("generation")
        if generation is None:
            raise InputParseError(
                f"Record at index {i} is missing required 'generation' field.",
                record_index=i,
            )
        generation = str(generation)
        context_passages = _normalize_context(
            raw.get("context", raw.get("contexts", raw.get("documents")))
        )
        rubric_tok = _rubric_tokens_raw(raw.get("rubric"))
        meta = _record_metadata(i, generation, context_passages, rubric_tok)
        if "case_id" in raw:
            meta["case_id"] = raw["case_id"]
        meta["question"] = raw.get("question")
        meta["generation"] = generation
        meta["context"] = context_passages
        rubric_raw = raw.get("rubric")
        meta["rubric"] = rubric_raw if isinstance(rubric_raw, list) else None

        has_ctx = bool(context_passages)
        has_rubric = _has_rubric(raw.get("rubric", raw.get("rubric")))
        flags = _node_eligibility(
            has_generation=_is_nonempty_text(generation),
            has_context=has_ctx,
            has_rubric=has_rubric,
        )
        meta["has_context"] = has_ctx
        meta["has_rubric"] = has_rubric
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
        with _make_rich_progress() as progress:
            task = progress.add_task("Scanning records", total=total)
            for i, raw in enumerate(records):
                _append_record(i, raw)
                progress.advance(task)
    else:
        for i, raw in enumerate(records):
            if show_progress and progress_callback is not None:
                progress_callback(i + 1, total)
            _append_record(i, raw)

    return _build_metadata(
        per_record,
        eligible_record_count=eligible_record_count,
        eligible_chunk_count=eligible_chunk_count,
        eligible_claim_count=eligible_claim_count,
    )


# ── Debug / manual test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from pprint import pprint

    sample_path = Path(__file__).parents[3] / "sample.json"
    if not sample_path.exists():
        print(f"sample.json not found at {sample_path}", file=sys.stderr)
        sys.exit(1)

    m = scan_file(sample_path, show_progress=True)
    print("m: ", m)
    print("\n\n")
    pprint(m.record_meta)
    # print(f"\n{'─' * 60}")
    # print(f"  Records scanned  : {m.record_count}")
    # print(
    #     f"  Total tokens     : {m.total_tokens:>8}"
    #     f"  (gen={m.generation_tokens}  ctx={m.context_tokens}  rubric={m.rubric_tokens})"
    # )
    # print(f"  Total chars      : {m.total_chars}")
    # print(f"  Context chunks   : {m.context_chunk_count}")
    # print(f"  Generation chunks: {m.generation_chunk_count}")
    # print(f"  Estimated claims : {m.estimated_claim_count}")
    # print(f"{'─' * 60}")

    # # Node eligibility summary
    # print("\nNode eligibility (eligible records out of total):")
    # for node, count in sorted(m.eligible_record_count.items()):
    #     print(f"  {node:<12} {count:>3} / {m.record_count}")

    # # Per-record breakdown
    # print(f"\n{'─' * 60}")
    # print(
    #     f"  {'case_id':<40}"
    #     f" {'gen_tok':>7} {'ctx_tok':>7} {'rub_tok':>7}"
    #     f" {'ctx_chk':>7} {'gen_chk':>7} {'claims':>6}"
    # )
    # print(f"{'─' * 60}")
    # for r in m.per_record:
    #     label = str(r.get("case_id", r.get("record_index", "")))
    #     print(
    #         f"  {label:<40}"
    #         f"  {r['generation_tokens']:>5}"
    #         f"  {r['context_tokens']:>5}"
    #         f"  {r['rubric_tokens']:>5}"
    #         f"  {r['context_chunks']:>5}"
    #         f"  {r['generation_chunks']:>5}"
    #         f"  {r['estimated_claims']:>5}"
    #     )
    # print(f"{'─' * 60}")

    # assert m.total_tokens == m.generation_tokens + m.context_tokens + m.rubric_tokens, (
    #     "token sum mismatch"
    # )
    # print("\ntoken sum check: OK")
