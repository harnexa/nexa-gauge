"""
Metadata Scanner — scans input cases and returns token/chunk/claim counts.

Tokens are counted with tiktoken (cl100k_base) and generation chunks are
produced by the actual chunk_text() function, so counts are precise.

Token breakdown per record:
  question_tokens    — tokens in the question field
  generation_tokens  — tokens in the model's generated response
  context_tokens     — tokens across all context passages
  rubric_tokens      — tokens in rubric rule statements
  total_tokens       — sum of all four above
"""

import csv
import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import tiktoken
from lumiseval_core.constants import (
    CLAIMS_PER_CHUNK,
    GENERATION_CHUNK_SIZE_TOKENS,
    TIKTOKEN_ENCODING,
)
from lumiseval_core.errors import InputParseError
from lumiseval_core.pipeline import CONTEXT_REQUIRED_NODES as _CONTEXT_NODES
from lumiseval_core.pipeline import REFERENCE_REQUIRED_NODES as _REFERENCE_NODES
from lumiseval_core.pipeline import NODE_ORDER as _NODE_ORDER
from lumiseval_core.pipeline import RUBRIC_REQUIRED_NODES as _RUBRIC_NODES
from lumiseval_core.types import (
    CostMetadata,
    EvalCase,
    ReferenceCostMeta,
    GorundingCostMeta,
    InputMetadata,
    Record,
    RecordMeta,
    RedTeamCostMeta,
    RelevanceCostMeta,
    RubricCostMeta,
)
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from lumiseval_ingest.chunker import chunk_text

_ENCODING = tiktoken.get_encoding(TIKTOKEN_ENCODING)

ProgressCallback = Callable[[int, int], None]


# ── Normalisation helpers ─────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_context(raw: Any) -> list[str]:
    """Normalise any context field to a clean list[str]."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item) for item in raw if item and str(item).strip()]
    return [str(raw)] if str(raw).strip() else []


def _normalize_rubric(raw: Any) -> list[str]:
    """Normalise any rubric field to a flat list of statement strings."""
    if not raw or not isinstance(raw, list):
        return []
    result = []
    for rule in raw:
        if isinstance(rule, str) and rule.strip():
            result.append(rule)
        elif isinstance(rule, dict) and rule.get("statement", "").strip():
            result.append(rule["statement"])
    return result


# ── Node eligibility ──────────────────────────────────────────────────────────


def _node_eligibility(
    *,
    has_generation: bool,
    has_context: bool,
    has_rubric: bool,
    has_reference: bool,
) -> dict[str, bool]:
    flags = {node: has_generation for node in _NODE_ORDER}
    for node in _CONTEXT_NODES:
        flags[node] = has_generation and has_context
    for node in _RUBRIC_NODES:
        flags[node] = has_generation and has_rubric
    for node in _REFERENCE_NODES:
        flags[node] = has_generation and has_reference
    return flags


# ── Record builder ────────────────────────────────────────────────────────────


def _build_record(
    *,
    idx: int,
    case_id: str,
    question: Optional[str],
    generation: Optional[str],
    context: list[str],
    rubric: list[str],
    reference: str,
) -> Record:
    """Compute token counts, chunk the generation, and return a typed Record."""
    question_token_count = _count_tokens(question) if question else 0
    generation_token_count = _count_tokens(generation) if generation else 0
    context_token_count = sum(_count_tokens(p) for p in context)
    rubric_token_count = sum(_count_tokens(r) for r in rubric)

    generation_chunk_objects = (
        chunk_text(generation, GENERATION_CHUNK_SIZE_TOKENS) if generation else []
    )
    generation_chunk_count = len(generation_chunk_objects)

    has_context = bool(context)
    has_rubric = bool(rubric)
    has_reference = bool(reference)
    eligible_nodes = [
        node
        for node, ok in _node_eligibility(
            has_generation=_is_nonempty_text(generation or ""),
            has_context=has_context,
            has_rubric=has_rubric,
            has_reference=has_reference,
        ).items()
        if ok
    ]

    return Record(
        record_index=idx,
        case_id=case_id,
        question=question,
        generation=generation,
        context=context,
        generation_chunks=[c.text for c in generation_chunk_objects],
        rubric=rubric,
        record_metadata=RecordMeta(
            context_token_count=context_token_count,
            generation_chunk_count=generation_chunk_count,
            generation_token_count=generation_token_count,
            rubric_token_count=rubric_token_count,
            total_token_count=question_token_count
            + generation_token_count
            + context_token_count
            + rubric_token_count,
            estimated_claim_count=generation_chunk_count * CLAIMS_PER_CHUNK,
            has_context=has_context,
            has_rubric=has_rubric,
            has_reference=has_reference,
            eligible_nodes=eligible_nodes,
        ),
    )


# ── Progress bar ──────────────────────────────────────────────────────────────


def _make_rich_progress() -> Progress:
    return Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


# ── Scanner ───────────────────────────────────────────────────────────────────


class Scanner:
    """Accumulates per-record metadata and node eligibility across a scan run.

    Usage::

        scanner = Scanner()
        for i, case in enumerate(cases):
            scanner.add_case(i, case)
        metadata = scanner.build()
    """

    def __init__(self) -> None:
        self._records: list[Record] = []
        self._eligible_record_count: dict[str, int] = {node: 0 for node in _NODE_ORDER}
        self._eligible_chunk_count: dict[str, int] = {node: 0 for node in _NODE_ORDER}
        self._eligible_claim_count: dict[str, int] = {node: 0 for node in _NODE_ORDER}

    # ── Per-record ingestion ──────────────────────────────────────────────────

    def add_case(self, idx: int, case: EvalCase) -> None:
        """Ingest one typed EvalCase."""
        record = _build_record(
            idx=idx,
            case_id=case.case_id,
            question=case.question,
            generation=case.generation,
            context=_normalize_context(case.context),
            rubric=[r.statement for r in case.rubric],
            reference=case.reference,
        )
        self._add(record)

    def add_raw(self, idx: int, raw: dict[str, Any]) -> None:
        """Ingest one raw dict (from JSON/JSONL/CSV).

        Raises:
            InputParseError: If ``generation`` field is missing.
        """
        generation = raw.get("generation")
        if generation is None:
            raise InputParseError(
                f"Record at index {idx} is missing required 'generation' field.",
                record_index=idx,
            )
        record = _build_record(
            idx=idx,
            case_id=raw.get("case_id", f"record-{idx}"),
            question=raw.get("question"),
            generation=str(generation),
            context=_normalize_context(
                raw.get("context", raw.get("contexts", raw.get("documents")))
            ),
            rubric=_normalize_rubric(raw.get("rubric_rules", raw.get("rubric"))),
            reference=raw.get("reference", None),
        )
        self._add(record)

    # ── Internal accumulation ─────────────────────────────────────────────────

    def _add(self, record: Record) -> None:
        meta = record.record_metadata
        for node in meta.eligible_nodes:
            self._eligible_record_count[node] += 1
            self._eligible_chunk_count[node] += meta.generation_chunk_count
            self._eligible_claim_count[node] += meta.estimated_claim_count
        self._records.append(record)

    # ── Output ────────────────────────────────────────────────────────────────

    def build(self) -> InputMetadata:
        """Aggregate all accumulated records into an InputMetadata object.


        cost_meta:
        {
            'grounding': {
               'eligible_records': 3,
               'avg_claims_per_record': 2.3333,
               'avg_context_tokens': 66.6667,
               'avg_claim_tokens': 15,
               'avg_output_token': 5
            },
            'relevance': {
                'eligible_records': 3,
                'avg_claims_per_record': 2.3333,
                'avg_question_tokens': 13.1667,
                'avg_claim_tokens': 15,
                'avg_output_token': 10
            },
            'rubric': {
                'eligible_records': 6,
                'rule_count': 28,
                'unique_rule_count': 24,
                'rule_tokens': 331.0,
                'unique_rule_tokens': 303.0,
                'avg_input_tokens': 400,
                'avg_output_tokens': 60
            },
            'readteam': {
                'eligible_records': 12,
                'avg_input_tokens': 350,
                'avg_output_tokens': 50
            }
        }


        {
            'case_id': 'eiffel-tower-basic',
            'record_index': 0,
            'question': 'What is the Eiffel Tower and where is it located?',
            'context': ['The Eiffel Tower (/ˈaɪfəl/ EYE-fəl; French: Tour Eiffel) is a '
                        'wrought-iron lattice tower on the Champ de Mars in Paris, '
                        'France. It is named after the engineer Gustave Eiffel, whose '
                        'company designed and built the tower from 1887 to 1889 as the '
                        "centerpiece of the 1889 World's Fair."],
            'generation': 'The Eiffel Tower is a wrought-iron lattice tower located in '
                        'Paris, France. It was constructed between 1887 and 1889 and '
                        "served as the entrance arch to the 1889 World's Fair. Standing "
                        'at 330 metres, it is one of the most recognisable structures '
                        'in the world.',
            'generation_chunks': ['The Eiffel Tower is a wrought-iron lattice tower '
                                'located in Paris, France. It was constructed between '
                                '1887 and 1889 and served as the entrance arch to the '
                                "1889 World's Fair. Standing at 330 metres, it is one "
                                'of the most recognisable structures in the world.'],
            'rubric': [],
            'record_metadata': {'context_token_count': 84,
                                'generation_chunk_count': 1,
                                'generation_token_count': 63,
                                'rubric_token_count': 0,
                                'total_token_count': 160,
                                'estimated_claim_count': 1,
                                'has_context': True,
                                'has_rubric': False,
                                'eligible_nodes': ['scan',
                                                    'estimate',
                                                    'approve',
                                                    'chunk',
                                                    'claims',
                                                    'dedupe',
                                                    'relevance',
                                                    'grounding',
                                                    'redteam',
                                                    'eval']}
        }

        """
        metas = [r.record_metadata for r in self._records]

        # ── Token aggregates ──────────────────────────────────────────────────
        total_tokens = sum(m.total_token_count for m in metas)
        generation_tokens = sum(m.generation_token_count for m in metas)
        context_tokens = sum(m.context_token_count for m in metas)
        rubric_tokens = sum(m.rubric_token_count for m in metas)
        generation_chunk_count = sum(m.generation_chunk_count for m in metas)

        # ── Unique rubric deduplication ───────────────────────────────────────
        all_rubric_stmts = [s for r in self._records for s in (r.rubric or [])]
        rubric_rule_count = len(all_rubric_stmts)
        unique_stmts = list(dict.fromkeys(all_rubric_stmts))
        unique_rubric_rule_count = len(unique_stmts)
        unique_rubric_tokens = sum(_count_tokens(s) for s in unique_stmts)

        # ── Per-node eligible counts ──────────────────────────────────────────
        grounding_eligible = self._eligible_record_count["grounding"]
        relevance_eligible = self._eligible_record_count["relevance"]
        rubric_eligible = self._eligible_record_count["rubric"]
        redteam_eligible = self._eligible_record_count["redteam"]
        reference_eligible = self._eligible_record_count["reference"]

        # ── Grounding cost meta ───────────────────────────────────────────────
        grounding_claims = self._eligible_claim_count["grounding"]
        avg_grounding_claims = grounding_claims / max(1, grounding_eligible)
        context_records = [r for r in self._records if r.record_metadata.has_context]
        avg_context_tokens = sum(
            r.record_metadata.context_token_count for r in context_records
        ) / max(1, len(context_records))

        # ── Relevance cost meta ───────────────────────────────────────────────
        relevance_claims = self._eligible_claim_count["relevance"]
        avg_relevance_claims = relevance_claims / max(1, relevance_eligible)
        q_records = [r for r in self._records if r.question]
        avg_question_tokens = sum(_count_tokens(r.question) for r in q_records) / max(
            1, len(q_records)
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        return InputMetadata(
            record_count=len(self._records),
            total_tokens=total_tokens,
            generation_tokens=generation_tokens,
            context_tokens=context_tokens,
            rubric_tokens=rubric_tokens,
            unique_rubric_tokens=unique_rubric_tokens,
            rubric_rule_count=rubric_rule_count,
            unique_rubric_rule_count=unique_rubric_rule_count,
            generation_chunk_count=generation_chunk_count,
            cost_meta=CostMetadata(
                grounding=GorundingCostMeta(
                    eligible_records=grounding_eligible,
                    avg_claims_per_record=round(avg_grounding_claims, 4),
                    avg_context_tokens=round(avg_context_tokens, 4),
                ),
                relevance=RelevanceCostMeta(
                    eligible_records=relevance_eligible,
                    avg_claims_per_record=round(avg_relevance_claims, 4),
                    avg_question_tokens=round(avg_question_tokens, 4),
                ),
                rubric=RubricCostMeta(
                    eligible_records=rubric_eligible,
                    rule_count=rubric_rule_count,
                    unique_rule_count=unique_rubric_rule_count,
                    rule_tokens=round(float(rubric_tokens), 4),
                    unique_rule_tokens=round(float(unique_rubric_tokens), 4),
                ),
                readteam=RedTeamCostMeta(
                    eligible_records=redteam_eligible,
                ),
                reference=ReferenceCostMeta(
                    eligible_records=reference_eligible,
                ),
            ),
            records=self._records,
        )


# ── Public API ────────────────────────────────────────────────────────────────


def scan_text(
    generation: str,
    reference: Optional[str] = None,
    **_ignored: Any,
) -> InputMetadata:
    """Scan a single generation string and return metadata."""
    scanner = Scanner()
    scanner.add_case(
        0,
        EvalCase(case_id="record-0", generation=generation, reference=reference),
    )
    return scanner.build()


def scan_cases(
    cases: Sequence[EvalCase],
    show_progress: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> InputMetadata:
    """Scan a sequence of typed EvalCase rows and return aggregate metadata."""
    scanner = Scanner()

    if show_progress and progress_callback is None:
        with _make_rich_progress() as progress:
            task = progress.add_task("Scanning records", total=len(cases))
            for i, case in enumerate(cases):
                scanner.add_case(i, case)
                progress.advance(task)
    else:
        for i, case in enumerate(cases):
            if show_progress and progress_callback is not None:
                progress_callback(i + 1, len(cases))
            scanner.add_case(i, case)

    return scanner.build()


def scan_file(
    path: str | Path,
    show_progress: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> InputMetadata:
    """Scan a local file and return aggregate metadata.

    Supported formats: ``.json``, ``.jsonl``, ``.csv``.
    Any other suffix is treated as a raw generation string (single-record scan).

    Raises:
        InputParseError: If any record is missing the ``generation`` field.
    """
    path = Path(path)

    if path.suffix == ".jsonl":
        with path.open() as f:
            raw_records = [json.loads(line) for line in f]
    elif path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        raw_records = data if isinstance(data, list) else [data]
    elif path.suffix == ".csv":
        with path.open(newline="") as f:
            raw_records = list(csv.DictReader(f))
    else:
        return scan_text(path.read_text())

    scanner = Scanner()

    if show_progress and progress_callback is None:
        with _make_rich_progress() as progress:
            task = progress.add_task("Scanning records", total=len(raw_records))
            for i, raw in enumerate(raw_records):
                scanner.add_raw(i, raw)
                progress.advance(task)
    else:
        for i, raw in enumerate(raw_records):
            if show_progress and progress_callback is not None:
                progress_callback(i + 1, len(raw_records))
            scanner.add_raw(i, raw)

    return scanner.build()


# ── Debug / manual test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sample_path = Path(__file__).parents[3] / "sample.json"
    if not sample_path.exists():
        print(f"sample.json not found at {sample_path}", file=sys.stderr)
        sys.exit(1)

    m = scan_file(sample_path, show_progress=True)
    from pprint import pprint

    pprint(m.cost_meta.model_dump(), sort_dicts=False)
    print("\n\n")
    pprint(m.records[0].model_dump(), sort_dicts=False)
