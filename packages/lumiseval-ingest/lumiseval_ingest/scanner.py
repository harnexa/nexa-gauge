"""
Metadata Scanner — scans input cases and returns token/chunk/claim counts.

Tokens are counted with tiktoken (cl100k_base) and generation chunks are
produced by the actual chunk_text() function, so counts are precise.

Token breakdown per record:
  question_tokens    — tokens in the question field
  generation_tokens  — tokens in the model's generated response
  context_tokens     — tokens across all context passages
  geval_tokens       — tokens in GEval instructions (criteria/evaluation_steps)
  total_tokens       — sum of all four above
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from lumiseval_core.constants import (
    CLAIMS_PER_CHUNK,
    GENERATION_CHUNK_SIZE_TOKENS,
)
from lumiseval_core.pipeline import NODES_BY_NAME as _NODES_BY_NAME
from lumiseval_core.types import (
    ClaimCostMeta,
    CostMetadata,
    EvalCase,
    GevalCostMeta,
    GevalMetricSpec,
    GevalStepsCostMeta,
    GorundingCostMeta,
    InputMetadata,
    Record,
    RecordGeval,
    RecordMeta,
    RedTeamCostMeta,
    ReferenceCostMeta,
    RelevanceCostMeta,
)
from lumiseval_core.utils import _count_tokens
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from lumiseval_ingest.adapters.registry import create_dataset_adapter
from lumiseval_ingest.canonical import canonical_case_from_raw, normalize_context
from lumiseval_ingest.chunker import chunk_text

ProgressCallback = Callable[[int, int], None]


# ── Normalisation helpers ─────────────────────────────────────────────────────


def _metric_key_sequence(metrics: list[GevalMetricSpec]) -> list[str]:
    """Return deterministic per-record metric keys.

    Rule:
      - use ``metric.name`` when unique in the record
      - use ``metric.name#N`` (1-based) when duplicates exist
    """
    name_counts: dict[str, int] = {}
    for metric in metrics:
        name_counts[metric.name] = name_counts.get(metric.name, 0) + 1

    name_seen: dict[str, int] = {}
    keys: list[str] = []
    for metric in metrics:
        name = metric.name
        if name_counts[name] == 1:
            keys.append(name)
            continue
        name_seen[name] = name_seen.get(name, 0) + 1
        keys.append(f"{name}#{name_seen[name]}")
    return keys


def _metric_instruction_text(metric: GevalMetricSpec) -> str:
    """Return token-bearing GEval instruction text for one metric."""
    if metric.criteria and metric.criteria.strip():
        return metric.criteria.strip()
    steps = [step.strip() for step in (metric.evaluation_steps or []) if step and step.strip()]
    return "\n".join(steps)


def _metric_signature(metric: GevalMetricSpec) -> str:
    """Return deterministic signature for GEval metric uniqueness."""
    payload = {
        "name": metric.name,
        "record_fields": list(metric.record_fields),
        "criteria": metric.criteria,
        "evaluation_steps": list(metric.evaluation_steps or []),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _project_record_geval(metrics: list[GevalMetricSpec]) -> tuple[
    RecordGeval,
    list[str],
    list[str],
    list[str],
    int,
    dict[str, int],
]:
    """Build record-level GEval projection and related scanner aggregates."""
    keys = _metric_key_sequence(metrics)
    criteria_map: dict[str, str | None] = {}
    steps_map: dict[str, list[str]] = {}

    instruction_units: list[str] = []
    step_criteria: list[str] = []
    metric_signatures: list[str] = []
    metric_tokens_total = 0
    signature_to_tokens: dict[str, int] = {}

    for key, metric in zip(keys, metrics):
        criteria = metric.criteria.strip() if metric.criteria else None
        steps = [step.strip() for step in (metric.evaluation_steps or []) if step and step.strip()]
        criteria_map[key] = criteria
        steps_map[key] = steps

        instruction_text = _metric_instruction_text(metric)
        if instruction_text:
            instruction_units.append(instruction_text)
            metric_tokens_total += _count_tokens(instruction_text)

        if criteria and not steps:
            step_criteria.append(criteria)

        signature = _metric_signature(metric)
        metric_signatures.append(signature)
        signature_to_tokens.setdefault(signature, _count_tokens(instruction_text) if instruction_text else 0)

    return (
        RecordGeval(criteria=criteria_map, evaluation_steps=steps_map),
        instruction_units,
        step_criteria,
        metric_signatures,
        metric_tokens_total,
        signature_to_tokens,
    )




def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# ── Node eligibility ──────────────────────────────────────────────────────────


def _node_eligibility(
    *,
    has_generation: bool,
    has_context: bool,
    has_question: bool,
    has_geval: bool,
    has_reference: bool,
) -> dict[str, bool]:
    flags = {}
    for name, spec in _NODES_BY_NAME.items():
        if spec.requires_context:
            flags[name] = has_generation and has_context
        elif spec.requires_question:
            flags[name] = has_generation and has_question
        elif spec.requires_geval:
            flags[name] = has_generation and has_geval
        elif spec.requires_reference:
            flags[name] = has_generation and has_reference
        else:
            flags[name] = has_generation
    return flags


# ── Record builder ────────────────────────────────────────────────────────────


def _build_record(
    *,
    idx: int,
    case_id: str,
    question: Optional[str],
    generation: Optional[str],
    context: list[str],
    geval: Optional[RecordGeval],
    geval_instruction_units: list[str],
    has_geval: bool,
    reference: Optional[str],
) -> Record:
    """Compute token counts, chunk the generation, and return a typed Record."""
    question_token_count = _count_tokens(question) if question else 0
    generation_token_count = _count_tokens(generation) if generation else 0
    context_token_count = sum(_count_tokens(p) for p in context)
    geval_token_count = sum(_count_tokens(text) for text in geval_instruction_units)

    generation_chunk_objects = (
        chunk_text(generation, GENERATION_CHUNK_SIZE_TOKENS) if generation else []
    )
    generation_chunk_count = len(generation_chunk_objects)

    has_context = bool(context)
    has_reference = _is_nonempty_text(reference or "")
    has_question = _is_nonempty_text(question or "")
    eligible_nodes = [
        node
        for node, ok in _node_eligibility(
            has_generation=_is_nonempty_text(generation or ""),
            has_context=has_context,
            has_question=has_question,
            has_geval=has_geval,
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
        geval=geval,
        record_metadata=RecordMeta(
            context_token_count=context_token_count,
            generation_chunk_count=generation_chunk_count,
            generation_token_count=generation_token_count,
            geval_token_count=geval_token_count,
            question_token_count=question_token_count,
            total_token_count=question_token_count
            + generation_token_count
            + context_token_count
            + geval_token_count,
            estimated_claim_count=generation_chunk_count * CLAIMS_PER_CHUNK,
            has_question=has_question,
            has_context=has_context,
            has_geval=has_geval,
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
        self._eligible_record_count: dict[str, int] = {node: 0 for node in _NODES_BY_NAME}
        self._eligible_chunk_count: dict[str, int] = {node: 0 for node in _NODES_BY_NAME}
        self._eligible_claim_count: dict[str, int] = {node: 0 for node in _NODES_BY_NAME}
        self._geval_metric_count = 0
        self._geval_metric_signatures: list[str] = []
        self._geval_signature_to_tokens: dict[str, int] = {}
        self._geval_instruction_units: list[str] = []
        self._geval_rule_tokens_total = 0
        self._geval_step_criteria: list[str] = []
        self._geval_steps_record_count = 0

    # ── Per-record ingestion ──────────────────────────────────────────────────

    def add_case(self, idx: int, case: EvalCase) -> None:
        """Ingest one typed EvalCase."""
        has_geval = case.geval is not None and len(case.geval.metrics) > 0
        record_geval: Optional[RecordGeval] = None
        instruction_units: list[str] = []
        step_criteria: list[str] = []
        metric_signatures: list[str] = []
        metric_tokens_total = 0
        signature_to_tokens: dict[str, int] = {}
        if case.geval and case.geval.metrics:
            (
                record_geval,
                instruction_units,
                step_criteria,
                metric_signatures,
                metric_tokens_total,
                signature_to_tokens,
            ) = _project_record_geval(case.geval.metrics)

        record = _build_record(
            idx=idx,
            case_id=case.case_id,
            question=case.question,
            generation=case.generation,
            context=normalize_context(case.context),
            geval=record_geval,
            geval_instruction_units=instruction_units,
            has_geval=has_geval,
            reference=case.reference,
        )
        self._add(record)
        if case.geval and case.geval.metrics:
            self._geval_metric_count += len(case.geval.metrics)
            self._geval_metric_signatures.extend(metric_signatures)
            self._geval_instruction_units.extend(instruction_units)
            self._geval_rule_tokens_total += metric_tokens_total
            for signature, token_count in signature_to_tokens.items():
                self._geval_signature_to_tokens.setdefault(signature, token_count)
            self._geval_step_criteria.extend(step_criteria)
            if step_criteria:
                self._geval_steps_record_count += 1

    def add_raw(self, idx: int, raw: dict[str, Any]) -> None:
        """Ingest one raw dict (from JSON/JSONL/CSV) via strict canonicalization.

        Raises:
            InputParseError: If canonicalization/validation fails.
        """
        case = canonical_case_from_raw(raw, idx)
        self.add_case(idx, case)

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
        """Aggregate all scanned rows into InputMetadata and per-node cost metadata."""
        metas = [r.record_metadata for r in self._records]

        # ── Token aggregates ──────────────────────────────────────────────────
        total_tokens = sum(m.total_token_count for m in metas)
        question_tokens = sum(m.question_token_count for m in metas)
        generation_tokens = sum(m.generation_token_count for m in metas)
        context_tokens = sum(m.context_token_count for m in metas)
        geval_tokens = sum(m.geval_token_count for m in metas)
        generation_chunk_count = sum(m.generation_chunk_count for m in metas)

        # ── Unique GEval instruction/metric deduplication ─────────────────────
        geval_metric_count = self._geval_metric_count
        unique_instruction_units = list(dict.fromkeys(self._geval_instruction_units))
        unique_geval_metric_count = len(set(self._geval_metric_signatures))
        unique_geval_tokens = sum(_count_tokens(text) for text in unique_instruction_units)

        # ── Per-node eligible counts ──────────────────────────────────────────
        grounding_eligible = self._eligible_record_count["grounding"]
        relevance_eligible = self._eligible_record_count["relevance"]
        geval_eligible = self._eligible_record_count["geval"]
        redteam_eligible = self._eligible_record_count["redteam"]
        reference_eligible = self._eligible_record_count["reference"]
        geval_steps_eligible = self._geval_steps_record_count

        # ── Claim extraction cost meta ────────────────────────────────────────
        claim_eligible = self._eligible_record_count["claims"]
        claim_chunks = self._eligible_chunk_count["claims"]
        avg_claim_chunks = claim_chunks / max(1, claim_eligible)
        avg_claim_generation_tokens = (
            generation_tokens // max(1, generation_chunk_count)
            if generation_chunk_count
            else GENERATION_CHUNK_SIZE_TOKENS
        )

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
        q_records = [r for r in self._records if _is_nonempty_text(r.question)]
        avg_question_tokens = sum(_count_tokens(r.question or "") for r in q_records) / max(
            1, len(q_records)
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        unique_geval_steps_criteria = list(dict.fromkeys(self._geval_step_criteria))
        geval_steps_cost_tokens = sum(_count_tokens(s) for s in self._geval_step_criteria)
        unique_geval_steps_cost_tokens = sum(_count_tokens(s) for s in unique_geval_steps_criteria)
        unique_rule_count = len(self._geval_signature_to_tokens)
        unique_rule_tokens = sum(self._geval_signature_to_tokens.values())

        # ── Assemble ──────────────────────────────────────────────────────────
        return InputMetadata(
            record_count=len(self._records),
            total_tokens=total_tokens,
            question_tokens=question_tokens,
            generation_tokens=generation_tokens,
            context_tokens=context_tokens,
            geval_tokens=geval_tokens,
            unique_geval_tokens=unique_geval_tokens,
            geval_metric_count=geval_metric_count,
            unique_geval_metric_count=unique_geval_metric_count,
            generation_chunk_count=generation_chunk_count,
            cost_meta=CostMetadata(
                claim=ClaimCostMeta(
                    eligible_records=claim_eligible,
                    avg_generation_chunks=round(avg_claim_chunks, 4),
                    avg_generation_tokens=round(float(avg_claim_generation_tokens), 4),
                ),
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
                geval_steps=GevalStepsCostMeta(
                    eligible_records=geval_steps_eligible,
                    criteria_count=len(self._geval_step_criteria),
                    unique_criteria_count=len(unique_geval_steps_criteria),
                    criteria_tokens=round(float(geval_steps_cost_tokens), 4),
                    unique_criteria_tokens=round(float(unique_geval_steps_cost_tokens), 4),
                ),
                geval=GevalCostMeta(
                    eligible_records=geval_eligible,
                    rule_count=self._geval_metric_count,
                    unique_rule_count=unique_rule_count,
                    rule_tokens=round(float(self._geval_rule_tokens_total), 4),
                    unique_rule_tokens=round(float(unique_rule_tokens), 4),
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
    """Case where only generation text is provided.

    Case:
        - Blogs:
        - Other Generation that requires verification with source
    """
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
    """Scan a local file by routing through the canonical local adapter path."""
    adapter = create_dataset_adapter(Path(path), adapter="local")
    cases = list(adapter.iter_cases())
    return scan_cases(
        cases,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )


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
