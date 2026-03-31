from lumiseval_core.types import EvalCase, Rubric
from lumiseval_ingest.scanner import scan_cases


def _rule() -> Rubric:
    return Rubric(
        id="R-1",
        statement="Must mention Paris.",
        pass_condition="Response includes Paris.",
    )


def test_scan_cases_reports_node_eligibility_counts() -> None:
    cases = [
        EvalCase(
            case_id="with-context-and-rubric",
            generation="Paris is in France.",
            context=["Paris is the capital of France."],
            rubric=[_rule()],
        ),
        EvalCase(
            case_id="generation-only",
            generation="Just an answer.",
        ),
        EvalCase(
            case_id="rubric-only",
            generation="Another answer.",
            rubric=[_rule()],
        ),
    ]

    meta = scan_cases(cases)

    assert meta.record_count == 3
    assert meta.eligible_record_count["scan"] == 3
    assert meta.eligible_record_count["estimate"] == 3
    assert meta.eligible_record_count["approve"] == 3
    assert meta.eligible_record_count["redteam"] == 3
    assert meta.eligible_record_count["eval"] == 3

    assert meta.eligible_record_count["chunk"] == 1
    assert meta.eligible_record_count["claims"] == 1
    assert meta.eligible_record_count["dedupe"] == 1
    assert meta.eligible_record_count["relevance"] == 1
    assert meta.eligible_record_count["grounding"] == 1

    assert meta.eligible_record_count["rubric"] == 2

    context_case = next(r for r in meta.per_record if r["case_id"] == "with-context-and-rubric")
    # claims node processes generation chunks; chunk node processes context chunks
    assert meta.eligible_chunk_count["claims"] == context_case["generation_chunks"]
    assert meta.eligible_chunk_count["chunk"] == context_case["context_chunks"]
    assert meta.eligible_claim_count["dedupe"] == context_case["estimated_claims"]

    # ── Per-field token breakdown ─────────────────────────────────────────────
    assert context_case["generation_tokens"] > 0
    assert context_case["context_tokens"] > 0
    assert context_case["rubric_tokens"] > 0
    assert context_case["tokens"] == (
        context_case["generation_tokens"]
        + context_case["context_tokens"]
        + context_case["rubric_tokens"]
    )

    gen_only = next(r for r in meta.per_record if r["case_id"] == "generation-only")
    assert gen_only["generation_tokens"] > 0
    assert gen_only["context_tokens"] == 0
    assert gen_only["rubric_tokens"] == 0

    # Aggregate breakdown on InputMetadata
    assert meta.generation_tokens > 0
    assert meta.context_tokens > 0  # from the with-context case
    assert meta.rubric_tokens > 0  # from the two rubric cases
    assert meta.total_tokens == meta.generation_tokens + meta.context_tokens + meta.rubric_tokens
