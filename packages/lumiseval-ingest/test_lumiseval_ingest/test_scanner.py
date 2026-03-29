from lumiseval_core.types import EvalCase, RubricRule
from lumiseval_ingest.scanner import scan_cases


def _rule() -> RubricRule:
    return RubricRule(
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
            rubric_rules=[_rule()],
        ),
        EvalCase(
            case_id="generation-only",
            generation="Just an answer.",
        ),
        EvalCase(
            case_id="rubric-only",
            generation="Another answer.",
            rubric_rules=[_rule()],
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
    assert meta.eligible_chunk_count["claims"] == context_case["estimated_chunks"]
    assert meta.eligible_claim_count["dedupe"] == context_case["estimated_claims"]
