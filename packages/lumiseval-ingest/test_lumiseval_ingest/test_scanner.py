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

    # ── Node eligibility via cost_meta ────────────────────────────────────────
    # grounding / relevance: only the with-context case qualifies
    assert meta.cost_meta.grounding.eligible_records == 1
    assert meta.cost_meta.relevance.eligible_records == 1

    # rubric: with-context-and-rubric + rubric-only
    assert meta.cost_meta.rubric.eligible_records == 2

    # redteam: all three cases have a generation
    assert meta.cost_meta.readteam.eligible_records == 3

    # ── Cost meta averages ────────────────────────────────────────────────────
    assert meta.cost_meta.grounding.avg_claims_per_record > 0
    assert meta.cost_meta.grounding.avg_context_tokens > 0
    assert meta.cost_meta.relevance.avg_claims_per_record > 0

    # ── Rubric deduplication ──────────────────────────────────────────────────
    # Two records have 1 rule each, but it's the same statement both times
    assert meta.cost_meta.rubric.rule_count == 2
    assert meta.cost_meta.rubric.unique_rule_count == 1
    assert meta.cost_meta.rubric.rule_tokens > 0
    assert meta.cost_meta.rubric.unique_rule_tokens > 0
    assert meta.cost_meta.rubric.unique_rule_tokens <= meta.cost_meta.rubric.rule_tokens

    # ── Per-record token breakdown ────────────────────────────────────────────
    context_case = next(r for r in meta.records if r.case_id == "with-context-and-rubric")
    ctx_meta = context_case.record_metadata
    assert ctx_meta.generation_token_count > 0
    assert ctx_meta.context_token_count > 0
    assert ctx_meta.rubric_token_count > 0
    assert ctx_meta.total_token_count == (
        ctx_meta.generation_token_count + ctx_meta.context_token_count + ctx_meta.rubric_token_count
    )

    gen_only = next(r for r in meta.records if r.case_id == "generation-only")
    gen_meta = gen_only.record_metadata
    assert gen_meta.generation_token_count > 0
    assert gen_meta.context_token_count == 0
    assert gen_meta.rubric_token_count == 0

    # ── Aggregate token breakdown on InputMetadata ────────────────────────────
    assert meta.generation_tokens > 0
    assert meta.context_tokens > 0  # from the with-context case
    assert meta.rubric_tokens > 0  # from the two rubric cases
    assert meta.total_tokens == meta.generation_tokens + meta.context_tokens + meta.rubric_tokens
