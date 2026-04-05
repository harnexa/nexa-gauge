import json

import pytest
from lumiseval_core.errors import InputParseError
from lumiseval_core.types import EvalCase, GevalConfig, GevalMetricSpec
from lumiseval_core.utils import _count_tokens
from lumiseval_ingest.scanner import Scanner, scan_cases, scan_file


def _geval_metric() -> GevalMetricSpec:
    return GevalMetricSpec(
        name="mention_paris",
        record_fields=["generation"],
        criteria="Must mention Paris.",
    )


def test_scan_cases_reports_node_eligibility_counts() -> None:
    cases = [
        EvalCase(
            case_id="with-context-and-geval",
            question="What is the capital of France?",
            generation="Paris is in France.",
            context=["Paris is the capital of France."],
            geval=GevalConfig(metrics=[_geval_metric()]),
        ),
        EvalCase(
            case_id="generation-only",
            generation="Just an answer.",
        ),
        EvalCase(
            case_id="geval-only",
            generation="Another answer.",
            geval=GevalConfig(metrics=[_geval_metric()]),
        ),
    ]

    meta = scan_cases(cases)
    assert meta.record_count == 3

    # grounding / relevance: only the context-bearing case qualifies
    assert meta.cost_meta.grounding.eligible_records == 1
    assert meta.cost_meta.relevance.eligible_records == 1
    assert meta.cost_meta.claim.eligible_records == 3
    assert meta.cost_meta.claim.avg_generation_chunks > 0
    assert meta.cost_meta.claim.avg_generation_tokens > 0

    # geval: with-context-and-geval + geval-only
    assert meta.cost_meta.geval.eligible_records == 2
    assert meta.cost_meta.geval.rule_count == 2
    assert meta.cost_meta.geval.unique_rule_count == 1
    assert meta.cost_meta.geval.rule_tokens > 0
    assert meta.cost_meta.geval.unique_rule_tokens > 0
    assert meta.cost_meta.geval_steps.eligible_records == 2
    assert meta.cost_meta.geval_steps.criteria_count == 2
    assert meta.cost_meta.geval_steps.unique_criteria_count == 1
    assert meta.cost_meta.geval_steps.criteria_tokens > 0
    assert meta.cost_meta.geval_steps.unique_criteria_tokens > 0

    # redteam: all three cases have a generation
    assert meta.cost_meta.readteam.eligible_records == 3

    # Per-record token breakdown
    context_case = next(r for r in meta.records if r.case_id == "with-context-and-geval")
    ctx_meta = context_case.record_metadata
    assert context_case.geval is not None
    assert context_case.geval.criteria == {"mention_paris": "Must mention Paris."}
    assert context_case.geval.evaluation_steps == {"mention_paris": []}
    assert ctx_meta.generation_token_count > 0
    assert ctx_meta.context_token_count > 0
    assert ctx_meta.geval_token_count > 0
    assert ctx_meta.total_token_count == (
        ctx_meta.generation_token_count
        + ctx_meta.context_token_count
        + ctx_meta.geval_token_count
        + ctx_meta.question_token_count
    )

    gen_only = next(r for r in meta.records if r.case_id == "generation-only")
    gen_meta = gen_only.record_metadata
    assert gen_meta.generation_token_count > 0
    assert gen_meta.context_token_count == 0
    assert gen_meta.geval_token_count == 0

    # Aggregate token breakdown
    assert meta.generation_tokens > 0
    assert meta.context_tokens > 0
    assert meta.geval_tokens > 0
    assert (
        meta.total_tokens
        == meta.question_tokens + meta.generation_tokens + meta.context_tokens + meta.geval_tokens
    )


def test_add_raw_parity_with_add_case() -> None:
    metric_criteria = GevalMetricSpec(
        name="factuality",
        record_fields=["generation"],
        criteria="Must be factual.",
    )
    metric_steps = GevalMetricSpec(
        name="concise",
        record_fields=["generation"],
        evaluation_steps=["Check if concise.", "Check no fluff."],
    )

    case = EvalCase(
        case_id="c1",
        generation="Paris is the capital of France.",
        question="What is the capital of France?",
        reference="Paris.",
        context=["Paris is France's capital city."],
        geval=GevalConfig(metrics=[metric_criteria, metric_steps]),
    )
    raw = {
        "id": "c1",
        "response": "Paris is the capital of France.",
        "query": "What is the capital of France?",
        "ground_truth": "Paris.",
        "documents": ["Paris is France's capital city."],
        "geval": {
            "metrics": [
                {
                    "name": "factuality",
                    "record_fields": ["generation"],
                    "criteria": "Must be factual.",
                },
                {
                    "name": "concise",
                    "record_fields": ["generation"],
                    "evaluation_steps": ["Check if concise.", "Check no fluff."],
                },
            ]
        },
    }

    scanner_case = Scanner()
    scanner_case.add_case(0, case)
    meta_case = scanner_case.build()

    scanner_raw = Scanner()
    scanner_raw.add_raw(0, raw)
    meta_raw = scanner_raw.build()

    assert meta_raw.model_dump() == meta_case.model_dump()


def test_scan_file_rejects_invalid_geval_contract(tmp_path) -> None:
    file_path = tmp_path / "bad_geval.json"
    file_path.write_text(
        json.dumps(
            [
                {
                    "generation": "hello",
                    "geval": {
                        "metrics": [
                            {
                                "name": "bad",
                                "record_fields": ["generation"],
                                "criteria": "x",
                                "evaluation_steps": ["y"],
                            }
                        ]
                    },
                }
            ]
        )
    )

    with pytest.raises(InputParseError):
        scan_file(file_path)


def test_geval_steps_meta_counts_only_criteria_without_steps() -> None:
    case = EvalCase(
        case_id="mixed-geval",
        generation="Paris is in France.",
        geval=GevalConfig(
            metrics=[
                GevalMetricSpec(
                    name="criteria_only",
                    record_fields=["generation"],
                    criteria="Must mention Paris.",
                ),
                GevalMetricSpec(
                    name="steps_only",
                    record_fields=["generation"],
                    evaluation_steps=["Check mention of Paris."],
                ),
            ]
        ),
    )

    meta = scan_cases([case])
    record = meta.records[0]
    assert record.geval is not None
    assert record.geval.criteria == {
        "criteria_only": "Must mention Paris.",
        "steps_only": None,
    }
    assert record.geval.evaluation_steps == {
        "criteria_only": [],
        "steps_only": ["Check mention of Paris."],
    }

    expected_tokens = _count_tokens("Must mention Paris.") + _count_tokens("Check mention of Paris.")
    assert record.record_metadata.geval_token_count == expected_tokens
    assert meta.geval_metric_count == 2
    assert meta.unique_geval_metric_count == 2
    assert meta.geval_tokens == expected_tokens
    assert meta.unique_geval_tokens == expected_tokens
    assert meta.cost_meta.geval.eligible_records == 1
    assert meta.cost_meta.geval.rule_count == 2
    assert meta.cost_meta.geval.unique_rule_count == 2
    assert meta.cost_meta.geval.rule_tokens == expected_tokens
    assert meta.cost_meta.geval.unique_rule_tokens == expected_tokens
    assert meta.cost_meta.geval_steps.eligible_records == 1
    assert meta.cost_meta.geval_steps.criteria_count == 1
    assert meta.cost_meta.geval_steps.unique_criteria_count == 1


def test_record_geval_projection_uses_deterministic_keys_for_duplicate_names() -> None:
    case = EvalCase(
        case_id="duplicate-names",
        generation="Paris is in France.",
        geval=GevalConfig(
            metrics=[
                GevalMetricSpec(
                    name="factuality",
                    record_fields=["generation"],
                    criteria="Must mention Paris.",
                ),
                GevalMetricSpec(
                    name="factuality",
                    record_fields=["generation"],
                    evaluation_steps=["Check if Paris is present."],
                ),
            ]
        ),
    )

    meta = scan_cases([case])
    record = meta.records[0]
    assert record.geval is not None
    assert set(record.geval.criteria.keys()) == {"factuality#1", "factuality#2"}
    assert record.geval.criteria["factuality#1"] == "Must mention Paris."
    assert record.geval.criteria["factuality#2"] is None
    assert record.geval.evaluation_steps["factuality#1"] == []
    assert record.geval.evaluation_steps["factuality#2"] == ["Check if Paris is present."]
