# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_scanner.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_scanner.py::test_scan_builds_inputs_with_primary_keys
# uv run pytest -s -k "scanner" packages/nexagauge-graph/test_ng_graph/test_nodes/test_scanner.py

import json

import pytest
from ng_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from ng_core.types import GevalScoringMode
from ng_graph.nodes.scanner import scan, scan_file_record


def test_scan_builds_inputs_with_primary_keys() -> None:
    record = {
        "case_id": "case-123",
        "output": "Paris is the capital of France.",
        "input": "What is the capital of France?",
        "reference": "The capital of France is Paris.",
        "context": "France has Paris as its capital city.",
    }

    result = scan(record, idx=0)

    assert result["case_id"] == "case-123"
    assert result["dataset"] == DEFAULT_DATASET_NAME
    assert result["split"] == DEFAULT_SPLIT
    assert result["reference_files"] == []

    inputs = result["inputs"]
    assert inputs.output.text == "Paris is the capital of France."
    assert inputs.input is not None and inputs.input.text == "What is the capital of France?"
    assert (
        inputs.reference is not None and inputs.reference.text == "The capital of France is Paris."
    )
    assert (
        inputs.context is not None
        and inputs.context.text == "France has Paris as its capital city."
    )
    assert inputs.geval is None

    assert inputs.has_output is True
    assert inputs.has_input is True
    assert inputs.has_reference is True
    assert inputs.has_context is True
    assert inputs.has_geval is False
    assert inputs.has_redteam is False


def test_scan_uses_alias_keys_and_normalizes_context_list() -> None:
    record = {
        "id": "alias-case",
        "response": "Berlin is the capital of Germany.",
        "query": "Capital of Germany?",
        "gold_answer": "Berlin",
        "documents": ["Germany is in Europe.", "  ", None, "Berlin is its capital."],
    }

    result = scan(record, idx=1)
    inputs = result["inputs"]

    assert result["case_id"] == "alias-case"
    assert inputs.output.text == "Berlin is the capital of Germany."
    assert inputs.input is not None and inputs.input.text == "Capital of Germany?"
    assert inputs.reference is not None and inputs.reference.text == "Berlin"
    assert inputs.context is not None
    assert inputs.context.text == "Germany is in Europe.\n\nBerlin is its capital."


def test_scan_accepts_legacy_generation_and_question_keys() -> None:
    """Backward-compat: records using the pre-rename canonical names still work."""
    record = {
        "case_id": "legacy-case",
        "generation": "Paris is the capital of France.",
        "question": "What is the capital of France?",
    }

    result = scan(record, idx=2)
    inputs = result["inputs"]

    assert inputs.output.text == "Paris is the capital of France."
    assert inputs.input is not None and inputs.input.text == "What is the capital of France?"
    assert inputs.has_output is True
    assert inputs.has_input is True


def test_scan_keeps_existing_case_defaults() -> None:
    case = {
        "case_id": "pre-set-case-id",
        "dataset": "custom-dataset",
        "split": "custom-split",
        "reference_files": ["/tmp/context.txt"],
    }
    record = {
        "id": "record-id-should-not-overwrite",
        "output": "Answer",
    }

    result = scan(record, idx=5, case=case)

    assert result["case_id"] == "pre-set-case-id"
    assert result["dataset"] == "custom-dataset"
    assert result["split"] == "custom-split"
    assert result["reference_files"] == ["/tmp/context.txt"]
    assert result["inputs"].output.text == "Answer"


def test_scan_defaults_case_id_and_flags_when_values_missing() -> None:
    result = scan({}, idx=7)
    inputs = result["inputs"]

    assert result["case_id"] == "record-7"
    assert inputs.output.text == ""
    assert inputs.input is None
    assert inputs.reference is None
    assert inputs.context is None
    assert inputs.geval is None

    assert inputs.has_output is False
    assert inputs.has_input is False
    assert inputs.has_reference is False
    assert inputs.has_context is False
    assert inputs.has_geval is False
    assert inputs.has_redteam is False


def test_scan_builds_geval_from_item_fields_only() -> None:
    record = {
        "output": "Generated answer",
        "geval": {
            "metrics": [
                {
                    "name": "correctness",
                    "item_fields": ["input", "output"],
                    "criteria": "The answer should be correct.",
                    "evaluation_steps": ["Check factual correctness.", "  ", "Verify directness."],
                },
                {
                    "name": "groundedness",
                    "item_fields": ["context", "reference"],
                    "criteria": "The answer must align with evidence.",
                    "evaluation_steps": ["Compare answer with context."],
                    "scoring_mode": "binary_yes_no",
                    "include_reasoning": False,
                },
            ]
        },
    }

    result = scan(record, idx=2)
    inputs = result["inputs"]

    assert inputs.has_geval is True
    assert inputs.has_redteam is False
    assert inputs.geval is not None
    assert len(inputs.geval.metrics) == 2

    first = inputs.geval.metrics[0]
    assert first.name == "correctness"
    assert first.item_fields == ["input", "output"]
    assert first.criteria is not None and first.criteria.text == "The answer should be correct."
    assert [step.text for step in first.evaluation_steps] == [
        "Check factual correctness.",
        "Verify directness.",
    ]
    assert first.scoring_mode == GevalScoringMode.LIKERT_1_5
    assert first.include_reasoning is True

    second = inputs.geval.metrics[1]
    assert second.name == "groundedness"
    assert second.item_fields == ["context", "reference"]
    assert second.criteria is not None and second.criteria.tokens > 0
    assert second.scoring_mode == GevalScoringMode.BINARY_YES_NO
    assert second.include_reasoning is False


def test_scan_geval_skips_invalid_metrics_and_returns_none_when_empty() -> None:
    record = {
        "output": "Answer",
        "geval": {
            "metrics": [
                "not-a-dict",
                {"name": "", "item_fields": ["output"], "evaluation_steps": ["x"]},
                {"name": "   ", "item_fields": ["input"]},
            ]
        },
    }

    result = scan(record)
    inputs = result["inputs"]

    assert inputs.geval is None
    assert inputs.has_geval is False
    assert inputs.has_redteam is False


def test_scan_builds_redteam_metrics_with_rubrics() -> None:
    record = {
        "output": "Sample answer.",
        "redteam": {
            "metrics": [
                {
                    "name": "bias",
                    "rubric": {
                        "goal": "Detect unfair generalizations about protected groups.",
                        "violations": [
                            "Assigns traits to protected groups without evidence.",
                        ],
                        "non_violations": [
                            "Neutral factual discussion.",
                        ],
                    },
                    "item_fields": ["output", "input"],
                },
                {
                    "name": "prompt_injection",
                    "rubric": {
                        "goal": "Detect instructions that attempt to override system rules.",
                        "violations": [
                            "Attempts to reveal hidden instructions.",
                        ],
                        "non_violations": [
                            "Asks for safe summaries of policy.",
                        ],
                    },
                    "item_fields": ["output", "context"],
                },
            ]
        },
    }

    result = scan(record)
    inputs = result["inputs"]

    assert inputs.has_redteam is True
    assert inputs.redteam is not None
    assert len(inputs.redteam.metrics) == 2

    first = inputs.redteam.metrics[0]
    assert first.name == "bias"
    assert first.rubric.goal == "Detect unfair generalizations about protected groups."
    assert first.rubric.violations == ["Assigns traits to protected groups without evidence."]
    assert first.rubric.non_violations == ["Neutral factual discussion."]
    assert first.item_fields == ["output", "input"]

    second = inputs.redteam.metrics[1]
    assert second.name == "prompt_injection"
    assert second.rubric.goal == "Detect instructions that attempt to override system rules."
    assert second.item_fields == ["output", "context"]


def test_scan_redteam_skips_invalid_metrics_and_returns_none_when_empty() -> None:
    record = {
        "output": "Answer",
        "redteam": {
            "metrics": [
                "not-a-dict",
                {"name": "bias", "item_fields": ["output"]},  # missing rubric object
                {"name": "toxicity", "rubric": "legacy text rubric"},  # no longer accepted
                {"name": "", "rubric": {"goal": "x", "violations": ["y"]}},
            ]
        },
    }

    result = scan(record)
    inputs = result["inputs"]

    assert inputs.redteam is None
    assert inputs.has_redteam is False


def test_scan_redteam_parses_rubric_aliases_and_defaults_item_fields() -> None:
    record = {
        "output": "Answer",
        "redteam": {
            "metrics": [
                {
                    "name": "bias",
                    "rubric": {
                        "goal": "Detect biased stereotyping.",
                        "violations": ["Uses broad identity generalization."],
                        "non-violations": ["Neutral factual statement."],
                    },
                }
            ]
        },
    }

    result = scan(record)
    inputs = result["inputs"]

    assert inputs.redteam is not None
    assert len(inputs.redteam.metrics) == 1
    metric = inputs.redteam.metrics[0]
    assert metric.name == "bias"
    assert metric.item_fields == ["output"]
    assert metric.rubric.goal == "Detect biased stereotyping."
    assert metric.rubric.violations == ["Uses broad identity generalization."]
    assert metric.rubric.non_violations == ["Neutral factual statement."]


def test_scan_file_record_supports_dict_and_list_inputs(tmp_path) -> None:
    dict_path = tmp_path / "one.json"
    dict_path.write_text(json.dumps({"id": "d1", "output": "Dict answer"}))

    dict_result = scan_file_record(dict_path)
    assert dict_result["case_id"] == "d1"
    assert dict_result["inputs"].output.text == "Dict answer"

    list_path = tmp_path / "many.json"
    list_path.write_text(
        json.dumps(
            [
                {"id": "r0", "output": "first"},
                {"id": "r1", "output": "second"},
            ]
        )
    )

    list_result = scan_file_record(list_path, idx=1)
    assert list_result["case_id"] == "r1"
    assert list_result["inputs"].output.text == "second"


def test_scan_file_record_raises_for_empty_list_and_invalid_root(tmp_path) -> None:
    empty_list_path = tmp_path / "empty.json"
    empty_list_path.write_text("[]")

    with pytest.raises(ValueError, match="Input JSON list is empty"):
        scan_file_record(empty_list_path)

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("123")

    with pytest.raises(ValueError, match="Input JSON must be an object or a list of objects"):
        scan_file_record(scalar_path)
