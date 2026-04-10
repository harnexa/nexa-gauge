# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_scanner.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_scanner.py::test_scan_builds_inputs_with_primary_keys
# uv run pytest -s -k "scanner" packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_scanner.py

import json

import pytest

from lumiseval_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from lumiseval_graph.nodes.scanner import scan, scan_file_record


def test_scan_builds_inputs_with_primary_keys() -> None:
    record = {
        "case_id": "case-123",
        "generation": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "reference": "The capital of France is Paris.",
        "context": "France has Paris as its capital city.",
    }

    result = scan(record, idx=0)

    assert result["case_id"] == "case-123"
    assert result["dataset"] == DEFAULT_DATASET_NAME
    assert result["split"] == DEFAULT_SPLIT
    assert result["reference_files"] == []

    inputs = result["inputs"]
    assert inputs.generation.text == "Paris is the capital of France."
    assert inputs.question is not None and inputs.question.text == "What is the capital of France?"
    assert inputs.reference is not None and inputs.reference.text == "The capital of France is Paris."
    assert inputs.context is not None and inputs.context.text == "France has Paris as its capital city."
    assert inputs.geval is None

    assert inputs.has_generation is True
    assert inputs.has_question is True
    assert inputs.has_reference is True
    assert inputs.has_context is True
    assert inputs.has_geval is False


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
    assert inputs.generation.text == "Berlin is the capital of Germany."
    assert inputs.question is not None and inputs.question.text == "Capital of Germany?"
    assert inputs.reference is not None and inputs.reference.text == "Berlin"
    assert inputs.context is not None
    assert inputs.context.text == "Germany is in Europe.\n\nBerlin is its capital."


def test_scan_keeps_existing_case_defaults() -> None:
    case = {
        "case_id": "pre-set-case-id",
        "dataset": "custom-dataset",
        "split": "custom-split",
        "reference_files": ["/tmp/context.txt"],
    }
    record = {
        "id": "record-id-should-not-overwrite",
        "generation": "Answer",
    }

    result = scan(record, idx=5, case=case)

    assert result["case_id"] == "pre-set-case-id"
    assert result["dataset"] == "custom-dataset"
    assert result["split"] == "custom-split"
    assert result["reference_files"] == ["/tmp/context.txt"]
    assert result["inputs"].generation.text == "Answer"


def test_scan_defaults_case_id_and_flags_when_values_missing() -> None:
    result = scan({}, idx=7)
    inputs = result["inputs"]

    assert result["case_id"] == "record-7"
    assert inputs.generation.text == ""
    assert inputs.question is None
    assert inputs.reference is None
    assert inputs.context is None
    assert inputs.geval is None

    assert inputs.has_generation is False
    assert inputs.has_question is False
    assert inputs.has_reference is False
    assert inputs.has_context is False
    assert inputs.has_geval is False


def test_scan_builds_geval_from_item_fields_only() -> None:
    record = {
        "generation": "Generated answer",
        "geval": {
            "metrics": [
                {
                    "name": "correctness",
                    "item_fields": ["question", "generation"],
                    "criteria": "The answer should be correct.",
                    "evaluation_steps": ["Check factual correctness.", "  ", "Verify directness."],
                },
                {
                    "name": "groundedness",
                    "item_fields": ["context", "reference"],
                    "criteria": "The answer must align with evidence.",
                    "evaluation_steps": ["Compare answer with context."],
                },
            ]
        },
    }

    result = scan(record, idx=2)
    inputs = result["inputs"]

    assert inputs.has_geval is True
    assert inputs.geval is not None
    assert len(inputs.geval.metrics) == 2

    first = inputs.geval.metrics[0]
    assert first.name == "correctness"
    assert first.item_fields == ["question", "generation"]
    assert first.criteria is not None and first.criteria.text == "The answer should be correct."
    assert [step.text for step in first.evaluation_steps] == [
        "Check factual correctness.",
        "Verify directness.",
    ]

    second = inputs.geval.metrics[1]
    assert second.name == "groundedness"
    assert second.item_fields == ["context", "reference"]
    assert second.criteria is not None and second.criteria.tokens > 0


def test_scan_ignores_legacy_record_fields_key() -> None:
    record = {
        "generation": "Generated answer",
        "geval": {
            "metrics": [
                {
                    "name": "legacy-shape",
                    "record_fields": ["context", "reference"],
                    "criteria": "Legacy format should not override new schema.",
                    "evaluation_steps": ["Check this answer."],
                }
            ]
        },
    }

    result = scan(record)
    inputs = result["inputs"]
    assert inputs.geval is not None
    assert len(inputs.geval.metrics) == 1

    metric = inputs.geval.metrics[0]
    # `item_fields` is the only supported schema input; when omitted, default is ["generation"].
    assert metric.item_fields == ["generation"]


def test_scan_geval_skips_invalid_metrics_and_returns_none_when_empty() -> None:
    record = {
        "generation": "Answer",
        "geval": {
            "metrics": [
                "not-a-dict",
                {"name": "", "item_fields": ["generation"], "evaluation_steps": ["x"]},
                {"name": "   ", "item_fields": ["question"]},
            ]
        },
    }

    result = scan(record)
    inputs = result["inputs"]

    assert inputs.geval is None
    assert inputs.has_geval is False


def test_scan_file_record_supports_dict_and_list_inputs(tmp_path) -> None:
    dict_path = tmp_path / "one.json"
    dict_path.write_text(json.dumps({"id": "d1", "generation": "Dict answer"}))

    dict_result = scan_file_record(dict_path)
    assert dict_result["case_id"] == "d1"
    assert dict_result["inputs"].generation.text == "Dict answer"

    list_path = tmp_path / "many.json"
    list_path.write_text(
        json.dumps(
            [
                {"id": "r0", "generation": "first"},
                {"id": "r1", "generation": "second"},
            ]
        )
    )

    list_result = scan_file_record(list_path, idx=1)
    assert list_result["case_id"] == "r1"
    assert list_result["inputs"].generation.text == "second"


def test_scan_file_record_raises_for_empty_list_and_invalid_root(tmp_path) -> None:
    empty_list_path = tmp_path / "empty.json"
    empty_list_path.write_text("[]")

    with pytest.raises(ValueError, match="Input JSON list is empty"):
        scan_file_record(empty_list_path)

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("123")

    with pytest.raises(ValueError, match="Input JSON must be an object or a list of objects"):
        scan_file_record(scalar_path)
