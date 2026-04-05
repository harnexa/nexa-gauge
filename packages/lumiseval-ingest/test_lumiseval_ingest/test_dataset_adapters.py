import json

import pytest
from lumiseval_core.errors import InputParseError
from lumiseval_ingest.adapters import (
    HuggingFaceDatasetAdapter,
    LocalFileDatasetAdapter,
    create_dataset_adapter,
)


def test_local_file_adapter_maps_jsonl_to_eval_cases(tmp_path) -> None:
    dataset_file = tmp_path / "cases.jsonl"
    rows = [
        {
            "id": "row-1",
            "question": "What is Paris?",
            "generation": "Paris is the capital of France.",
            "context": "Paris is France's capital city.",
            "reference": "Paris is the capital of France.",
            "geval": {
                "metrics": [
                    {
                        "name": "factuality",
                        "record_fields": ["generation"],
                        "criteria": "Answer should be factual.",
                    }
                ]
            },
            "extra_field": "x",
        },
        {
            "id": "row-2",
            "query": "Who invented the telephone?",
            "response": "Alexander Graham Bell is credited with inventing the telephone.",
            "contexts": ["Bell patented the telephone in 1876."],
        },
    ]
    dataset_file.write_text("\n".join(json.dumps(row) for row in rows))

    adapter = LocalFileDatasetAdapter(dataset_file, dataset_name="demo")
    cases = list(adapter.iter_cases(split="validation"))

    assert len(cases) == 2
    assert cases[0].case_id == "row-1"
    assert cases[0].dataset == "demo"
    assert cases[0].split == "validation"
    assert cases[0].question == "What is Paris?"
    assert cases[0].generation == "Paris is the capital of France."
    assert cases[0].context == ["Paris is France's capital city."]
    assert cases[0].geval is not None
    assert len(cases[0].geval.metrics) == 1
    assert cases[0].metadata["extra_field"] == "x"

    assert cases[1].case_id == "row-2"
    assert cases[1].question == "Who invented the telephone?"
    assert "Alexander Graham Bell" in cases[1].generation


def test_local_file_adapter_raises_for_missing_generation(tmp_path) -> None:
    dataset_file = tmp_path / "bad.jsonl"
    dataset_file.write_text(json.dumps({"id": "missing-gen", "question": "q"}))

    adapter = LocalFileDatasetAdapter(dataset_file)
    with pytest.raises(InputParseError):
        list(adapter.iter_cases())


def test_local_file_adapter_accepts_geval_contract_and_autoincludes_generation(tmp_path) -> None:
    dataset_file = tmp_path / "geval.json"
    dataset_file.write_text(
        json.dumps(
            {
                "case_id": "g-1",
                "generation": "Paris is in France.",
                "geval": {
                    "metrics": [
                        {
                            "name": "factuality",
                            "record_fields": ["question"],
                            "criteria": "Response must be factually correct.",
                        }
                    ]
                },
            }
        )
    )

    adapter = LocalFileDatasetAdapter(dataset_file, dataset_name="demo")
    cases = list(adapter.iter_cases(split="train"))

    assert len(cases) == 1
    assert cases[0].geval is not None
    metric = cases[0].geval.metrics[0]
    assert metric.record_fields == ["question", "generation"]


def test_local_file_adapter_rejects_invalid_geval_contract(tmp_path) -> None:
    dataset_file = tmp_path / "bad-geval.json"
    dataset_file.write_text(
        json.dumps(
            {
                "generation": "hello",
                "geval": {
                    "metrics": [
                        {
                            "name": "bad",
                            "record_fields": ["generation", "unknown_field"],
                            "criteria": "x",
                            "evaluation_steps": ["y"],
                        }
                    ]
                },
            }
        )
    )

    adapter = LocalFileDatasetAdapter(dataset_file)
    with pytest.raises(InputParseError):
        list(adapter.iter_cases())


def test_adapter_registry_auto_modes(tmp_path) -> None:
    dataset_file = tmp_path / "single.json"
    dataset_file.write_text(json.dumps({"generation": "hello"}))

    local_adapter = create_dataset_adapter(dataset_file, adapter="auto")
    assert isinstance(local_adapter, LocalFileDatasetAdapter)

    hf_adapter = create_dataset_adapter("hf://HuggingFaceH4/mt_bench_prompts", adapter="auto")
    assert isinstance(hf_adapter, HuggingFaceDatasetAdapter)
