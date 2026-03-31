import json

from lumiseval_core.cache import CacheStore, compute_case_hash
from lumiseval_core.types import Rubric


def test_compute_case_hash_changes_when_reference_files_change() -> None:
    h1 = compute_case_hash(
        generation="answer",
        question="q",
        ground_truth="gt",
        rubric=[],
        context=[],
        reference_files=["docs/a.txt"],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        ground_truth="gt",
        rubric=[],
        context=[],
        reference_files=["docs/b.txt"],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_rubric_pass_condition_changes() -> None:
    rules_a = [
        Rubric(id="R-1", statement="Mention year", pass_condition="Must include a year.")
    ]
    rules_b = [
        Rubric(
            id="R-1", statement="Mention year", pass_condition="Must include month and year."
        )
    ]

    h1 = compute_case_hash(
        generation="answer",
        question="q",
        ground_truth="gt",
        rubric=rules_a,
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        ground_truth="gt",
        rubric=rules_b,
        context=[],
        reference_files=[],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_context_changes() -> None:
    h1 = compute_case_hash(
        generation="answer",
        question="q",
        ground_truth="gt",
        rubric=[],
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        ground_truth="gt",
        rubric=[],
        context=["retrieval context"],
        reference_files=[],
    )
    assert h1 != h2


def test_cache_get_returns_none_for_corrupt_json(tmp_path) -> None:
    store = CacheStore(tmp_path)
    path = tmp_path / "case" / "cfg" / "scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")

    assert store.get("case", "cfg", "scan") is None


def test_cache_get_returns_none_for_invalid_envelope(tmp_path) -> None:
    store = CacheStore(tmp_path)
    path = tmp_path / "case" / "cfg" / "scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unexpected": "shape"}))

    assert store.get("case", "cfg", "scan") is None
