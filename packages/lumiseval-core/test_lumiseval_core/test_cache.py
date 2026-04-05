import json

from lumiseval_core.cache import CacheStore, compute_case_hash
from lumiseval_core.types import GevalConfig, GevalMetricSpec, NodeCostBreakdown


def test_compute_case_hash_changes_when_reference_files_change() -> None:
    h1 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        context=[],
        reference_files=["docs/a.txt"],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        context=[],
        reference_files=["docs/b.txt"],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_geval_steps_change() -> None:
    geval_a = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                record_fields=["generation"],
                evaluation_steps=["Check factual correctness."],
            )
        ]
    )
    geval_b = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                record_fields=["generation"],
                evaluation_steps=["Check factual correctness.", "Check conciseness."],
            )
        ]
    )

    h1 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        geval=geval_a,
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        geval=geval_b,
        context=[],
        reference_files=[],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_context_changes() -> None:
    h1 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        context=["retrieval context"],
        reference_files=[],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_geval_contract_changes() -> None:
    geval_a = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                record_fields=["generation"],
                criteria="Must be factual.",
            )
        ]
    )
    geval_b = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                record_fields=["generation"],
                criteria="Must be concise and factual.",
            )
        ]
    )

    h1 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        geval=geval_a,
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        generation="answer",
        question="q",
        reference="gt",
        geval=geval_b,
        context=[],
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


def test_cache_entry_round_trip_includes_node_cost(tmp_path) -> None:
    store = CacheStore(tmp_path)
    store.put(
        "case",
        "cfg",
        "claims",
        {"raw_claims": []},
        node_cost=NodeCostBreakdown(model_calls=3, cost_usd=0.0123),
    )

    entry = store.get_entry("case", "cfg", "claims")
    assert entry is not None
    assert entry["node_output"] == {"raw_claims": []}
    assert entry["node_cost"] is not None
    assert entry["node_cost"].model_calls == 3
    assert entry["node_cost"].cost_usd == 0.0123


def test_cache_get_entry_backward_compatible_without_node_cost(tmp_path) -> None:
    store = CacheStore(tmp_path)
    path = tmp_path / "case" / "cfg" / "scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "node_name": "scan",
                "case_hash": "case",
                "config_hash": "cfg",
                "created_at": "2026-04-01T00:00:00+00:00",
                "node_output": {"metadata": None},
            }
        )
    )

    entry = store.get_entry("case", "cfg", "scan")
    assert entry is not None
    assert entry["node_output"] == {"metadata": None}
    assert entry["node_cost"] is None
