import hashlib
import json

from lumiseval_core.cache import (
    CacheStore,
    cache_read_allowed,
    cache_write_allowed,
    compute_case_hash,
)
from lumiseval_core.types import GevalConfig, GevalMetricSpec


def _kv_path(tmp_path, cache_key: str):
    digest = hashlib.sha256(cache_key.encode()).hexdigest()
    return tmp_path / "kv" / digest[:2] / f"{digest}.json"


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
                item_fields=["generation"],
                evaluation_steps=["Check factual correctness."],
            )
        ]
    )
    geval_b = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                item_fields=["generation"],
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
                item_fields=["generation"],
                criteria="Must be factual.",
            )
        ]
    )
    geval_b = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                item_fields=["generation"],
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
    cache_key = "v2:run:scan:case:route"
    path = _kv_path(tmp_path, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")

    assert store.get_by_key(cache_key) is None


def test_cache_get_returns_none_for_invalid_envelope(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v2:run:scan:case:route"
    path = _kv_path(tmp_path, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unexpected": "shape"}))

    assert store.get_by_key(cache_key) is None


def test_cache_entry_round_trip_includes_metadata(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v2:run:claims:case:route"
    store.put_by_key(
        cache_key,
        "claims",
        {"raw_claims": []},
        metadata={"execution_mode": "run"},
    )

    entry = store.get_entry_by_key(cache_key)
    assert entry is not None
    assert entry["cache_key"] == cache_key
    assert entry["node_name"] == "claims"
    assert entry["node_output"] == {"raw_claims": []}
    assert entry["metadata"] == {"execution_mode": "run"}


def test_cache_get_entry_defaults_metadata_to_empty_dict(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v2:run:scan:case:route"
    path = _kv_path(tmp_path, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "node_name": "scan",
                "created_at": "2026-04-01T00:00:00+00:00",
                "node_output": {"metadata": None},
            }
        )
    )

    entry = store.get_entry_by_key(cache_key)
    assert entry is not None
    assert entry["node_output"] == {"metadata": None}
    assert entry["metadata"] == {}


def test_cache_get_entry_returns_none_when_cache_key_mismatch(tmp_path) -> None:
    store = CacheStore(tmp_path)
    expected_key = "v2:run:scan:case:route"
    actual_key = "v2:run:scan:case:other-route"
    path = _kv_path(tmp_path, expected_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cache_key": actual_key,
                "node_name": "scan",
                "created_at": "2026-04-01T00:00:00+00:00",
                "node_output": {"inputs": None},
            }
        )
    )

    assert store.get_entry_by_key(expected_key) is None


def test_eval_and_report_nodes_are_non_cacheable() -> None:
    assert cache_read_allowed(execution_mode="run", node_name="eval") is False
    assert cache_write_allowed(execution_mode="run", node_name="eval") is False
    assert cache_read_allowed(execution_mode="estimate", node_name="report") is False
    assert cache_write_allowed(execution_mode="estimate", node_name="report") is False
