import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from ng_core.cache import (
    CacheStore,
    cache_read_allowed,
    cache_write_allowed,
    compute_case_hash,
)
from ng_core.types import GevalConfig, GevalMetricSpec, ScoringMode


def _kv_path(tmp_path, cache_key: str):
    digest = hashlib.sha256(cache_key.encode()).hexdigest()
    return tmp_path / "kv" / digest[:2] / f"{digest}.json"


def test_compute_case_hash_changes_when_reference_files_change() -> None:
    h1 = compute_case_hash(
        output="answer",
        input="q",
        reference="gt",
        context=[],
        reference_files=["docs/a.txt"],
    )
    h2 = compute_case_hash(
        output="answer",
        input="q",
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
                item_fields=["output"],
                evaluation_steps=["Check factual correctness."],
            )
        ]
    )
    geval_b = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                item_fields=["output"],
                evaluation_steps=["Check factual correctness.", "Check conciseness."],
            )
        ]
    )

    h1 = compute_case_hash(
        output="answer",
        input="q",
        reference="gt",
        geval=geval_a,
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        output="answer",
        input="q",
        reference="gt",
        geval=geval_b,
        context=[],
        reference_files=[],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_context_changes() -> None:
    h1 = compute_case_hash(
        output="answer",
        input="q",
        reference="gt",
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        output="answer",
        input="q",
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
                item_fields=["output"],
                criteria="Must be factual.",
            )
        ]
    )
    geval_b = GevalConfig(
        metrics=[
            GevalMetricSpec(
                name="factuality",
                item_fields=["output"],
                criteria="Must be concise and factual.",
            )
        ]
    )

    h1 = compute_case_hash(
        output="answer",
        input="q",
        reference="gt",
        geval=geval_a,
        context=[],
        reference_files=[],
    )
    h2 = compute_case_hash(
        output="answer",
        input="q",
        reference="gt",
        geval=geval_b,
        context=[],
        reference_files=[],
    )
    assert h1 != h2


def test_compute_case_hash_changes_when_geval_mode_or_reasoning_changes() -> None:
    """Node-level knobs differentiate case hashes; two modes never collide."""
    from ng_core.types import Geval, GevalMetricInput, Item

    metric_input = GevalMetricInput(
        name="factuality",
        item_fields=["output"],
        criteria=Item(text="Must be factual.", tokens=4.0),
        evaluation_steps=[Item(text="Check.", tokens=1.0)],
    )
    geval_a = Geval(
        metrics=[metric_input],
        scoring_mode=ScoringMode.BINARY_YES_NO,
        include_reasoning=False,
    )
    geval_b = Geval(
        metrics=[metric_input],
        scoring_mode=ScoringMode.SCALE_1_5,
        include_reasoning=False,
    )
    geval_c = Geval(
        metrics=[metric_input],
        scoring_mode=ScoringMode.SCALE_1_5,
        include_reasoning=True,
    )

    h_a = compute_case_hash(output="answer", input="q", reference="gt", geval=geval_a)
    h_b = compute_case_hash(output="answer", input="q", reference="gt", geval=geval_b)
    h_c = compute_case_hash(output="answer", input="q", reference="gt", geval=geval_c)
    assert len({h_a, h_b, h_c}) == 3


def test_compute_case_hash_changes_when_grounding_or_relevance_knobs_change() -> None:
    """Non-default grounding/relevance configs perturb the hash; defaults don't."""
    from ng_core.types import Grounding, Relevance

    h_default = compute_case_hash(output="ans", input="q", reference="gt")
    h_default_explicit = compute_case_hash(
        output="ans",
        input="q",
        reference="gt",
        grounding=Grounding(),
        relevance=Relevance(),
    )
    # Default configs (binary + reasoning off) should be indistinguishable from None.
    assert h_default == h_default_explicit

    h_grounding_scale = compute_case_hash(
        output="ans",
        input="q",
        reference="gt",
        grounding=Grounding(scoring_mode=ScoringMode.SCALE_1_5),
    )
    h_relevance_reasoning = compute_case_hash(
        output="ans",
        input="q",
        reference="gt",
        relevance=Relevance(include_reasoning=True),
    )
    assert h_default != h_grounding_scale
    assert h_default != h_relevance_reasoning
    assert h_grounding_scale != h_relevance_reasoning


def test_cache_get_returns_none_for_corrupt_json(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v3:run:scan:case:route"
    path = _kv_path(tmp_path, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")

    assert store.get_by_key(cache_key) is None


def test_cache_get_returns_none_for_invalid_envelope(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v3:run:scan:case:route"
    path = _kv_path(tmp_path, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unexpected": "shape"}))

    assert store.get_by_key(cache_key) is None


def test_cache_entry_round_trip_includes_metadata(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v3:run:claims:case:route"
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
    cache_key = "v3:run:scan:case:route"
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
    expected_key = "v3:run:scan:case:route"
    actual_key = "v3:run:scan:case:other-route"
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


def test_cache_put_by_key_is_safe_under_concurrent_same_key_writes(tmp_path) -> None:
    store = CacheStore(tmp_path)
    cache_key = "v3:run:scan:same-case:same-route"

    def _write_once(i: int) -> None:
        store.put_by_key(
            cache_key,
            "scan",
            {"marker": i},
            metadata={"writer": str(i)},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write_once, range(50)))

    entry = store.get_entry_by_key(cache_key)
    assert entry is not None
    assert entry["cache_key"] == cache_key
    assert entry["node_name"] == "scan"
    assert isinstance(entry["node_output"]["marker"], int)
    assert all(not p.name.endswith(".tmp") for p in tmp_path.rglob("*"))
