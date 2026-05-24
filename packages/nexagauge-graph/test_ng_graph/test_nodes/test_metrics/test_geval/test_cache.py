from datetime import datetime, timezone
from types import SimpleNamespace

from ng_core.cache import CacheStore
from ng_core.types import GevalCacheArtifact, Item
from ng_graph.nodes.metrics.geval.cache import (
    build_geval_artifact_cache_key,
    collect_geval_signatures,
    compute_geval_signature,
)


def test_compute_geval_signature_changes_with_inputs() -> None:
    sig_a = compute_geval_signature(
        criteria="Must be factual.", item_fields=["output"], model="gpt-4o-mini"
    )
    sig_b = compute_geval_signature(
        criteria="Must be concise.", item_fields=["output"], model="gpt-4o-mini"
    )
    sig_c = compute_geval_signature(
        criteria="Must be factual.", item_fields=["output"], model="gpt-4o"
    )

    assert sig_a != sig_b
    assert sig_a != sig_c


def test_signature_depends_on_item_fields() -> None:
    base = dict(criteria="Score quality.", model="gpt-4o-mini")
    sig_a = compute_geval_signature(item_fields=["output"], **base)
    sig_b = compute_geval_signature(item_fields=["output", "reference"], **base)

    assert sig_a != sig_b


def test_signature_stable_under_field_order() -> None:
    base = dict(criteria="Score quality.", model="gpt-4o-mini")
    sig_a = compute_geval_signature(item_fields=["output", "reference"], **base)
    sig_b = compute_geval_signature(item_fields=["reference", "output"], **base)

    assert sig_a == sig_b


def test_signature_ignores_evaluation_steps() -> None:
    """Cache key must depend only on what the prompt depends on.

    The step-generator prompt sees criteria + item_fields only — any caller-
    supplied ``evaluation_steps`` is irrelevant to what the LLM would produce,
    so it must not be part of the cache key. Two metrics with identical
    criteria+item_fields share cached steps regardless of whatever
    evaluation_steps they happened to ship with.
    """

    sig = compute_geval_signature(
        criteria="Must be factual.", item_fields=["output"], model="gpt-4o-mini"
    )
    assert compute_geval_signature.__code__.co_varnames  # satisfy linters
    # compute_geval_signature has no evaluation_steps parameter, so there's
    # no way for caller-supplied steps to leak into the signature.
    import inspect

    assert "evaluation_steps" not in inspect.signature(compute_geval_signature).parameters
    # Determinism check: same inputs → same key, regardless of any other state.
    assert sig == compute_geval_signature(
        criteria="Must be factual.", item_fields=["output"], model="gpt-4o-mini"
    )


def test_build_geval_artifact_cache_key_stable() -> None:
    sig = compute_geval_signature(
        criteria="Be factual.", item_fields=["output"], model="gpt-4o-mini"
    )
    assert build_geval_artifact_cache_key(sig) == build_geval_artifact_cache_key(sig)

    other = compute_geval_signature(
        criteria="Be concise.", item_fields=["output"], model="gpt-4o-mini"
    )
    assert build_geval_artifact_cache_key(sig) != build_geval_artifact_cache_key(other)
    assert build_geval_artifact_cache_key(sig).startswith("v2:geval_artifact:")


def test_roundtrip_via_cache_store(tmp_path) -> None:
    store = CacheStore(tmp_path)
    signature = compute_geval_signature(
        criteria="Mention Paris.", item_fields=["output"], model="gpt-4o-mini"
    )
    artifact = GevalCacheArtifact(
        signature=signature,
        model="gpt-4o-mini",
        prompt_version="v2",
        parser_version="v2",
        item_fields=["output"],
        criteria=Item(text="Mention Paris.", tokens=2),
        evaluation_steps=[
            Item(text="Check if Paris is mentioned.", tokens=6),
            Item(text="Confirm the mention is factually correct.", tokens=7),
        ],
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    key = build_geval_artifact_cache_key(signature)
    store.put_by_key(
        key,
        "geval_steps_artifact",
        {"geval_artifact": artifact},
        metadata={"signature": signature},
    )

    entry = store.get_entry_by_key(key)
    assert entry is not None
    restored = entry["node_output"]["geval_artifact"]
    assert isinstance(restored, GevalCacheArtifact)
    assert restored.signature == signature
    assert restored.item_fields == ["output"]
    assert [s.text for s in restored.evaluation_steps] == [
        "Check if Paris is mentioned.",
        "Confirm the mention is factually correct.",
    ]


def test_collect_geval_signatures_deduplicates_and_skips_preprovided_steps() -> None:
    shared_criteria = Item(text="Response must be factual.", tokens=4)

    case_obj = SimpleNamespace(
        inputs=SimpleNamespace(
            geval=SimpleNamespace(
                metrics=[
                    SimpleNamespace(
                        criteria=shared_criteria,
                        evaluation_steps=[],
                        item_fields=["output"],
                    ),
                    SimpleNamespace(
                        criteria=Item(text="Ignored metric.", tokens=2),
                        evaluation_steps=[Item(text="Already provided.", tokens=2)],
                        item_fields=["output"],
                    ),
                ]
            )
        )
    )

    case_dict = {
        "inputs": {
            "geval": {
                "metrics": [
                    {
                        "criteria": {"text": "Response must be factual."},
                        "evaluation_steps": [],
                        "item_fields": ["output"],
                    },
                    {
                        "criteria": {"text": ""},
                        "evaluation_steps": [],
                        "item_fields": ["output"],
                    },
                ]
            }
        }
    }

    signatures = collect_geval_signatures(cases=[case_obj, case_dict], model="gpt-4o-mini")
    expected = compute_geval_signature(
        criteria=shared_criteria.text, item_fields=["output"], model="gpt-4o-mini"
    )

    assert signatures == {expected}
