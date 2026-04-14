# Debug commands:
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_geval/test_cache.py
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_geval/test_cache.py::test_collect_geval_signatures_deduplicates_and_skips_preprovided_steps
# uv run pytest -s -k "geval and cache" packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_geval/test_cache.py

from types import SimpleNamespace

from lumoseval_core.types import Item
from lumoseval_graph.nodes.metrics.geval.cache import (
    GevalArtifactCache,
    collect_geval_signatures,
    compute_geval_signature,
)


def test_compute_geval_signature_changes_with_inputs() -> None:
    sig_a = compute_geval_signature(criteria="Must be factual.", model="gpt-4o-mini")
    sig_b = compute_geval_signature(criteria="Must be concise.", model="gpt-4o-mini")
    sig_c = compute_geval_signature(criteria="Must be factual.", model="gpt-4o")

    assert sig_a != sig_b
    assert sig_a != sig_c


def test_put_and_get_steps_with_item_payload(tmp_path) -> None:
    cache = GevalArtifactCache(tmp_path)
    signature = compute_geval_signature(criteria="Mention Paris.", model="gpt-4o-mini")

    cache.put_steps(
        signature=signature,
        model="gpt-4o-mini",
        criteria=Item(text="Mention Paris.", tokens=2),
        evaluation_steps=[Item(text="Check if Paris is mentioned.", tokens=6)],
    )

    steps = cache.get_steps(signature)
    assert steps is not None
    assert len(steps) == 1
    assert steps[0].text == "Check if Paris is mentioned."
    assert cache.count_missing({signature}) == 0
    assert cache.has(signature)


def test_put_steps_accepts_string_criteria_and_string_steps(tmp_path) -> None:
    cache = GevalArtifactCache(tmp_path)
    signature = compute_geval_signature(criteria="Be polite.", model="gpt-4o-mini")

    cache.put_steps(
        signature=signature,
        model="gpt-4o-mini",
        criteria="Be polite.",
        evaluation_steps=["Check for rude or hostile language."],
    )

    artifact = cache.get(signature)
    assert artifact is not None
    assert artifact.criteria.text == "Be polite."
    assert artifact.evaluation_steps[0].text == "Check for rude or hostile language."


def test_collect_geval_signatures_deduplicates_and_skips_preprovided_steps() -> None:
    shared_criteria = Item(text="Response must be factual.", tokens=4)

    case_obj = SimpleNamespace(
        inputs=SimpleNamespace(
            geval=SimpleNamespace(
                metrics=[
                    SimpleNamespace(criteria=shared_criteria, evaluation_steps=[]),
                    SimpleNamespace(
                        criteria=Item(text="Ignored metric.", tokens=2),
                        evaluation_steps=[Item(text="Already provided.", tokens=2)],
                    ),
                ]
            )
        )
    )

    case_dict = {
        "inputs": {
            "geval": {
                "metrics": [
                    {"criteria": {"text": "Response must be factual."}, "evaluation_steps": []},
                    {"criteria": {"text": ""}, "evaluation_steps": []},
                ]
            }
        }
    }

    signatures = collect_geval_signatures(cases=[case_obj, case_dict], model="gpt-4o-mini")
    expected = compute_geval_signature(criteria=shared_criteria.text, model="gpt-4o-mini")

    assert signatures == {expected}
