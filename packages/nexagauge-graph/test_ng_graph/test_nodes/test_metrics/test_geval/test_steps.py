from datetime import datetime, timezone
from types import SimpleNamespace

import ng_graph.nodes.metrics.geval.steps as steps_module
import pytest
from ng_core.cache import CacheStore, NoOpCacheStore
from ng_core.types import GevalCacheArtifact, GevalMetricInput, Item
from ng_graph.nodes.metrics.geval.cache import (
    build_geval_artifact_cache_key,
    compute_geval_signature,
)
from ng_graph.nodes.metrics.geval.steps import GevalStepsNode


def _make_fake_llm(steps: list[str], prompt_tokens: int = 120, completion_tokens: int = 30):
    captured: dict[str, list] = {"messages": []}

    class FakeLLM:
        def invoke(self, messages):
            captured["messages"].append(messages)
            return {
                "parsed": SimpleNamespace(evaluation_steps=list(steps)),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "model": "gpt-4o-mini",
            }

    return FakeLLM(), captured


def _seed_cache(
    store: CacheStore,
    *,
    signature: str,
    item_fields: list[str],
    criteria: Item,
    steps: list[Item],
    model: str = "gpt-4o-mini",
) -> None:
    artifact = GevalCacheArtifact(
        signature=signature,
        model=model,
        prompt_version="v2",
        parser_version="v2",
        item_fields=item_fields,
        criteria=criteria,
        evaluation_steps=steps,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.put_by_key(
        build_geval_artifact_cache_key(signature),
        "geval_steps_artifact",
        {"geval_artifact": artifact},
    )


def test_run_returns_empty_when_disabled_or_no_metrics(tmp_path) -> None:
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=CacheStore(tmp_path))

    disabled = node.run(metrics=[], enable_geval=False)
    assert disabled.resolved_steps == []
    assert disabled.cost is not None
    assert disabled.cost.cost == 0.0

    empty = node.run(metrics=[], enable_geval=True)
    assert empty.resolved_steps == []
    assert empty.cost is not None
    assert empty.cost.cost == 0.0


def test_provided_steps_bypass_llm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=CacheStore(tmp_path))

    def _should_not_call_llm(*_args, **_kwargs):
        pytest.fail("LLM must not be called when evaluation_steps are provided")

    monkeypatch.setattr(steps_module, "get_llm", _should_not_call_llm)

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=None,
            evaluation_steps=[Item(text="Check factual accuracy.", tokens=4)],
        )
    ]

    result = node.run(metrics=metrics, enable_geval=True)
    assert len(result.resolved_steps) == 1
    assert result.resolved_steps[0].steps_source == "provided"
    assert result.resolved_steps[0].signature is None


def test_run_uses_cache_without_llm_call(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CacheStore(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=store)
    criteria = Item(text="The answer should be factually correct.", tokens=8)
    item_fields = ["generation"]

    signature = compute_geval_signature(
        criteria=criteria.text,
        item_fields=item_fields,
        model="gpt-4o-mini",
        prompt_version=node.prompt_version,
        parser_version=node.parser_version,
    )
    _seed_cache(
        store,
        signature=signature,
        item_fields=item_fields,
        criteria=criteria,
        steps=[
            Item(text="Verify every claim against known facts.", tokens=8),
            Item(text="Penalize unsupported assertions.", tokens=6),
            Item(text="Check the generation for hallucinations.", tokens=7),
        ],
    )

    def _should_not_call_llm(*_args, **_kwargs):
        pytest.fail("LLM should not be called when steps are already cached")

    monkeypatch.setattr(steps_module, "get_llm", _should_not_call_llm)

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=item_fields,
            criteria=criteria,
            evaluation_steps=[],
        )
    ]
    result = node.run(metrics=metrics, enable_geval=True)

    assert len(result.resolved_steps) == 1
    resolved = result.resolved_steps[0]
    assert resolved.steps_source == "cache_used"
    assert resolved.signature == signature
    assert result.cost is not None
    assert result.cost.cost == 0.0


def test_run_generates_steps_then_hits_cache_on_second_run(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CacheStore(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=store)

    fake_llm, _captured = _make_fake_llm(
        [
            "Verify every claim against known facts.",
            "Penalize unsupported or invented claims.",
            "Flag statements that contradict the reference.",
        ]
    )
    monkeypatch.setattr(steps_module, "get_llm", lambda *_a, **_kw: fake_llm)

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=Item(text="The answer must be factual.", tokens=6),
            evaluation_steps=[],
        )
    ]

    first = node.run(metrics=metrics, enable_geval=True)
    generated = first.resolved_steps[0]
    assert generated.steps_source == "generated"
    assert generated.signature is not None
    assert len(generated.evaluation_steps) == 3
    assert first.cost is not None
    assert first.cost.input_tokens == 120
    assert first.cost.output_tokens == 30

    def _should_not_call_llm(*_args, **_kwargs):
        pytest.fail("Second run should use cache and skip LLM call")

    monkeypatch.setattr(steps_module, "get_llm", _should_not_call_llm)
    second = node.run(metrics=metrics, enable_geval=True)
    assert second.resolved_steps[0].steps_source == "cache_used"


def test_no_cache_store_always_generates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-cache path: NoOpCacheStore never returns nor persists entries."""

    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=NoOpCacheStore())

    call_count = {"n": 0}

    class CountingFakeLLM:
        def invoke(self, _messages):
            call_count["n"] += 1
            return {
                "parsed": SimpleNamespace(
                    evaluation_steps=[
                        "Verify every factual claim.",
                        "Penalize unsupported assertions.",
                        "Flag contradictions with the reference.",
                    ]
                ),
                "parsing_error": None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(steps_module, "get_llm", lambda *_a, **_kw: CountingFakeLLM())

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=Item(text="The answer must be factual.", tokens=6),
            evaluation_steps=[],
        )
    ]

    first = node.run(metrics=metrics, enable_geval=True)
    assert first.resolved_steps[0].steps_source == "generated"

    second = node.run(metrics=metrics, enable_geval=True)
    assert second.resolved_steps[0].steps_source == "generated"

    assert call_count["n"] == 2


def test_shared_criteria_reuses_generated_steps_across_cases(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact cache is keyed only on (criteria, item_fields, model, versions).

    Simulates two separate cases (same `GevalStepsNode.run(...)` is called
    twice) with the same criterion but different metric names. The first call
    generates; the second must hit the artifact cache and skip the LLM.
    This is the cross-case reuse the G-Eval paper optimizes for.
    """

    store = CacheStore(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=store)

    call_count = {"n": 0}

    class CountingFakeLLM:
        def invoke(self, _messages):
            call_count["n"] += 1
            return {
                "parsed": SimpleNamespace(
                    evaluation_steps=[
                        "Verify every factual claim.",
                        "Penalize unsupported assertions.",
                        "Flag contradictions with the reference.",
                    ]
                ),
                "parsing_error": None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(steps_module, "get_llm", lambda *_a, **_kw: CountingFakeLLM())

    criteria = Item(text="The answer must be factual.", tokens=6)

    # Case A: metric name "factuality_case_a"
    result_a = node.run(
        metrics=[
            GevalMetricInput(
                name="factuality_case_a",
                item_fields=["generation"],
                criteria=criteria,
                evaluation_steps=[],
            )
        ],
        enable_geval=True,
    )
    assert result_a.resolved_steps[0].steps_source == "generated"
    assert call_count["n"] == 1

    # Case B: different metric name, same criteria+item_fields.
    # Must reuse the cached artifact from Case A — no new LLM call.
    result_b = node.run(
        metrics=[
            GevalMetricInput(
                name="factuality_case_b",
                item_fields=["generation"],
                criteria=criteria,
                evaluation_steps=[],
            )
        ],
        enable_geval=True,
    )
    assert result_b.resolved_steps[0].steps_source == "cache_used"
    assert result_b.resolved_steps[0].signature == result_a.resolved_steps[0].signature
    assert call_count["n"] == 1  # still 1 — B did not call the LLM


def test_same_criteria_different_fields_yield_different_cache_entries(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CacheStore(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=store)

    call_count = {"n": 0}

    class CountingFakeLLM:
        def invoke(self, _messages):
            call_count["n"] += 1
            return {
                "parsed": SimpleNamespace(
                    evaluation_steps=[
                        "Step one for this configuration.",
                        "Step two for this configuration.",
                        "Step three for this configuration.",
                    ]
                ),
                "parsing_error": None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(steps_module, "get_llm", lambda *_a, **_kw: CountingFakeLLM())

    criteria = Item(text="The answer must be factual.", tokens=6)
    metrics = [
        GevalMetricInput(
            name="factuality_gen",
            item_fields=["generation"],
            criteria=criteria,
            evaluation_steps=[],
        ),
        GevalMetricInput(
            name="factuality_ref",
            item_fields=["generation", "reference"],
            criteria=criteria,
            evaluation_steps=[],
        ),
    ]

    result = node.run(metrics=metrics, enable_geval=True)
    assert call_count["n"] == 2
    sig_a, sig_b = (m.signature for m in result.resolved_steps)
    assert sig_a != sig_b


def test_minimum_step_count_enforced(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CacheStore(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=store)

    class ShortFakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    evaluation_steps=["Only one step."],
                ),
                "parsing_error": None,
                "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(steps_module, "get_llm", lambda *_a, **_kw: ShortFakeLLM())

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=Item(text="Must be factual.", tokens=4),
            evaluation_steps=[],
        )
    ]

    with pytest.raises(RuntimeError, match="factuality"):
        node.run(metrics=metrics, enable_geval=True)


def test_prompt_contains_param_names(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CacheStore(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache_store=store)

    fake_llm, captured = _make_fake_llm(
        [
            "Check factual correctness of each statement.",
            "Penalize unsupported claims.",
            "Assess alignment with the reference answer.",
        ]
    )
    monkeypatch.setattr(steps_module, "get_llm", lambda *_a, **_kw: fake_llm)

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation", "reference"],
            criteria=Item(text="Must be factual.", tokens=4),
            evaluation_steps=[],
        )
    ]
    node.run(metrics=metrics, enable_geval=True)

    user_content = captured["messages"][0][1]["content"]
    assert "Actual Output" in user_content
    assert "Expected Output" in user_content
