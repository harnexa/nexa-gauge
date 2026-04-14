# Debug commands:
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_geval/test_steps.py
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_geval/test_steps.py::test_run_uses_cache_without_llm_call
# uv run pytest -s -k "geval and steps" packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_geval/test_steps.py

from types import SimpleNamespace

import lumoseval_graph.nodes.metrics.geval.steps as steps_module
import pytest
from lumoseval_core.types import GevalMetricInput, Item
from lumoseval_graph.nodes.metrics.geval.cache import GevalArtifactCache, compute_geval_signature
from lumoseval_graph.nodes.metrics.geval.steps import GevalStepsNode


def test_run_returns_empty_when_disabled_or_no_metrics(tmp_path) -> None:
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache=GevalArtifactCache(tmp_path))

    disabled = node.run(metrics=[], enable_geval=False)
    assert disabled.resolved_steps == []
    assert disabled.cost is not None
    assert disabled.cost.cost == 0.0
    assert disabled.cost.input_tokens is None
    assert disabled.cost.output_tokens is None

    empty_metrics = node.run(metrics=[], enable_geval=True)
    assert empty_metrics.resolved_steps == []
    assert empty_metrics.cost is not None
    assert empty_metrics.cost.cost == 0.0


def test_run_uses_provided_steps_and_duplicate_metric_keys(tmp_path) -> None:
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache=GevalArtifactCache(tmp_path))

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=None,
            evaluation_steps=[Item(text="Check factual accuracy.", tokens=4)],
        ),
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=None,
            evaluation_steps=[Item(text="Flag unsupported statements.", tokens=4)],
        ),
    ]

    result = node.run(metrics=metrics, enable_geval=True)

    assert [m.key for m in result.resolved_steps] == ["factuality#1", "factuality#2"]
    assert [m.steps_source for m in result.resolved_steps] == ["provided", "provided"]
    assert all(m.signature is None for m in result.resolved_steps)
    assert result.cost is not None
    assert result.cost.cost == 0.0


def test_run_uses_cache_without_llm_call(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = GevalArtifactCache(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache=cache)
    criteria = Item(text="The answer should be factually correct.", tokens=8)

    signature = compute_geval_signature(
        criteria=criteria.text,
        model="gpt-4o-mini",
        prompt_version=node.prompt_version,
        parser_version=node.parser_version,
    )
    cache.put_steps(
        signature=signature,
        model="gpt-4o-mini",
        criteria=criteria,
        evaluation_steps=[Item(text="Verify every claim against known facts.", tokens=8)],
        prompt_version=node.prompt_version,
        parser_version=node.parser_version,
    )

    def _should_not_call_llm(*_args, **_kwargs):
        pytest.fail("LLM should not be called when steps are already cached")

    monkeypatch.setattr(steps_module, "get_llm", _should_not_call_llm)

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=criteria,
            evaluation_steps=[],
        )
    ]
    result = node.run(metrics=metrics, enable_geval=True)

    assert len(result.resolved_steps) == 1
    resolved = result.resolved_steps[0]
    assert resolved.steps_source == "cache_used"
    assert resolved.signature == signature
    assert resolved.evaluation_steps[0].text == "Verify every claim against known facts."
    assert result.cost is not None
    assert result.cost.cost == 0.0
    assert result.cost.input_tokens is None
    assert result.cost.output_tokens is None


def test_run_generates_steps_then_hits_cache_on_second_run(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = GevalArtifactCache(tmp_path)
    node = GevalStepsNode(judge_model="gpt-4o-mini", artifact_cache=cache)

    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    evaluation_steps=[
                        "Check factual correctness of each statement.",
                        "Penalize unsupported or invented claims.",
                    ]
                ),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(steps_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    metrics = [
        GevalMetricInput(
            name="factuality",
            item_fields=["generation"],
            criteria=Item(text="The answer must be factual.", tokens=6),
            evaluation_steps=[],
        )
    ]

    first = node.run(metrics=metrics, enable_geval=True)
    assert len(first.resolved_steps) == 1
    generated = first.resolved_steps[0]
    assert generated.steps_source == "generated"
    assert generated.signature is not None
    assert len(generated.evaluation_steps) == 2
    assert first.cost is not None
    assert first.cost.input_tokens == 120
    assert first.cost.output_tokens == 30
    assert first.cost.cost > 0

    cached_steps = cache.get_steps(generated.signature)
    assert cached_steps is not None
    assert len(cached_steps) == 2

    def _should_not_call_llm(*_args, **_kwargs):
        pytest.fail("Second run should use cache and skip LLM call")

    monkeypatch.setattr(steps_module, "get_llm", _should_not_call_llm)
    second = node.run(metrics=metrics, enable_geval=True)
    assert second.resolved_steps[0].steps_source == "cache_used"
