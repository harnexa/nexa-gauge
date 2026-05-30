# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_grounding.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_grounding.py::test_run_returns_metric_and_cost_with_mocked_llm
# uv run pytest -s -k "grounding" packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_grounding.py

from types import SimpleNamespace

import pytest
from ng_core.types import Claim, Item, ScoringMode
from ng_graph.nodes.metrics import grounding as grounding_module
from ng_graph.nodes.metrics.grounding import GroundingNode


def _claims() -> list[Claim]:
    return [
        Claim(item=Item(text="The Eiffel Tower is in Paris.", tokens=8), source_chunk_index=0),
        Claim(item=Item(text="The Eiffel Tower is in Berlin.", tokens=8), source_chunk_index=1),
    ]


def test_run_returns_metric_and_cost_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(scores=[1, 0]),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 20,
                    "total_tokens": 140,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(grounding_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = GroundingNode(judge_model="gpt-4o-mini")
    result = node.run(claims=_claims(), context="Paris is in France.")

    assert len(result.metrics) == 1
    metric = result.metrics[0]
    assert metric.name == "grounding"
    assert metric.score == 0.5
    assert metric.verdict == "FAILED"

    assert result.cost.input_tokens == 120
    assert result.cost.output_tokens == 20
    assert result.cost.cost > 0


def test_run_skips_when_disabled_or_no_context() -> None:
    node = GroundingNode(judge_model="gpt-4o-mini")

    disabled = node.run(claims=_claims(), context="Paris is in France.", enable_grounding=False)
    assert disabled.metrics == []
    assert disabled.cost.cost == 0.0
    assert disabled.cost.input_tokens is None
    assert disabled.cost.output_tokens is None

    no_context = node.run(claims=_claims(), context="")
    assert no_context.metrics == []
    assert no_context.cost.cost == 0.0
    assert no_context.cost.input_tokens is None
    assert no_context.cost.output_tokens is None


def test_run_handles_no_verdicts_as_error_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(scores=[]),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 10,
                    "total_tokens": 90,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(grounding_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = GroundingNode(judge_model="gpt-4o-mini")
    result = node.run(claims=_claims(), context="Paris is in France.")

    assert len(result.metrics) == 1
    assert result.metrics[0].error == "No scores returned"
    assert result.metrics[0].verdict is None
    assert result.cost.input_tokens == 80


def test_run_scale_mode_normalizes_integer_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """In likert mode the judge returns 1-5 integers; each is normalized via (raw-1)/4."""
    captured: list = []

    class FakeLLM:
        def invoke(self, messages):
            captured.append(messages)
            return {
                "parsed": SimpleNamespace(scores=[5, 1]),  # one fully supported, one not
                "parsing_error": None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 15, "total_tokens": 115},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(grounding_module, "get_llm", lambda *_a, **_kw: FakeLLM())
    node = GroundingNode(judge_model="gpt-4o-mini")
    result = node.run(
        claims=_claims(),
        context="Paris is in France.",
        scoring_mode=ScoringMode.SCALE_1_5,
    )

    metric = result.metrics[0]
    # (5-1)/4 = 1.0; (1-1)/4 = 0.0; mean = 0.5.
    assert metric.score == 0.5
    # The output contract should enforce integer score list semantics.
    output_contract = captured[0][1]["content"]
    assert "list of integers" in output_contract
    assert "scale_1_5" in output_contract


def test_run_with_reasoning_surfaces_batch_rationale(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    scores=[1, 1],
                    reasoning="Both claims are supported by the context.",
                ),
                "parsing_error": None,
                "usage": {"prompt_tokens": 90, "completion_tokens": 25, "total_tokens": 115},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(grounding_module, "get_llm", lambda *_a, **_kw: FakeLLM())
    node = GroundingNode(judge_model="gpt-4o-mini")
    result = node.run(
        claims=_claims(),
        context="Paris is in France.",
        include_reasoning=True,
    )

    metric = result.metrics[0]
    assert metric.score == 1.0
    # Reasoning is appended after the per-claim verdict list.
    assert isinstance(metric.result, list) and len(metric.result) > len(_claims())
    assert metric.result[-1] == {"reasoning": "Both claims are supported by the context."}
