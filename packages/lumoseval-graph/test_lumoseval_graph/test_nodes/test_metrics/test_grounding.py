# Debug commands:
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_grounding.py
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_grounding.py::test_run_returns_metric_and_cost_with_mocked_llm
# uv run pytest -s -k "grounding" packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_grounding.py

from types import SimpleNamespace

import pytest
from lumoseval_core.types import Claim, Item
from lumoseval_graph.nodes.metrics import grounding as grounding_module
from lumoseval_graph.nodes.metrics.grounding import GroundingNode


def _claims() -> list[Claim]:
    return [
        Claim(item=Item(text="The Eiffel Tower is in Paris.", tokens=8), source_chunk_index=0),
        Claim(item=Item(text="The Eiffel Tower is in Berlin.", tokens=8), source_chunk_index=1),
    ]


def test_run_returns_metric_and_cost_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(verdicts=[True, False]),
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
                "parsed": SimpleNamespace(verdicts=[]),
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
    assert result.metrics[0].error == "No verdicts returned"
    assert result.cost.input_tokens == 80
