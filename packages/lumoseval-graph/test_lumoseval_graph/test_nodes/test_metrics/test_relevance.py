# Debug commands:
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_relevance.py
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_relevance.py::test_run_returns_metric_and_cost_with_mocked_llm
# uv run pytest -s -k "relevance" packages/lumos-graph/test_lumos_graph/test_nodes/test_metrics/test_relevance.py

from types import SimpleNamespace

import lumoseval_graph.nodes.metrics.relevance as relevance_module
import pytest
from lumoseval_core.types import Claim, Item
from lumoseval_graph.nodes.metrics.relevance import RelevanceNode


def _claims() -> list[Claim]:
    return [
        Claim(item=Item(text="Paris is the capital of France.", tokens=8), source_chunk_index=0),
        Claim(item=Item(text="Transformers use self-attention.", tokens=6), source_chunk_index=1),
    ]


def test_run_returns_metric_and_cost_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    verdicts=[
                        True,
                        False,
                    ]
                ),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 20,
                    "total_tokens": 130,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(relevance_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = RelevanceNode(judge_model="gpt-4o-mini")
    result = node.run(
        claims=_claims(), question=Item(text="What is the capital of France?", tokens=8)
    )

    assert len(result.metrics) == 1
    metric = result.metrics[0]
    assert metric.name == "answer_relevancy"
    assert metric.score == 0.5

    assert result.cost.input_tokens == 110
    assert result.cost.output_tokens == 20
    assert result.cost.cost > 0


def test_run_skips_when_disabled_or_no_question() -> None:
    node = RelevanceNode(judge_model="gpt-4o-mini")

    disabled = node.run(
        claims=_claims(), question="What is the capital of France?", enable_relevance=False
    )
    assert disabled.metrics == []
    assert disabled.cost.cost == 0.0

    no_question = node.run(claims=_claims(), question="")
    assert no_question.metrics == []
    assert no_question.cost.cost == 0.0


def test_run_handles_no_verdicts_as_error_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(verdicts=[]),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 90,
                    "completion_tokens": 10,
                    "total_tokens": 100,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(relevance_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = RelevanceNode(judge_model="gpt-4o-mini")
    result = node.run(claims=_claims(), question="What is the capital of France?")

    assert len(result.metrics) == 1
    assert result.metrics[0].error == "No verdicts returned"
    assert result.cost.input_tokens == 90
