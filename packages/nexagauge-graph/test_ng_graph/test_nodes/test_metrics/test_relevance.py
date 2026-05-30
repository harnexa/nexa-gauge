# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_relevance.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_relevance.py::test_run_returns_metric_and_cost_with_mocked_llm
# uv run pytest -s -k "relevance" packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_relevance.py

from types import SimpleNamespace

import ng_graph.nodes.metrics.relevance as relevance_module
import pytest
from ng_core.types import Claim, Item, ScoringMode
from ng_graph.nodes.metrics.relevance import RelevanceNode


def _claims() -> list[Claim]:
    return [
        Claim(item=Item(text="Paris is the capital of France.", tokens=8), source_chunk_index=0),
        Claim(item=Item(text="Transformers use self-attention.", tokens=6), source_chunk_index=1),
    ]


def test_run_returns_metric_and_cost_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages = []

    class FakeLLM:
        def invoke(self, messages):
            captured_messages.extend(messages)
            return {
                "parsed": SimpleNamespace(scores=[1, 0]),
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
    result = node.run(claims=_claims(), input=Item(text="What is the capital of France?", tokens=8))

    assert len(result.metrics) == 1
    metric = result.metrics[0]
    assert metric.name == "answer_relevancy"
    assert metric.score == 0.5
    assert metric.verdict == "FAILED"

    assert result.cost.input_tokens == 110
    assert result.cost.output_tokens == 20
    assert result.cost.cost > 0

    system_prompt = captured_messages[0]["content"]
    user_prompt = captured_messages[2]["content"]
    assert "Do not judge factual correctness" in system_prompt
    assert "supported by evidence" in system_prompt
    assert "## Input:" in user_prompt
    assert "## Answer Statements" in user_prompt


def test_run_skips_when_disabled_or_no_question() -> None:
    node = RelevanceNode(judge_model="gpt-4o-mini")

    disabled = node.run(
        claims=_claims(), input="What is the capital of France?", enable_relevance=False
    )
    assert disabled.metrics == []
    assert disabled.cost.cost == 0.0

    no_question = node.run(claims=_claims(), input="")
    assert no_question.metrics == []
    assert no_question.cost.cost == 0.0


def test_run_handles_no_verdicts_as_error_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(scores=[]),
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
    result = node.run(claims=_claims(), input="What is the capital of France?")

    assert len(result.metrics) == 1
    assert result.metrics[0].error == "No scores returned"
    assert result.metrics[0].verdict is None
    assert result.cost.input_tokens == 90


def test_run_scale_mode_normalizes_integer_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list = []

    class FakeLLM:
        def invoke(self, messages):
            captured.append(messages)
            return {
                "parsed": SimpleNamespace(scores=[5, 3]),
                "parsing_error": None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(relevance_module, "get_llm", lambda *_a, **_kw: FakeLLM())
    node = RelevanceNode(judge_model="gpt-4o-mini")
    result = node.run(
        claims=_claims(),
        input="What is the capital of France?",
        scoring_mode=ScoringMode.SCALE_1_5,
    )

    # (5-1)/4 = 1.0; (3-1)/4 = 0.5; mean = 0.75.
    assert result.metrics[0].score == 0.75
    output_contract = captured[0][1]["content"]
    assert "list of integers" in output_contract
    assert "scale_1_5" in output_contract


def test_run_with_reasoning_surfaces_batch_rationale(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    scores=[1, 1],
                    reasoning="Both statements address the user's input.",
                ),
                "parsing_error": None,
                "usage": {"prompt_tokens": 90, "completion_tokens": 25, "total_tokens": 115},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(relevance_module, "get_llm", lambda *_a, **_kw: FakeLLM())
    node = RelevanceNode(judge_model="gpt-4o-mini")
    result = node.run(
        claims=_claims(),
        input="What is the capital of France?",
        include_reasoning=True,
    )

    metric = result.metrics[0]
    assert metric.score == 1.0
    assert isinstance(metric.result, list) and len(metric.result) > len(_claims())
    assert metric.result[-1] == {"reasoning": "Both statements address the user's input."}
