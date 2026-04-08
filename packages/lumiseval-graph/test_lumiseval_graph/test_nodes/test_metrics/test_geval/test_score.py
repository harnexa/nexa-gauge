from types import SimpleNamespace

import pytest

from lumiseval_core.types import GevalStepsResolved, Item
import lumiseval_graph.nodes.metrics.geval.score as score_module
from lumiseval_graph.nodes.metrics.geval.score import GevalNode


class _FakeModelNumericCost:
    def calculate_cost(self, input_tokens, output_tokens):
        return (input_tokens + output_tokens) / 1000.0


class _FakeModelResponseCost:
    def calculate_cost(self, response):
        return 0.123


class _FakeGEvalNumeric:
    def __init__(self, **_kwargs):
        self.model = _FakeModelNumericCost()
        self.score = 0.9
        self.reason = "looks good"
        self.evaluation_cost = 0.0

    def measure(self, _test_case):
        self.model.calculate_cost(120, 30)
        self.evaluation_cost = 0.015


class _FakeGEvalResponse:
    def __init__(self, **_kwargs):
        self.model = _FakeModelResponseCost()
        self.score = 0.7
        self.reason = "acceptable"
        self.evaluation_cost = 0.0

    def measure(self, _test_case):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=20)
        )
        self.model.calculate_cost(response)
        self.evaluation_cost = 0.01


@pytest.fixture
def resolved_artifacts() -> list[GevalStepsResolved]:
    return [
        GevalStepsResolved(
            key="factuality",
            name="factuality",
            item_fields=["generation"],
            evaluation_steps=[Item(text="Check factual accuracy.", tokens=4)],
            steps_source="provided",
            signature=None,
        )
    ]


def test_run_aggregates_token_usage_from_calculate_cost_args(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    monkeypatch.setattr(score_module, "GEval", _FakeGEvalNumeric)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        generation=Item(text="Paris is the capital of France.", tokens=8),
        question=Item(text="What is the capital of France?", tokens=7),
        reference=Item(text="The capital of France is Paris.", tokens=7),
        context=Item(text="Paris is the French capital city.", tokens=7),
    )

    assert len(result.metrics) == 1
    assert result.cost is not None
    assert result.cost.input_tokens == 120
    assert result.cost.output_tokens == 30
    assert result.cost.cost == 0.015


def test_run_aggregates_token_usage_from_response_usage_payload(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    monkeypatch.setattr(score_module, "GEval", _FakeGEvalResponse)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        generation=Item(text="Answer", tokens=1),
        question=Item(text="Question", tokens=1),
        reference=Item(text="Reference", tokens=1),
        context=Item(text="Context", tokens=1),
    )

    assert result.cost is not None
    assert result.cost.input_tokens == 80
    assert result.cost.output_tokens == 20
    assert result.cost.cost == 0.01
