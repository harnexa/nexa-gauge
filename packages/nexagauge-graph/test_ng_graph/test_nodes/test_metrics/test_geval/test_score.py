import math

import ng_graph.nodes.metrics.geval.score as score_module
import pytest
from ng_core.types import GevalStepsResolved, Item
from ng_graph.nodes.metrics.geval.score import GevalNode, _GevalScoreResponse


class FakeLLM:
    def __init__(
        self,
        *,
        parsed=None,
        parsing_error=None,
        usage=None,
        logprobs=None,
        model="gpt-4o-mini",
    ):
        self._parsed = parsed
        self._parsing_error = parsing_error
        self._usage = usage or {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }
        self._logprobs = logprobs
        self._model = model
        self.captured_messages: list = []

    def invoke_with_logprobs(self, messages, **_kwargs):
        self.captured_messages.append(messages)
        return {
            "parsed": self._parsed,
            "parsing_error": self._parsing_error,
            "usage": self._usage,
            "model": self._model,
            "logprobs": self._logprobs,
        }


def _three_metric_fields() -> list[str]:
    return ["output"]


@pytest.fixture
def resolved_artifacts() -> list[GevalStepsResolved]:
    return [
        GevalStepsResolved(
            key="factuality",
            name="factuality",
            item_fields=_three_metric_fields(),
            evaluation_steps=[
                Item(text="Check factual accuracy.", tokens=4),
                Item(text="Penalize unsupported claims.", tokens=4),
                Item(text="Flag hallucinations.", tokens=3),
            ],
            steps_source="provided",
            signature=None,
        )
    ]


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, fake: FakeLLM) -> None:
    monkeypatch.setattr(score_module, "get_llm", lambda *_a, **_kw: fake)


def test_successful_scoring_with_logprobs(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    logprobs = [
        {
            "token": "7",
            "logprob": math.log(0.5),
            "top_logprobs": [
                {"token": "6", "logprob": math.log(0.3)},
                {"token": "7", "logprob": math.log(0.5)},
                {"token": "8", "logprob": math.log(0.2)},
            ],
        }
    ]
    fake = FakeLLM(
        parsed=_GevalScoreResponse(score=7, reason="Mostly accurate but misses one claim."),
        logprobs=logprobs,
    )
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris is the capital of France.", tokens=8),
        input=Item(text="What is France's capital?", tokens=6),
        reference=Item(text="The capital of France is Paris.", tokens=7),
        context=Item(text="France is in Europe.", tokens=5),
    )

    assert len(result.metrics) == 1
    m = result.metrics[0]
    # weighted ≈ (6·0.3 + 7·0.5 + 8·0.2)/1.0 = 6.9; normalized = (6.9-1)/9 ≈ 0.655...
    assert m.score is not None
    assert 0.0 < m.score < 1.0
    assert abs(m.score - (6.9 - 1) / 9) < 1e-9
    assert m.verdict == "PASSED"
    assert result.cost is not None
    assert result.cost.input_tokens == 120
    assert result.cost.output_tokens == 30


def test_successful_scoring_without_logprobs(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    fake = FakeLLM(
        parsed=_GevalScoreResponse(score=7, reason="ok"),
        logprobs=None,
    )
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )

    m = result.metrics[0]
    assert m.score is not None
    assert abs(m.score - (7 - 1) / 9) < 1e-9
    assert m.verdict == "PASSED"


def test_missing_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    # Artifact requires "output"; pass empty Item.
    fake = FakeLLM(parsed=_GevalScoreResponse(score=5, reason="noop"))
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="   ", tokens=0),
        input=None,
        reference=None,
        context=None,
    )

    m = result.metrics[0]
    assert m.score is None
    assert m.error is not None
    assert "output" in m.error
    assert m.verdict is None
    assert fake.captured_messages == []


def test_parse_error_yields_metric_error(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    fake = FakeLLM(parsed=None, parsing_error=ValueError("bad json"))
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )

    m = result.metrics[0]
    assert m.score is None
    assert m.error is not None
    assert "bad json" in m.error
    assert m.verdict is None


def test_pass_threshold_boundary(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    # Raw score 6, no logprobs → normalized = (6-1)/9 ≈ 0.5556 → not passed at 0.6 threshold.
    fake = FakeLLM(parsed=_GevalScoreResponse(score=6, reason="borderline"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )
    entry = result.metrics[0].result[0]
    assert entry["passed"] is False
    assert result.metrics[0].verdict == "FAILED"

    # Raw score 5 → normalized ≈ 0.444 → not passed.
    fake2 = FakeLLM(parsed=_GevalScoreResponse(score=5, reason="weak"), logprobs=None)
    _install_fake_llm(monkeypatch, fake2)

    node2 = GevalNode(judge_model="gpt-4o-mini")
    result2 = node2.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )
    entry2 = result2.metrics[0].result[0]
    assert entry2["passed"] is False
    assert result2.metrics[0].verdict == "FAILED"


def test_prompt_contains_evaluation_steps(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    fake = FakeLLM(parsed=_GevalScoreResponse(score=7, reason="ok"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )

    user_content = fake.captured_messages[0][1]["content"]
    for step in resolved_artifacts[0].evaluation_steps:
        assert step.text.strip() in user_content
    assert "Output" in user_content


def test_judge_model_prefix_accepted(
    monkeypatch: pytest.MonkeyPatch,
    resolved_artifacts: list[GevalStepsResolved],
) -> None:
    """Passing ``openai/gpt-4o-mini`` to GevalNode must not raise — the whole
    reason we dropped DeepEval."""
    fake = FakeLLM(parsed=_GevalScoreResponse(score=8, reason="fine"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="openai/gpt-4o-mini")
    result = node.run(
        resolved_artifacts=resolved_artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )
    assert result.metrics[0].score is not None


def test_multi_metric_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(parsed=_GevalScoreResponse(score=7, reason="ok"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    artifacts = [
        GevalStepsResolved(
            key=name,
            name=name,
            item_fields=["output"],
            evaluation_steps=[
                Item(text="Step A.", tokens=2),
                Item(text="Step B.", tokens=2),
                Item(text="Step C.", tokens=2),
            ],
            steps_source="provided",
            signature=None,
        )
        for name in ("alpha", "beta", "gamma")
    ]

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=artifacts,
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )

    assert [m.name for m in result.metrics] == ["alpha", "beta", "gamma"]
    assert result.cost is not None
    # 3 metrics × 120 input tokens
    assert result.cost.input_tokens == 3 * 120
    assert result.cost.output_tokens == 3 * 30
