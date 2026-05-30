import math
from types import SimpleNamespace

import ng_graph.nodes.metrics.geval.score as score_module
import pytest
from ng_core.constants import USE_LOGPROBS_FORSCORE_WEIGHTING
from ng_core.types import GevalStepsResolved, Item, ScoringMode
from ng_graph.nodes.metrics.geval.score import GevalNode


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


def _resolved_metric(*, name: str = "factuality") -> GevalStepsResolved:
    """Resolved-steps fixture; scoring knobs are passed via `node.run(...)` kwargs."""
    return GevalStepsResolved(
        key=name,
        name=name,
        item_fields=["output"],
        evaluation_steps=[
            Item(text="Check factual accuracy.", tokens=4),
            Item(text="Penalize unsupported claims.", tokens=4),
            Item(text="Flag hallucinations.", tokens=3),
        ],
        steps_source="provided",
        signature=None,
    )


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, fake: FakeLLM) -> None:
    monkeypatch.setattr(score_module, "get_llm", lambda *_a, **_kw: fake)


def test_successful_scale_scoring_with_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    logprobs = [
        {
            "token": "4",
            "logprob": math.log(0.6),
            "top_logprobs": [
                {"token": "3", "logprob": math.log(0.25)},
                {"token": "4", "logprob": math.log(0.6)},
                {"token": "5", "logprob": math.log(0.15)},
            ],
        }
    ]
    fake = FakeLLM(
        parsed=SimpleNamespace(scores=[4], reasoning="Mostly accurate but misses one claim."),
        logprobs=logprobs,
    )
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris is the capital of France.", tokens=8),
        input=Item(text="What is France's capital?", tokens=6),
        reference=Item(text="The capital of France is Paris.", tokens=7),
        context=Item(text="France is in Europe.", tokens=5),
        scoring_mode=ScoringMode.SCALE_1_5,
        include_reasoning=True,
    )

    assert len(result.metrics) == 1
    m = result.metrics[0]
    assert m.score is not None
    # Metric score always reflects normalized raw score average.
    assert abs(m.score - ((4.0 - 1.0) / 4.0)) < 1e-9
    # Logprobs only affect pass/fail gate when weighting is enabled.
    if USE_LOGPROBS_FORSCORE_WEIGHTING:
        assert m.verdict == "PASSED"  # weighted remains above 0.6
    else:
        assert m.verdict == "PASSED"
    assert m.verdict == "PASSED"
    assert result.cost is not None
    assert result.cost.input_tokens == 120
    assert result.cost.output_tokens == 30


def test_successful_scale_scoring_without_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(parsed=SimpleNamespace(scores=[4], reasoning="ok"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
        scoring_mode=ScoringMode.SCALE_1_5,
        include_reasoning=True,
    )

    m = result.metrics[0]
    assert m.score is not None
    assert abs(m.score - ((4.0 - 1.0) / 4.0)) < 1e-9
    assert m.verdict == "PASSED"


def test_binary_yes_no_scoring_without_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(
        parsed=SimpleNamespace(scores=[1], reasoning="Criteria satisfied."), logprobs=None
    )
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
        scoring_mode=ScoringMode.BINARY_YES_NO,
        include_reasoning=True,
    )
    m = result.metrics[0]
    assert m.score == 1.0
    assert m.verdict == "PASSED"
    assert m.result is not None
    assert m.result[0]["passed"] is True


def test_binary_yes_no_scoring_with_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    logprobs = [
        {
            "token": "1",
            "logprob": math.log(0.8),
            "top_logprobs": [
                {"token": "0", "logprob": math.log(0.35)},
                {"token": "1", "logprob": math.log(0.65)},
            ],
        }
    ]
    fake = FakeLLM(parsed=SimpleNamespace(scores=[1], reasoning="yes"), logprobs=logprobs)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
        scoring_mode=ScoringMode.BINARY_YES_NO,
        include_reasoning=True,
    )
    m = result.metrics[0]
    assert m.score is not None
    assert abs(m.score - 1.0) < 1e-9
    assert m.verdict == "PASSED"


def test_reasoning_disabled_uses_score_only_prompt_and_empty_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeLLM(parsed=SimpleNamespace(scores=[5]), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
        scoring_mode=ScoringMode.SCALE_1_5,
        include_reasoning=False,
    )
    m = result.metrics[0]
    assert m.result is not None
    # With reasoning off, the result entry omits the reasoning/tokens keys entirely.
    assert "reasoning" not in m.result[0]
    assert "tokens" not in m.result[0]
    # Output contract should not request reasoning when include_reasoning=False.
    output_contract = fake.captured_messages[0][1]["content"]
    assert "`reasoning`" not in output_contract


def test_missing_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(parsed=SimpleNamespace(scores=[5], reasoning="noop"))
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
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


def test_parse_error_yields_metric_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(parsed=None, parsing_error=ValueError("bad json"))
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
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


def test_pass_threshold_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    # Raw score 3 on [1,5] -> normalized 0.5, below 0.6 threshold.
    fake = FakeLLM(parsed=SimpleNamespace(scores=[3], reasoning="borderline"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
        scoring_mode=ScoringMode.SCALE_1_5,
        include_reasoning=True,
    )
    entry = result.metrics[0].result[0]
    assert entry["passed"] is False
    assert result.metrics[0].verdict == "FAILED"


def test_prompt_contains_evaluation_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    metric = _resolved_metric()
    fake = FakeLLM(parsed=SimpleNamespace(scores=[4], reasoning="ok"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="gpt-4o-mini")
    node.run(
        resolved_artifacts=[metric],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )

    user_content = fake.captured_messages[0][2]["content"]
    for step in metric.evaluation_steps:
        assert step.text.strip() in user_content
    assert "Output" in user_content


def test_judge_model_prefix_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(parsed=SimpleNamespace(scores=[5], reasoning="fine"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    node = GevalNode(judge_model="openai/gpt-4o-mini")
    result = node.run(
        resolved_artifacts=[_resolved_metric()],
        output=Item(text="Paris.", tokens=2),
        input=None,
        reference=None,
        context=None,
    )
    assert result.metrics[0].score is not None


def test_multi_metric_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(parsed=SimpleNamespace(scores=[4], reasoning="ok"), logprobs=None)
    _install_fake_llm(monkeypatch, fake)

    artifacts = [_resolved_metric(name=name) for name in ("alpha", "beta", "gamma")]

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
    assert result.cost.input_tokens == 3 * 120
    assert result.cost.output_tokens == 3 * 30
