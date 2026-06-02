# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_refmatch.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_refmatch.py::test_run_returns_refmatch_metrics_for_item_inputs
# uv run pytest -s -k "refmatch" packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_refmatch.py

from ng_core.types import Item, MetricCategory
from ng_graph.nodes.metrics import refmatch as refmatch_module
from ng_graph.nodes.metrics.refmatch import RefmatchNode


def test_run_returns_refmatch_metrics_for_item_inputs() -> None:
    node = RefmatchNode()

    result = node.run(
        output=Item(text="Paris is the capital of France.", tokens=7),
        reference=Item(text="Paris is the capital of France.", tokens=7),
        enable_output_metrics=True,
    )

    assert len(result.metrics) == 5
    assert [m.name for m in result.metrics] == ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]
    assert all(m.category == MetricCategory.ANSWER for m in result.metrics)
    assert all(m.score is not None and 0.0 <= m.score <= 1.0 for m in result.metrics)
    assert all(m.verdict == "PASSED" for m in result.metrics)

    assert result.cost.cost == 0.0
    assert result.cost.input_tokens is None
    assert result.cost.output_tokens is None


def test_run_accepts_string_inputs() -> None:
    node = RefmatchNode()

    result = node.run(
        output="The sky is blue.",
        reference="The sky is blue.",
        enable_output_metrics=True,
    )

    assert len(result.metrics) == 5
    assert all(m.score is not None for m in result.metrics)
    assert all(m.verdict in {"PASSED", "FAILED"} for m in result.metrics)


def test_run_skips_when_disabled_or_reference_missing() -> None:
    node = RefmatchNode()

    disabled = node.run(output="answer", reference="reference", enable_output_metrics=False)
    assert disabled.metrics == []
    assert disabled.cost.cost == 0.0

    no_reference = node.run(output="answer", reference=None, enable_output_metrics=True)
    assert no_reference.metrics == []
    assert no_reference.cost.cost == 0.0

    blank_reference = node.run(output="answer", reference="   ", enable_output_metrics=True)
    assert blank_reference.metrics == []
    assert blank_reference.cost.cost == 0.0


def test_estimate_returns_zero_cost() -> None:
    node = RefmatchNode()

    cost = node.estimate(input_tokens=100.0, output_tokens=50.0)
    assert cost.cost == 0.0
    assert cost.input_tokens is None
    assert cost.output_tokens is None


def test_run_meteor_falls_back_when_wordnet_path_errors(monkeypatch) -> None:
    node = RefmatchNode()

    def _fake_meteor_score(references, hypothesis, **kwargs):
        del references, hypothesis
        if "wordnet" not in kwargs:
            raise AttributeError("'NoneType' object has no attribute 'close'")
        assert isinstance(kwargs["wordnet"], refmatch_module._NoWordNet)
        return 0.1234

    monkeypatch.setattr(refmatch_module, "meteor_score", _fake_meteor_score)

    result = node.run(
        output="Paris is the capital of France.",
        reference="Paris is the capital of France.",
        enable_output_metrics=True,
    )

    meteor = next(m for m in result.metrics if m.name == "meteor")
    assert meteor.score == 0.1234
    assert meteor.verdict == "FAILED"
