# Debug commands:
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_reference.py
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_reference.py::test_run_returns_reference_metrics_for_item_inputs
# uv run pytest -s -k "reference" packages/lumos-graph/test_lumos_graph/test_nodes/test_reference.py

from lumoseval_core.types import Item, MetricCategory
from lumoseval_graph.nodes.metrics.reference import ReferenceNode


def test_run_returns_reference_metrics_for_item_inputs() -> None:
    node = ReferenceNode()

    result = node.run(
        generation=Item(text="Paris is the capital of France.", tokens=7),
        reference=Item(text="Paris is the capital of France.", tokens=7),
        enable_generation_metrics=True,
    )

    assert len(result.metrics) == 5
    assert [m.name for m in result.metrics] == ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]
    assert all(m.category == MetricCategory.ANSWER for m in result.metrics)
    assert all(m.score is not None and 0.0 <= m.score <= 1.0 for m in result.metrics)

    assert result.cost.cost == 0.0
    assert result.cost.input_tokens is None
    assert result.cost.output_tokens is None


def test_run_accepts_string_inputs() -> None:
    node = ReferenceNode()

    result = node.run(
        generation="The sky is blue.",
        reference="The sky is blue.",
        enable_generation_metrics=True,
    )

    assert len(result.metrics) == 5
    assert all(m.score is not None for m in result.metrics)


def test_run_skips_when_disabled_or_reference_missing() -> None:
    node = ReferenceNode()

    disabled = node.run(generation="answer", reference="reference", enable_generation_metrics=False)
    assert disabled.metrics == []
    assert disabled.cost.cost == 0.0

    no_reference = node.run(generation="answer", reference=None, enable_generation_metrics=True)
    assert no_reference.metrics == []
    assert no_reference.cost.cost == 0.0

    blank_reference = node.run(generation="answer", reference="   ", enable_generation_metrics=True)
    assert blank_reference.metrics == []
    assert blank_reference.cost.cost == 0.0


def test_estimate_returns_zero_cost() -> None:
    node = ReferenceNode()

    cost = node.estimate(input_tokens=100.0, output_tokens=50.0)
    assert cost.cost == 0.0
    assert cost.input_tokens is None
    assert cost.output_tokens is None
