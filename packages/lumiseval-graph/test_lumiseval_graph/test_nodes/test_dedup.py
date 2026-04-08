# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_dedup.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_dedup.py::test_run_returns_dedup_artifacts_from_mocked_deduplicate
# uv run pytest -s -k "dedup" packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_dedup.py

import pytest

from lumiseval_core.types import Item
from lumiseval_graph.nodes import dedup as dedup_module
from lumiseval_graph.nodes.dedup import DedupNode


def test_run_returns_dedup_artifacts_from_mocked_deduplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        Item(text="Paris is the capital of France.", tokens=7, confidence=0.95),
        Item(text="France's capital is Paris.", tokens=6, confidence=0.90),
        Item(text="Tokyo is the capital of Japan.", tokens=7, confidence=0.93),
    ]

    def fake_deduplicate(_items):
        return [items[0], items[2]], {1: 0}

    monkeypatch.setattr(dedup_module, "deduplicate", fake_deduplicate)

    node = DedupNode()
    result = node.run(items)

    assert len(result.items) == 2
    assert result.items[0].text == "Paris is the capital of France."
    assert result.items[1].text == "Tokyo is the capital of Japan."
    assert result.dropped == 1
    assert result.dedup_map == {1: 0}
    assert result.cost.input_tokens == 0.0
    assert result.cost.output_tokens == 0.0
    assert result.cost.cost == 0.0


def test_estimate_returns_zero_cost() -> None:
    node = DedupNode()
    estimate = node.estimate(123.0, 456.0)

    assert estimate.input_tokens == 0.0
    assert estimate.output_tokens == 0.0
    assert estimate.cost == 0.0
