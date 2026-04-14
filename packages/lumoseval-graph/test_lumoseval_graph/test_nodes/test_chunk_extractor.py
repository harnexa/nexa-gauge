# Debug commands:
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_chunk_extractor.py
# uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_chunk_extractor.py::test_run_returns_single_chunk_for_short_input
# uv run pytest -s -k "chunk_extractor" packages/lumos-graph/test_lumos_graph/test_nodes/test_chunk_extractor.py

from lumoseval_core.types import Item
from lumoseval_graph.nodes.chunk_extractor import ChunkExtractorNode


def test_run_returns_single_chunk_for_short_input() -> None:
    node = ChunkExtractorNode()
    item = Item(text="Paris is in France.", tokens=4)

    result = node.run(item)

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.index == 0
    assert chunk.item.text == item.text
    assert chunk.char_start == 0
    assert chunk.char_end == len(item.text)
    assert result.cost.cost == 0.0


def test_run_splits_long_input_with_small_chunk_size() -> None:
    node = ChunkExtractorNode(chunk_size=20)
    text = " ".join(["The Eiffel Tower is in Paris."] * 120)
    item = Item(text=text, tokens=300)

    result = node.run(item)

    assert len(result.chunks) >= 2
    assert [c.index for c in result.chunks] == list(range(len(result.chunks)))
    assert all(c.item.text.strip() for c in result.chunks)
    assert all(c.char_end >= c.char_start for c in result.chunks)
