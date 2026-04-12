# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_claim_extractor.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_claim_extractor.py::test_run_builds_claim_artifacts_with_mocked_llm
# uv run pytest -s -k "claim_extractor" packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_claim_extractor.py

import hashlib
from types import SimpleNamespace

import pytest

from lumiseval_core.types import Chunk, Item
from lumiseval_graph.nodes import claim_extractor as claim_module
from lumiseval_graph.nodes.claim_extractor import ClaimExtractorNode


def _make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        index=index,
        item=Item(text=text, tokens=float(len(text.split()))),
        char_start=0,
        char_end=len(text),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_run_builds_claim_artifacts_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    claims=["The Eiffel Tower is located in Paris."],
                    confidences=[0.93],
                ),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(claim_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = ClaimExtractorNode(model="gpt-4o-mini")
    chunks = [
        _make_chunk(0, "The Eiffel Tower stands in Paris, France."),
        _make_chunk(1, "It was completed in 1889."),
    ]

    result = node.run(chunks)

    assert len(result.claims) == 2
    assert result.claims[0].item.text == "The Eiffel Tower is located in Paris."
    assert result.claims[0].source_chunk_index == 0
    assert result.claims[1].source_chunk_index == 1

    assert result.cost.input_tokens == 200  # 100 per chunk * 2 chunks
    assert result.cost.output_tokens == 50   # 25 per chunk * 2 chunks
    assert result.cost.cost > 0


def test_run_raises_on_parsing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": None,
                "parsing_error": ValueError("bad parse"),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(claim_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = ClaimExtractorNode(model="gpt-4o-mini")
    chunks = [_make_chunk(0, "Short chunk")]

    with pytest.raises(ValueError, match="bad parse"):
        node.run(chunks)
