from __future__ import annotations

from types import SimpleNamespace

import ng_graph.nodes.metrics.refalign as refalign_module
import numpy as np
import pytest
from ng_core.types import Chunk, Item, Refalign
from ng_graph.nodes.metrics.refalign import RefalignNode


def _chunk(text: str) -> Chunk:
    return Chunk(
        index=0,
        item=Item(text=text, tokens=float(len(text.split()))),
        char_start=0,
        char_end=len(text),
        sha256="x",
    )


class _FakeEmbedModel:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts, show_progress_bar=False):
        del show_progress_bar
        return np.asarray([self._vectors[t] for t in texts], dtype=float)


def test_run_returns_five_metrics_with_standard_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = {
        "Need strong python experience.": [1.0, 0.0],
        "Need kubernetes experience. Need strong python experience.": [1.0, 1.0],
        "Need strong python experience": [1.0, 0.0],
        "Need kubernetes experience.": [0.0, 1.0],
    }
    monkeypatch.setattr(refalign_module, "_get_embedding_model", lambda: _FakeEmbedModel(vectors))

    node = RefalignNode(judge_model="openai/gpt-4o-mini")
    result = node.run(
        output=Item(text="Need strong python experience.", tokens=4),
        reference=Item(
            text="Need kubernetes experience. Need strong python experience.",
            tokens=8,
        ),
        output_chunks=[_chunk("Need strong python experience")],
    )

    assert [m.name for m in result.metrics] == [
        "refalign_precision",
        "refalign_recall",
        "refalign_f1",
        "refalign_global_similarity",
        "refalign_score",
    ]
    assert all(m.score is not None for m in result.metrics)
    assert all(m.verdict in {"PASSED", "FAILED"} for m in result.metrics)

    by_name = {m.name: m for m in result.metrics}
    assert by_name["refalign_precision"].score == 1.0
    assert by_name["refalign_recall"].score == 0.5
    assert by_name["refalign_f1"].score == 0.6667
    assert by_name["refalign_global_similarity"].score == 0.7071
    assert by_name["refalign_score"].score == 1.0

    payload = by_name["refalign_score"].result[0]
    assert payload["counts"]["covered_reference_chunks"] == 1
    assert payload["counts"]["reference_chunks"] == 2
    assert len(payload["missed_reference_chunks"]) == 1
    assert payload["extra_output_chunks"] == []
    assert result.cost.cost == 0.0


def test_run_atomic_chunks_uses_llm_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = {
        "Need strong python experience.": [1.0, 0.0],
        "Need kubernetes experience and python experience.": [1.0, 1.0],
        "Need python experience.": [1.0, 0.0],
        "Need kubernetes experience.": [0.0, 1.0],
    }
    monkeypatch.setattr(refalign_module, "_get_embedding_model", lambda: _FakeEmbedModel(vectors))

    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                units = ["Need python experience."]
            else:
                units = ["Need kubernetes experience.", "Need python experience."]
            return {
                "parsed": SimpleNamespace(units=units),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                },
                "model": "openai/gpt-4o-mini",
            }

    fake_llm = FakeLLM()
    monkeypatch.setattr(refalign_module, "get_llm", lambda *_a, **_kw: fake_llm)

    node = RefalignNode(judge_model="openai/gpt-4o-mini")
    result = node.run(
        output=Item(text="Need strong python experience.", tokens=4),
        reference=Item(text="Need kubernetes experience and python experience.", tokens=7),
        output_chunks=[_chunk("Need strong python experience.")],
        refalign=Refalign(atomic_chunks=True),
    )

    by_name = {m.name: m for m in result.metrics}
    assert by_name["refalign_precision"].score == 1.0
    assert by_name["refalign_recall"].score == 0.5
    assert by_name["refalign_f1"].score == 0.6667
    assert result.cost.cost > 0.0
    assert result.cost.input_tokens == 60.0
    assert result.cost.output_tokens == 20.0


def test_run_skips_when_reference_missing() -> None:
    node = RefalignNode(judge_model="openai/gpt-4o-mini")
    result = node.run(output="Answer", reference=None)
    assert result.metrics == []
    assert result.cost.cost == 0.0
