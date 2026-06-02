from __future__ import annotations

import hashlib

from ng_core.types import Chunk, ChunkArtifacts, CostEstimate, Inputs, Item, RefalignMetrics


def _chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        index=index,
        item=Item(text=text, tokens=float(len(text.split()))),
        char_start=0,
        char_end=len(text),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_node_refalign_uses_explicit_output_and_reference_refined_chunks(
    graph_module, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeRefalignNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            captured["judge_model"] = judge_model
            captured["llm_overrides"] = llm_overrides

        def run(self, *, output, reference, output_chunks, reference_chunks, refalign):
            captured["output"] = output.text
            captured["reference"] = reference.text
            captured["output_chunks"] = [c.item.text for c in output_chunks]
            captured["reference_chunks"] = [c.item.text for c in reference_chunks]
            captured["refalign_cfg"] = refalign
            return RefalignMetrics(
                metrics=[],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

        def estimate(self, inputs):
            del inputs
            return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)

        def get_model_usage(self):
            return {"primary": 0, "fallback": 0}

    monkeypatch.setattr(graph_module, "get_judge_model", lambda *_a, **_kw: "resolved-refalign")
    monkeypatch.setattr(graph_module, "RefalignNode", _FakeRefalignNode)

    output_chunks = ChunkArtifacts(
        chunks=[_chunk("Output chunk A")],
        cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
    )
    reference_chunks = ChunkArtifacts(
        chunks=[_chunk("Reference chunk A")],
        cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
    )
    inputs = Inputs(
        case_id="case-refalign-wiring",
        output=Item(text="Output full", tokens=2),
        reference=Item(text="Reference full", tokens=2),
        has_output=True,
        has_reference=True,
    )

    out = graph_module.node_refalign(
        {
            "inputs": inputs,
            "output_refined_chunks": output_chunks,
            "reference_refined_chunks": reference_chunks,
            "llm_overrides": None,
            "execution_mode": "run",
        }
    )

    assert captured["judge_model"] == "resolved-refalign"
    assert captured["output"] == "Output full"
    assert captured["reference"] == "Reference full"
    assert captured["output_chunks"] == ["Output chunk A"]
    assert captured["reference_chunks"] == ["Reference chunk A"]
    assert out["refalign_metrics"].cost.cost == 0.0


def test_node_reference_refiner_passes_through_when_refine_top_k_missing(graph_module) -> None:
    source_chunks = ChunkArtifacts(
        chunks=[_chunk("Reference chunk A", index=0), _chunk("Reference chunk B", index=1)],
        cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
    )
    inputs = Inputs(
        case_id="case-reference-refiner-pass",
        output=Item(text="Output", tokens=1),
        reference=Item(text="Reference", tokens=1),
        has_output=True,
        has_reference=True,
    )

    out = graph_module.node_reference_refiner(
        {
            "inputs": inputs,
            "reference_chunk": source_chunks,
            "refiner": "mmr",
            "refiner_top_k": 1,
            "execution_mode": "run",
        }
    )

    artifact = out["reference_refined_chunks"]
    assert artifact is not None
    assert [c.item.text for c in artifact.chunks] == ["Reference chunk A", "Reference chunk B"]
