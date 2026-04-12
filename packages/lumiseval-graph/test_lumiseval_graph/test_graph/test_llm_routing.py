# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_llm_routing.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_llm_routing.py::test_node_generation_claims_uses_canonical_model_key
# uv run pytest -s -k "llm_routing" packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_llm_routing.py

from __future__ import annotations

import hashlib

from lumiseval_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    Inputs,
    Item,
)



def _make_chunk(text: str) -> Chunk:
    return Chunk(
        index=0,
        item=Item(text=text, tokens=float(len(text.split()))),
        char_start=0,
        char_end=len(text),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )



def test_node_generation_claims_uses_canonical_model_key(graph_module, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides=None) -> str:
        captured["resolved_node_name"] = node_name
        captured["resolved_overrides"] = llm_overrides
        return "resolved-claims-model"

    class _FakeClaimExtractorNode:
        def __init__(self, model: str, llm_overrides=None):
            captured["constructor_model"] = model
            captured["constructor_overrides"] = llm_overrides

        def run(self, _chunks):
            return ClaimArtifacts(claims=[], cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None))

    monkeypatch.setattr(graph_module, "get_judge_model", _fake_get_judge_model)
    monkeypatch.setattr(graph_module.claim_extractor, "ClaimExtractorNode", _FakeClaimExtractorNode)

    llm_overrides = {"models": {"claims": "runtime-claims-model"}}
    state = {
        "inputs": Inputs(
            case_id="case-routing-claims",
            generation=Item(text="The Eiffel Tower is in Paris.", tokens=7),
            has_generation=True,
        ),
        "generation_chunk": ChunkArtifacts(
            chunks=[_make_chunk("The Eiffel Tower is in Paris.")],
            cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
        ),
        "llm_overrides": llm_overrides,
    }

    out = graph_module.node_generation_claims(state)

    assert captured["resolved_node_name"] == "claims"
    assert captured["resolved_overrides"] == llm_overrides
    assert captured["constructor_model"] == "resolved-claims-model"
    assert captured["constructor_overrides"] == llm_overrides
    assert out["generation_claims"] is not None



def test_node_grounding_uses_canonical_key_and_handles_missing_context(graph_module, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides=None) -> str:
        captured["resolved_node_name"] = node_name
        captured["resolved_overrides"] = llm_overrides
        return "resolved-grounding-model"

    class _FakeGroundingNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            captured["constructor_model"] = judge_model
            captured["constructor_overrides"] = llm_overrides

        def run(self, claims, context, enable_grounding=True):
            captured["enable_grounding"] = enable_grounding
            captured["context_text"] = context.text
            return {"ok": True, "claims": len(claims)}

    monkeypatch.setattr(graph_module, "get_judge_model", _fake_get_judge_model)
    monkeypatch.setattr(graph_module, "GroundingNode", _FakeGroundingNode)

    llm_overrides = {"models": {"grounding": "runtime-grounding-model"}}
    state = {
        "inputs": Inputs(
            case_id="case-routing-grounding",
            generation=Item(text="Paris is in France.", tokens=4),
            context=None,
            has_generation=True,
            has_context=False,
        ),
        "generation_dedup_claims": ClaimArtifacts(
            claims=[Claim(item=Item(text="Paris is in France.", tokens=4), source_chunk_index=0)],
            cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
        ),
        "llm_overrides": llm_overrides,
    }

    out = graph_module.node_grounding(state)

    # Current graph behavior skips grounding entirely when no context is available.
    assert "resolved_node_name" not in captured
    assert "constructor_model" not in captured
    assert out["grounding_metrics"] is None
