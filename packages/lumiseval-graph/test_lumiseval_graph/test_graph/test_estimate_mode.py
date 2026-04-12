from __future__ import annotations

import pytest

from lumiseval_core.types import ChunkArtifacts, CostEstimate, Inputs, Item


def test_node_generation_claims_estimate_calls_estimate_without_chunks(
    graph_module, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides=None) -> str:
        captured["resolved_node_name"] = node_name
        return "resolved-claims-model"

    class _FakeClaimExtractorNode:
        def __init__(self, model: str, llm_overrides=None):
            captured["constructor_model"] = model
            captured["constructor_overrides"] = llm_overrides

        def run(self, _chunks):
            raise AssertionError("run() should not be called in estimate mode")

        def estimate(self, chunks) -> CostEstimate:
            captured["estimate_chunk_count"] = len(chunks)
            return CostEstimate(cost=0.123, input_tokens=0.0, output_tokens=0.0)

    monkeypatch.setattr(graph_module, "get_judge_model", _fake_get_judge_model)
    monkeypatch.setattr(graph_module.claim_extractor, "ClaimExtractorNode", _FakeClaimExtractorNode)

    llm_overrides = {"models": {"claims": "runtime-claims-model"}}
    state = {
        "execution_mode": "estimate",
        "inputs": Inputs(
            generation=Item(text="Paris is in France.", tokens=4),
            has_generation=True,
        ),
        "generation_chunk": ChunkArtifacts(
            chunks=[],
            cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
        ),
        "llm_overrides": llm_overrides,
    }

    out = graph_module.node_generation_claims(state)

    assert captured["resolved_node_name"] == "claims"
    assert captured["constructor_model"] == "resolved-claims-model"
    assert captured["constructor_overrides"] == llm_overrides
    assert captured["estimate_chunk_count"] == 0
    assert out["generation_claims"].claims == []
    assert out["generation_claims"].cost.cost == 0.123
    assert out["estimated_costs"]["claims"].cost == 0.123


def test_node_grounding_estimate_calls_estimate_without_claim_artifact(
    graph_module, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides=None) -> str:
        captured["resolved_node_name"] = node_name
        return "resolved-grounding-model"

    class _FakeGroundingNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            captured["constructor_model"] = judge_model
            captured["constructor_overrides"] = llm_overrides

        def run(self, claims, context, enable_grounding=True):
            raise AssertionError("run() should not be called in estimate mode")

        def estimate(self, claims, context) -> CostEstimate:
            captured["estimate_claim_count"] = len(claims)
            captured["estimate_context"] = context
            return CostEstimate(cost=0.456, input_tokens=0.0, output_tokens=0.0)

    monkeypatch.setattr(graph_module, "get_judge_model", _fake_get_judge_model)
    monkeypatch.setattr(graph_module, "GroundingNode", _FakeGroundingNode)

    llm_overrides = {"models": {"grounding": "runtime-grounding-model"}}
    state = {
        "execution_mode": "estimate",
        "inputs": Inputs(
            generation=Item(text="Paris is in France.", tokens=4),
            context=None,
            has_generation=True,
            has_context=False,
        ),
        "generation_dedup_claims": None,
        "llm_overrides": llm_overrides,
    }

    out = graph_module.node_grounding(state)

    assert "resolved_node_name" not in captured
    assert "constructor_model" not in captured
    assert out["grounding_metrics"] is None
    assert "estimated_costs" not in out


def test_node_report_sets_cost_estimate_in_estimate_mode(graph_module) -> None:
    state = {
        "execution_mode": "estimate",
        "job_id": "job-estimate",
        "grounding_metrics": [],
        "relevance_metrics": [],
        "redteam_metrics": [],
        "geval_metrics": [],
        "reference_metrics": [],
        "cost_actual_usd": 0.0,
        "estimated_costs": {
            "claims": CostEstimate(cost=0.1, input_tokens=10.0, output_tokens=2.0),
            "grounding": CostEstimate(cost=0.2, input_tokens=None, output_tokens=3.0),
        },
    }

    out = graph_module.node_report(state)
    assert out["report"] == []
    assert out["cost_estimate"].cost == pytest.approx(0.3)
    assert out["cost_estimate"].input_tokens == 10.0
    assert out["cost_estimate"].output_tokens == 5.0
