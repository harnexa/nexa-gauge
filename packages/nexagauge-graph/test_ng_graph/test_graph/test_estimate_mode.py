from __future__ import annotations

import pytest
from ng_core.types import ChunkArtifacts, CostEstimate, Inputs, Item
from conftest import make_fake_get_judge_model


def test_node_generation_claims_estimate_calls_estimate_without_chunks(
    graph_module, monkeypatch
) -> None:
    """Verify claims node estimate mode calls estimate() with zero chunks.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_estimate_mode.py::test_node_generation_claims_estimate_calls_estimate_without_chunks
    """
    captured: dict[str, object] = {}

    class _FakeClaimExtractorNode:
        def __init__(self, model: str, llm_overrides=None):
            captured["constructor_model"] = model
            captured["constructor_overrides"] = llm_overrides

        def run(self, _chunks):
            raise AssertionError("run() should not be called in estimate mode")

        def estimate(self, chunks) -> CostEstimate:
            captured["estimate_chunk_count"] = len(chunks)
            return CostEstimate(cost=0.123, input_tokens=0.0, output_tokens=0.0)

    monkeypatch.setattr(
        graph_module, "get_judge_model", make_fake_get_judge_model(captured)
    )
    monkeypatch.setattr(graph_module.claim_extractor, "ClaimExtractorNode", _FakeClaimExtractorNode)

    llm_overrides = {"models": {"claims": "runtime-claims-model"}}
    state = {
        "execution_mode": "estimate",
        "inputs": Inputs(
            case_id="case-estimate-claims",
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
    """Verify grounding node is skipped in estimate mode when required context is missing.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_estimate_mode.py::test_node_grounding_estimate_calls_estimate_without_claim_artifact
    """
    captured: dict[str, object] = {}

    class _FakeGroundingNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            captured["constructor_model"] = judge_model
            captured["constructor_overrides"] = llm_overrides

        def run(self, claims, context, enable_grounding=True):
            raise AssertionError("run() should not be called in estimate mode")

        def estimate(self, context) -> CostEstimate:
            captured["estimate_context"] = context
            return CostEstimate(cost=0.456, input_tokens=0.0, output_tokens=0.0)

    monkeypatch.setattr(
        graph_module, "get_judge_model", make_fake_get_judge_model(captured)
    )
    monkeypatch.setattr(graph_module, "GroundingNode", _FakeGroundingNode)

    llm_overrides = {"models": {"grounding": "runtime-grounding-model"}}
    state = {
        "execution_mode": "estimate",
        "inputs": Inputs(
            case_id="case-estimate-grounding",
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


def test_node_relevance_estimate_calls_estimate_with_claims_and_question(
    graph_module, monkeypatch
) -> None:
    """Verify relevance node estimate mode resolves model and records estimated cost.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_estimate_mode.py::test_node_relevance_estimate_calls_estimate_with_claims_and_question
    """
    captured: dict[str, object] = {}

    class _FakeRelevanceNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            captured["constructor_model"] = judge_model
            captured["constructor_overrides"] = llm_overrides

        def run(self, claims, question):
            raise AssertionError("run() should not be called in estimate mode")

        def estimate(self, question) -> CostEstimate:
            captured["estimate_question"] = question
            return CostEstimate(cost=0.234, input_tokens=0.0, output_tokens=0.0)

    monkeypatch.setattr(
        graph_module, "get_judge_model", make_fake_get_judge_model(captured)
    )
    monkeypatch.setattr(graph_module, "RelevanceNode", _FakeRelevanceNode)

    llm_overrides = {"models": {"relevance": "runtime-relevance-model"}}
    state = {
        "execution_mode": "estimate",
        "inputs": Inputs(
            case_id="case-estimate-relevance",
            generation=Item(text="Paris is in France.", tokens=4),
            question=Item(text="Where is Paris?", tokens=4),
            has_generation=True,
            has_question=True,
        ),
        "generation_dedup_claims": None,
        "llm_overrides": llm_overrides,
    }

    out = graph_module.node_relevance(state)

    assert captured["resolved_node_name"] == "relevance"
    assert captured["constructor_model"] == "resolved-relevance-model"
    assert captured["constructor_overrides"] == llm_overrides
    assert captured["estimate_question"] == state["inputs"].question
    assert out["relevance_metrics"].metrics == []
    assert out["relevance_metrics"].cost.cost == 0.234
    assert out["estimated_costs"]["relevance"].cost == 0.234


def test_node_report_sets_cost_estimate_in_estimate_mode(graph_module) -> None:
    """Verify node_report computes aggregate cost_estimate in estimate mode.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_estimate_mode.py::test_node_report_sets_cost_estimate_in_estimate_mode
    """
    state = {
        "execution_mode": "estimate",
        "target_node": "eval",
        "job_id": "job-estimate",
        "grounding_metrics": None,
        "relevance_metrics": None,
        "redteam_metrics": None,
        "geval_metrics": None,
        "reference_metrics": None,
        "cost_actual_usd": 0.0,
        "estimated_costs": {
            "claims": CostEstimate(cost=0.1, input_tokens=10.0, output_tokens=2.0),
            "grounding": CostEstimate(cost=0.2, input_tokens=None, output_tokens=3.0),
        },
    }

    out = graph_module.node_report(state)
    report = out["report"]
    assert isinstance(report, dict)
    assert report["target_node"] == "eval"
    # Metric sections are absent when their state key is None
    assert "grounding" not in report
    assert "relevance" not in report
    assert out["cost_estimate"].cost == pytest.approx(0.3)
    assert out["cost_estimate"].input_tokens == 10.0
    assert out["cost_estimate"].output_tokens == 5.0
