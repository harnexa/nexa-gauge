from __future__ import annotations

from ng_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    GroundingMetrics,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    RelevanceMetrics,
)
from ng_graph.nodes import report


def _base_inputs() -> Inputs:
    return Inputs(
        case_id="case-1",
        output=Item(text="Generated answer.", tokens=5.0),
        input=Item(text="What is X?", tokens=3.0),
        context=Item(text="X is a thing.", tokens=4.0),
        reference=Item(text="X", tokens=1.0),
        has_output=True,
        has_input=True,
        has_context=True,
        has_reference=True,
    )


def _chunks() -> ChunkArtifacts:
    return ChunkArtifacts(
        chunks=[
            Chunk(
                index=0,
                item=Item(text="Generated answer.", tokens=5.0),
                char_start=0,
                char_end=18,
                sha256="abc123",
            )
        ],
        cost=CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0),
    )


def _claims() -> ClaimArtifacts:
    return ClaimArtifacts(
        claims=[
            Claim(
                item=Item(text="X is true.", tokens=3.0),
                source_chunk_index=0,
                confidence=0.9,
            )
        ],
        cost=CostEstimate(cost=0.001, input_tokens=10.0, output_tokens=3.0),
    )


def test_report_omits_none_state_keys() -> None:
    state = {
        "target_node": "grounding",
        "inputs": _base_inputs(),
        "output_chunk": _chunks(),
        "output_refined_chunks": _chunks(),
        "output_claims": _claims(),
        "grounding_metrics": None,
    }

    result = report.aggregate(state=state)

    assert "output_chunk" in result
    assert "output_refined_chunks" in result
    assert "output_claims" in result
    assert "grounding_metrics" not in result


def test_report_includes_metrics_by_state_key() -> None:
    state = {
        "target_node": "eval",
        "inputs": _base_inputs(),
        "grounding_metrics": GroundingMetrics(
            metrics=[
                MetricResult(
                    name="grounding",
                    category=MetricCategory.ANSWER,
                    score=1.0,
                    verdict="PASSED",
                )
            ],
            cost=CostEstimate(cost=0.002, input_tokens=20.0, output_tokens=2.0),
        ),
        "relevance_metrics": RelevanceMetrics(
            metrics=[
                MetricResult(
                    name="answer_relevancy",
                    category=MetricCategory.ANSWER,
                    score=0.9,
                    verdict="PASSED",
                )
            ],
            cost=CostEstimate(cost=0.003, input_tokens=30.0, output_tokens=5.0),
        ),
    }

    result = report.aggregate(state=state)

    assert result["grounding_metrics"]["metrics"][0]["name"] == "grounding"
    assert result["relevance_metrics"]["metrics"][0]["name"] == "answer_relevancy"
