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
)
from ng_graph.nodes import report


def test_report_aggregate_emits_state_key_sections() -> None:
    state = {
        "target_node": "grounding",
        "inputs": Inputs(
            case_id="case-1",
            output=Item(text="Paris is in France.", tokens=4.0),
            input=Item(text="Where is Paris?", tokens=3.0),
            context=Item(text="Paris is in France.", tokens=4.0),
            reference=Item(text="Paris", tokens=1.0),
            has_output=True,
            has_input=True,
            has_context=True,
            has_reference=True,
        ),
        "output_chunk": ChunkArtifacts(
            chunks=[
                Chunk(
                    index=0,
                    item=Item(text="Paris is in France.", tokens=4.0),
                    char_start=0,
                    char_end=19,
                    sha256="abc",
                )
            ],
            cost=CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0),
        ),
        "output_claims": ClaimArtifacts(
            claims=[
                Claim(item=Item(text="Paris is in France.", tokens=4.0), source_chunk_index=0),
            ],
            cost=CostEstimate(cost=0.1, input_tokens=10.0, output_tokens=3.0),
        ),
    }

    out = report.aggregate(state=state)

    assert out["target_node"] == "grounding"
    assert out["input"]["case_id"] == "case-1"
    assert out["output_chunk"]["text"] == ["Paris is in France."]
    assert out["output_claims"]["text"] == ["Paris is in France."]
    assert "output_refined_chunks" not in out


def test_report_aggregate_projects_metric_wrappers() -> None:
    state = {
        "target_node": "grounding",
        "inputs": Inputs(case_id="case-2", output=Item(text="A", tokens=1.0), has_output=True),
        "grounding_metrics": GroundingMetrics(
            metrics=[
                MetricResult(
                    name="grounding",
                    category=MetricCategory.ANSWER,
                    score=1.0,
                    verdict="PASSED",
                )
            ],
            cost=CostEstimate(cost=0.2, input_tokens=3.0, output_tokens=1.0),
        ),
    }

    out = report.aggregate(state=state)

    assert out["grounding_metrics"]["metrics"][0]["name"] == "grounding"
    assert out["grounding_metrics"]["metrics"][0]["verdict"] == "PASSED"
    assert out["grounding_metrics"]["cost"]["cost"] == 0.2
