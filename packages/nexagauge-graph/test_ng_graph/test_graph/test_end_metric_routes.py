# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py::test_report_for_metric_targets_contains_only_that_branch
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py::test_report_for_eval_contains_all_metric_branches

from __future__ import annotations

from collections.abc import Callable

import pytest
from ng_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    GevalMetrics,
    GroundingMetrics,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    RedteamMetrics,
    RefalignMetrics,
    RefmatchMetrics,
    RelevanceMetrics,
)

_ZERO_COST = CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0)


def _empty_metric_groups() -> dict[str, None]:
    return {
        "grounding_metrics": None,
        "relevance_metrics": None,
        "redteam_metrics": None,
        "geval_metrics": None,
        "refmatch_metrics": None,
        "refalign_metrics": None,
    }


_GROUP_KEY_TO_WRAPPER: dict[str, type] = {
    "grounding_metrics": GroundingMetrics,
    "relevance_metrics": RelevanceMetrics,
    "redteam_metrics": RedteamMetrics,
    "geval_metrics": GevalMetrics,
    "refmatch_metrics": RefmatchMetrics,
    "refalign_metrics": RefalignMetrics,
}

_GROUP_KEY_TO_SECTION: dict[str, str] = {
    "grounding_metrics": "grounding_metrics",
    "relevance_metrics": "relevance_metrics",
    "redteam_metrics": "redteam_metrics",
    "geval_metrics": "geval_metrics",
    "refmatch_metrics": "refmatch_metrics",
    "refalign_metrics": "refalign_metrics",
}


@pytest.mark.parametrize(
    ("target_node", "group_key", "metric_name", "category"),
    [
        ("grounding", "grounding_metrics", "grounding", MetricCategory.ANSWER),
        ("relevance", "relevance_metrics", "answer_relevancy", MetricCategory.ANSWER),
        ("refmatch", "refmatch_metrics", "rouge_l", MetricCategory.RETRIEVAL),
        ("refalign", "refalign_metrics", "refalign_f1", MetricCategory.ANSWER),
        ("geval", "geval_metrics", "geval_coherence", MetricCategory.ANSWER),
        ("redteam", "redteam_metrics", "vulnerability_prompt_injection", MetricCategory.ANSWER),
    ],
)
def test_report_for_metric_targets_contains_only_that_branch(
    target_node: str,
    group_key: str,
    metric_name: str,
    category: MetricCategory,
    make_metric: Callable[[str, float, MetricCategory], MetricResult],
    graph_module,
) -> None:
    """Verify eval-target report exposes exactly one metric section for each routed metric wrapper.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py::test_report_for_metric_targets_contains_only_that_branch
    """
    groups = _empty_metric_groups()
    wrapper_cls = _GROUP_KEY_TO_WRAPPER[group_key]
    groups[group_key] = wrapper_cls(
        metrics=[make_metric(metric_name, 0.8, category)],
        cost=_ZERO_COST,
    )

    state = {
        "target_node": "eval",
        "job_id": f"job-{target_node}",
        **groups,
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    eval_out = graph_module.node_eval(state)
    state.update(eval_out)
    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert isinstance(report, dict)
    assert report["target_node"] == "eval"

    section_name = _GROUP_KEY_TO_SECTION[group_key]
    assert section_name in report
    assert isinstance(report[section_name]["metrics"], list)
    assert len(report[section_name]["metrics"]) == 1


def test_report_for_eval_contains_all_metric_branches(
    make_metric: Callable[[str, float, MetricCategory], MetricResult],
    graph_module,
) -> None:
    """Verify eval-target report includes all metric sections when all metric wrappers are present.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py::test_report_for_eval_contains_all_metric_branches
    """
    state = {
        "target_node": "eval",
        "job_id": "job-eval",
        "grounding_metrics": GroundingMetrics(
            metrics=[make_metric("grounding", 1.0, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "relevance_metrics": RelevanceMetrics(
            metrics=[make_metric("answer_relevancy", 0.9, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "redteam_metrics": RedteamMetrics(
            metrics=[make_metric("vulnerability_prompt_injection", 0.2, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "geval_metrics": GevalMetrics(
            metrics=[make_metric("geval_coherence", 0.7, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "refmatch_metrics": RefmatchMetrics(
            metrics=[make_metric("rouge_l", 0.6, MetricCategory.RETRIEVAL)],
            cost=_ZERO_COST,
        ),
        "refalign_metrics": RefalignMetrics(
            metrics=[make_metric("refalign_f1", 0.6, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    eval_out = graph_module.node_eval(state)
    state.update(eval_out)
    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert isinstance(report, dict)
    assert report["target_node"] == "eval"
    for section in (
        "grounding_metrics",
        "relevance_metrics",
        "redteam_metrics",
        "geval_metrics",
        "refmatch_metrics",
        "refalign_metrics",
    ):
        assert section in report, f"Expected section '{section}' in report"
        assert isinstance(report[section]["metrics"], list)
        assert len(report[section]["metrics"]) == 1


def test_node_eval_is_orchestration_only(graph_module) -> None:
    """Verify node_eval is a no-op orchestration node and returns an empty patch.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py::test_node_eval_is_orchestration_only
    """
    assert graph_module.node_eval({}) == {}


def test_node_eval_collects_metric_rows_for_cli_aggregation(
    make_metric: Callable[[str, float, MetricCategory], MetricResult],
    graph_module,
) -> None:
    state = {
        "grounding_metrics": GroundingMetrics(
            metrics=[make_metric("grounding", 1.0, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "relevance_metrics": RelevanceMetrics(
            metrics=[make_metric("answer_relevancy", 0.9, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "redteam_metrics": RedteamMetrics(
            metrics=[make_metric("vulnerability_prompt_injection", 0.2, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "geval_metrics": GevalMetrics(
            metrics=[make_metric("geval_coherence", 0.7, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
        "refmatch_metrics": RefmatchMetrics(
            metrics=[make_metric("rouge_l", 0.6, MetricCategory.RETRIEVAL)],
            cost=_ZERO_COST,
        ),
        "refalign_metrics": RefalignMetrics(
            metrics=[make_metric("refalign_f1", 0.6, MetricCategory.ANSWER)],
            cost=_ZERO_COST,
        ),
    }

    eval_out = graph_module.node_eval(state)
    summary = eval_out["eval_summary"]
    rows = summary["metric_rows"]

    assert summary["schema_version"] == 1
    assert len(rows) == 6
    assert {row["source_node"] for row in rows} == {
        "grounding",
        "relevance",
        "redteam",
        "geval",
        "refmatch",
        "refalign",
    }
    assert {row["metric_name"] for row in rows} == {
        "grounding",
        "answer_relevancy",
        "vulnerability_prompt_injection",
        "geval_coherence",
        "rouge_l",
        "refalign_f1",
    }
    assert all("verdict" in row for row in rows)


def test_report_for_grounding_target_includes_inputs_and_branch_nodes(graph_module) -> None:
    """Verify grounding-target report includes input fields and expected grounding/claims sections.

    Run: uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_graph/test_end_metric_routes.py::test_report_for_grounding_target_includes_inputs_and_branch_nodes
    """
    state = {
        "target_node": "grounding",
        "record": {
            "case_id": "case-1",
            "output": "Paris is the capital of France.",
            "input": "What is the capital of France?",
            "reference": "Paris",
            "context": "France has Paris as its capital.",
        },
        "inputs": Inputs(
            case_id="case-1",
            output=Item(text="Paris is the capital of France.", tokens=7.0),
            input=Item(text="What is the capital of France?", tokens=7.0),
            context=Item(text="France has Paris as its capital.", tokens=7.0),
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
                    item=Item(text="Paris is the capital of France.", tokens=7.0),
                    char_start=0,
                    char_end=31,
                    sha256="abc123",
                )
            ],
            cost=CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0),
        ),
        "output_claims": ClaimArtifacts(
            claims=[
                Claim(
                    item=Item(text="Paris is the capital of France.", tokens=7.0),
                    source_chunk_index=0,
                    confidence=0.95,
                )
            ],
            cost=CostEstimate(cost=0.001, input_tokens=10.0, output_tokens=3.0),
        ),
        "output_refined_chunks": ChunkArtifacts(
            chunks=[
                Chunk(
                    index=0,
                    item=Item(text="Paris is the capital of France.", tokens=7.0),
                    char_start=0,
                    char_end=31,
                    sha256="abc123",
                )
            ],
            cost=CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0),
        ),
        "grounding_metrics": graph_module.GroundingMetrics(
            metrics=[
                MetricResult(
                    name="grounding",
                    category=MetricCategory.ANSWER,
                    score=1.0,
                )
            ],
            cost=CostEstimate(cost=0.002, input_tokens=20.0, output_tokens=2.0),
        ),
        "node_model_usage": {
            "claims": {
                "used_models": ["openai/gpt-4o-mini"],
                "used_model_counts": {"openai/gpt-4o-mini": 1},
                "total_calls": 1,
                "fallback_hits": 0,
            },
            "grounding": {
                "used_models": ["openai/gpt-4o"],
                "used_model_counts": {"openai/gpt-4o": 1},
                "total_calls": 1,
                "fallback_hits": 1,
            },
        },
        "estimated_costs": {},
    }

    report_out = graph_module.node_report(state)
    report = report_out["report"]

    # Always-present sections
    assert report["target_node"] == "grounding"
    assert report["input"]["case_id"] == "case-1"
    assert report["input"]["input"] == "What is the capital of France?"
    assert report["input"]["output"] == "Paris is the capital of France."
    assert report["input"]["context"] == "France has Paris as its capital."
    assert report["input"]["reference"] == "Paris"

    # Chunk section
    assert report["output_chunk"]["text"] == ["Paris is the capital of France."]

    # Claims section
    assert report["output_claims"]["text"] == ["Paris is the capital of France."]
    assert report["output_claims"]["cost"]["cost"] == pytest.approx(0.001)

    # Refined chunks section
    assert report["output_refined_chunks"]["text"] == ["Paris is the capital of France."]

    # Grounding section
    assert isinstance(report["grounding_metrics"]["metrics"], list)
    assert len(report["grounding_metrics"]["metrics"]) == 1
    assert report["grounding_metrics"]["cost"]["cost"] == pytest.approx(0.002)
