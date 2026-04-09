# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_metric_targets_contains_only_that_branch
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_eval_contains_all_metric_branches

from __future__ import annotations

from collections.abc import Callable

import pytest

from lumiseval_core.types import MetricCategory, MetricResult


def _empty_metric_groups() -> dict[str, list[MetricResult]]:
    return {
        "grounding_metrics": [],
        "relevance_metrics": [],
        "redteam_metrics": [],
        "geval_metrics": [],
        "reference_metrics": [],
    }


@pytest.mark.parametrize(
    ("target_node", "group_key", "metric_name", "category"),
    [
        ("grounding", "grounding_metrics", "grounding", MetricCategory.ANSWER),
        ("relevance", "relevance_metrics", "answer_relevancy", MetricCategory.ANSWER),
        ("reference", "reference_metrics", "rouge_l", MetricCategory.RETRIEVAL),
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
    """
    pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_metric_targets_contains_only_that_branch
    """
    groups = _empty_metric_groups()
    groups[group_key] = [make_metric(metric_name, 0.8, category)]

    state = {
        "job_id": f"job-{target_node}",
        **groups,
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    eval_out = graph_module.node_eval(state)
    print("eval_out ", eval_out)
    state.update(eval_out)
    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert isinstance(report, list)
    assert len(report) == 1
    assert report[0]["name"] == metric_name


def test_report_for_eval_contains_all_metric_branches(
    make_metric: Callable[[str, float, MetricCategory], MetricResult],
    graph_module,
) -> None:
    """
    pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_eval_contains_all_metric_branches
    """
    state = {
        "job_id": "job-eval",
        "grounding_metrics": [make_metric("grounding", 1.0, MetricCategory.ANSWER)],
        "relevance_metrics": [make_metric("answer_relevancy", 0.9, MetricCategory.ANSWER)],
        "redteam_metrics": [
            make_metric("vulnerability_prompt_injection", 0.2, MetricCategory.ANSWER)
        ],
        "geval_metrics": [make_metric("geval_coherence", 0.7, MetricCategory.ANSWER)],
        "reference_metrics": [make_metric("rouge_l", 0.6, MetricCategory.RETRIEVAL)],
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    eval_out = graph_module.node_eval(state)
    state.update(eval_out)
    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert isinstance(report, list)
    assert len(report) == 5
    assert [m["name"] for m in report] == [
        "grounding",
        "answer_relevancy",
        "vulnerability_prompt_injection",
        "geval_coherence",
        "rouge_l",
    ]


def test_node_eval_is_orchestration_only(graph_module) -> None:
    assert graph_module.node_eval({}) == {}
