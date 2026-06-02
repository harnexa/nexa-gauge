from __future__ import annotations

from ng_graph.nodes import eval as eval_node


def test_eval_batch_collector_groups_by_node_and_metric() -> None:
    collector = eval_node.EvalBatchCollector()

    assert collector.ingest_final_state({}) is False
    assert collector.ingest_final_state({"eval_summary": {"metric_rows": []}}) is False

    assert (
        collector.ingest_final_state(
            {
                "eval_summary": {
                    "metric_rows": [
                        {
                            "source_node": "grounding",
                            "metric_name": "grounding",
                            "score": 0.7,
                            "verdict": "PASSED",
                            "error": None,
                            "weight": 1.0,
                        },
                        {
                            "source_node": "refmatch",
                            "metric_name": "rouge_l",
                            "score": None,
                            "verdict": None,
                            "error": "timeout",
                            "weight": 1.0,
                        },
                    ]
                }
            }
        )
        is True
    )
    assert (
        collector.ingest_final_state(
            {
                "eval_summary": {
                    "metric_rows": [
                        {
                            "source_node": "grounding",
                            "metric_name": "grounding",
                            "score": 0.9,
                            "verdict": "FAILED",
                            "error": None,
                            "weight": 2.0,
                        }
                    ]
                }
            }
        )
        is True
    )

    summary = collector.snapshot()

    assert summary["cases_with_eval"] == 2
    assert summary["total"]["metrics"] == 3
    assert summary["total"]["errors"] == 1

    assert summary["by_node"]["grounding"]["metrics"] == 2
    assert summary["by_node"]["grounding"]["scored"] == 2
    # Weighted mean: (0.7*1 + 0.9*2) / (1+2) = 0.8333...
    assert summary["by_node"]["grounding"]["avg_score"] == 0.8333333333333334
    assert summary["by_node"]["grounding"]["median_score"] == 0.8
    assert summary["by_node"]["grounding"]["passed"] == 1
    assert summary["by_node"]["grounding"]["verdict_total"] == 2

    assert summary["by_metric"]["grounding"]["grounding"]["metrics"] == 2
    assert summary["by_metric"]["refmatch"]["rouge_l"]["errors"] == 1
    assert summary["by_metric"]["refmatch"]["rouge_l"]["verdict_total"] == 0


def test_build_eval_summary_tables_returns_node_and_metric_tables() -> None:
    collector = eval_node.EvalBatchCollector()
    collector.ingest_final_state(
        {
            "eval_summary": {
                "metric_rows": [
                    {
                        "source_node": "grounding",
                        "metric_name": "grounding",
                        "score": 0.8,
                        "verdict": "PASSED",
                        "error": None,
                        "weight": 1.0,
                    }
                ]
            }
        }
    )

    tables = eval_node.build_eval_summary_tables(collector.snapshot())
    assert len(tables) == 2
    assert len(tables[0].rows) == 1
    assert len(tables[1].rows) == 1
    assert [col.header for col in tables[0].columns][-2:] == ["median_score", "passed"]
    assert [col.header for col in tables[1].columns][-2:] == ["median_score", "passed"]
