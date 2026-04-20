from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from ng_core.cache import CacheStore, NoOpCacheStore
from ng_core.types import CostEstimate
from ng_graph import runner as runner_module
from ng_graph.runner import CachedNodeRunner


@pytest.fixture
def isolated_cache_dir() -> Path:
    cache_dir = Path(".nexagauge_test")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield cache_dir
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _install_fake_pipeline_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _scan(_state: dict) -> dict:
        return {"scan_marker": "ok"}

    def _chunk(_state: dict) -> dict:
        return {"chunk_marker": "chunked"}

    def _claims(_state: dict) -> dict:
        return {"claims_marker": "claims"}

    def _dedup(_state: dict) -> dict:
        return {"dedup_marker": "dedup"}

    def _geval_steps(_state: dict) -> dict:
        return {"geval_steps_marker": "steps"}

    def _grounding(state: dict) -> dict:
        has_context = bool(state.get("record", {}).get("context"))
        if state.get("execution_mode") == "estimate":
            if not has_context:
                return {"grounding_marker": None}
            return {
                "grounding_marker": "estimated-grounding",
                "estimated_costs": {
                    "grounding": CostEstimate(cost=0.3, input_tokens=12, output_tokens=3)
                },
            }
        return {"grounding_marker": "run-grounding" if has_context else None}

    def _relevance(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "relevance_marker": "estimated-relevance",
                "estimated_costs": {
                    "relevance": CostEstimate(cost=0.1, input_tokens=5, output_tokens=1)
                },
            }
        return {"relevance_marker": "run-relevance"}

    def _redteam(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "redteam_marker": "estimated-redteam",
                "estimated_costs": {
                    "redteam": CostEstimate(cost=0.1, input_tokens=5, output_tokens=1)
                },
            }
        return {"redteam_marker": "run-redteam"}

    def _geval(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {"geval_marker": None}
        return {"geval_marker": "run-geval"}

    def _reference(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {"reference_marker": None}
        return {"reference_marker": "run-reference"}

    def _eval(_state: dict) -> dict:
        return {"eval_marker": "eval"}

    def _report(_state: dict) -> dict:
        return {"report": []}

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _scan)
    monkeypatch.setitem(runner_module.NODE_FNS, "chunk", _chunk)
    monkeypatch.setitem(runner_module.NODE_FNS, "claims", _claims)
    monkeypatch.setitem(runner_module.NODE_FNS, "dedup", _dedup)
    monkeypatch.setitem(runner_module.NODE_FNS, "geval_steps", _geval_steps)
    monkeypatch.setitem(runner_module.NODE_FNS, "grounding", _grounding)
    monkeypatch.setitem(runner_module.NODE_FNS, "relevance", _relevance)
    monkeypatch.setitem(runner_module.NODE_FNS, "redteam", _redteam)
    monkeypatch.setitem(runner_module.NODE_FNS, "geval", _geval)
    monkeypatch.setitem(runner_module.NODE_FNS, "reference", _reference)
    monkeypatch.setitem(runner_module.NODE_FNS, "eval", _eval)
    monkeypatch.setitem(runner_module.NODE_FNS, "report", _report)


def _node_execution_counts(outcomes: list, node_name: str) -> tuple[int, int]:
    executed = 0
    cached = 0
    for outcome in outcomes:
        assert outcome.result is not None
        if node_name in outcome.result.executed_nodes:
            executed += 1
        if node_name in outcome.result.cached_nodes:
            cached += 1
    return executed, cached


def test_run_cases_iter_keeps_input_order_with_concurrency(monkeypatch) -> None:
    delays = {
        "case-0": 0.08,
        "case-1": 0.01,
        "case-2": 0.05,
        "case-3": 0.02,
    }

    def _fake_scan(state: dict) -> dict:
        case_id = state["record"]["case_id"]
        time.sleep(delays.get(case_id, 0.0))
        return {"inputs": None}

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _fake_scan)

    runner = CachedNodeRunner(cache_store=NoOpCacheStore())
    cases = [{"case_id": f"case-{i}", "generation": "hello"} for i in range(4)]
    outcomes = list(
        runner.run_cases_iter(
            cases=cases,
            node_name="scan",
            max_workers=3,
            max_in_flight=3,
            continue_on_error=True,
        )
    )

    assert [o.index for o in outcomes] == [0, 1, 2, 3]
    assert [o.case_id for o in outcomes] == ["case-0", "case-1", "case-2", "case-3"]
    assert all(o.result is not None for o in outcomes)
    assert all(o.error is None for o in outcomes)


def test_run_cases_iter_fail_fast_stops_after_first_ordered_failure(monkeypatch) -> None:
    def _fake_scan(state: dict) -> dict:
        case_id = state["record"]["case_id"]
        if case_id == "case-1":
            raise RuntimeError("boom")
        if case_id == "case-0":
            time.sleep(0.02)
        else:
            time.sleep(0.10)
        return {"inputs": None}

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _fake_scan)

    runner = CachedNodeRunner(cache_store=NoOpCacheStore())
    cases = [{"case_id": f"case-{i}", "generation": "hello"} for i in range(5)]
    outcomes = list(
        runner.run_cases_iter(
            cases=cases,
            node_name="scan",
            max_workers=3,
            max_in_flight=3,
            continue_on_error=False,
        )
    )

    assert len(outcomes) == 2
    assert outcomes[0].case_id == "case-0"
    assert outcomes[0].result is not None
    assert outcomes[0].error is None

    assert outcomes[1].case_id == "case-1"
    assert outcomes[1].result is None
    assert "boom" in (outcomes[1].error or "")


def test_estimate_mode_reuses_run_cache_for_shared_prerequisites(monkeypatch, tmp_path) -> None:
    def _fake_scan(_state: dict) -> dict:
        return {"scan_marker": "ok"}

    def _fake_chunk(_state: dict) -> dict:
        return {"chunk_marker": "chunked"}

    def _fake_claims(_state: dict) -> dict:
        return {"claims_marker": "claims"}

    def _fake_dedup(_state: dict) -> dict:
        return {"dedup_marker": "dedup"}

    def _fake_grounding(_state: dict) -> dict:
        return {"grounding_marker": "grounded"}

    def _fake_relevance(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "relevance_metrics": "estimated-relevance",
                "estimated_costs": {
                    "relevance": CostEstimate(cost=0.25, input_tokens=10, output_tokens=2)
                },
            }
        return {"relevance_metrics": "relevance"}

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _fake_scan)
    monkeypatch.setitem(runner_module.NODE_FNS, "chunk", _fake_chunk)
    monkeypatch.setitem(runner_module.NODE_FNS, "claims", _fake_claims)
    monkeypatch.setitem(runner_module.NODE_FNS, "dedup", _fake_dedup)
    monkeypatch.setitem(runner_module.NODE_FNS, "grounding", _fake_grounding)
    monkeypatch.setitem(runner_module.NODE_FNS, "relevance", _fake_relevance)

    runner = CachedNodeRunner(cache_store=CacheStore(tmp_path))
    case = {"case_id": "same-case", "generation": "hello world"}

    run_result = runner.run_case(case=case, node_name="grounding", execution_mode="run")
    assert "grounding" in run_result.executed_nodes

    estimate_result = runner.run_case(case=case, node_name="relevance", execution_mode="estimate")
    assert set(["scan", "chunk", "claims", "dedup"]).issubset(set(estimate_result.cached_nodes))
    assert "relevance" in estimate_result.executed_nodes
    assert estimate_result.final_state["estimated_costs"]["relevance"].cost == 0.25


def test_estimate_mode_only_executes_for_uncached_new_records(monkeypatch, tmp_path) -> None:
    def _fake_scan(_state: dict) -> dict:
        return {"scan_marker": "ok"}

    def _fake_chunk(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "chunk_marker": "chunked",
                "estimated_costs": {
                    "chunk": CostEstimate(cost=0.1, input_tokens=5, output_tokens=1)
                },
            }
        return {"chunk_marker": "chunked"}

    def _fake_claims(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "claims_marker": "claims",
                "estimated_costs": {
                    "claims": CostEstimate(cost=0.2, input_tokens=8, output_tokens=2)
                },
            }
        return {"claims_marker": "claims"}

    def _fake_dedup(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "dedup_marker": "dedup",
                "estimated_costs": {
                    "dedup": CostEstimate(cost=0.0, input_tokens=0, output_tokens=0)
                },
            }
        return {"dedup_marker": "dedup"}

    def _fake_grounding(state: dict) -> dict:
        if state.get("execution_mode") == "estimate":
            return {
                "grounding_marker": "estimated-grounding",
                "estimated_costs": {
                    "grounding": CostEstimate(cost=0.3, input_tokens=12, output_tokens=3)
                },
            }
        return {"grounding_marker": "grounding"}

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _fake_scan)
    monkeypatch.setitem(runner_module.NODE_FNS, "chunk", _fake_chunk)
    monkeypatch.setitem(runner_module.NODE_FNS, "claims", _fake_claims)
    monkeypatch.setitem(runner_module.NODE_FNS, "dedup", _fake_dedup)
    monkeypatch.setitem(runner_module.NODE_FNS, "grounding", _fake_grounding)

    runner = CachedNodeRunner(cache_store=CacheStore(tmp_path))
    cached_case = {"case_id": "case-0", "generation": "same text"}
    new_case = {"case_id": "case-1", "generation": "different text"}

    runner.run_case(case=cached_case, node_name="grounding", execution_mode="run")

    outcomes = list(
        runner.run_cases_iter(
            cases=[cached_case, new_case],
            node_name="grounding",
            execution_mode="estimate",
            max_workers=1,
            continue_on_error=True,
        )
    )

    first = outcomes[0].result
    assert first is not None
    assert first.executed_nodes == ["eval", "report"]
    assert "grounding" in first.cached_nodes
    assert first.final_state.get("estimated_costs", {}) == {}

    second = outcomes[1].result
    assert second is not None
    assert "grounding" in second.executed_nodes
    assert second.final_state["estimated_costs"]["grounding"].cost == 0.3


def test_eval_estimate_mode_merges_parallel_metric_estimated_costs(monkeypatch) -> None:
    def _ok(_state: dict) -> dict:
        return {}

    def _relevance(_state: dict) -> dict:
        return {
            "relevance_metrics": "estimated-relevance",
            "estimated_costs": {
                "relevance": CostEstimate(cost=0.11, input_tokens=3, output_tokens=1)
            },
        }

    def _grounding(_state: dict) -> dict:
        return {
            "grounding_metrics": "estimated-grounding",
            "estimated_costs": {
                "grounding": CostEstimate(cost=0.22, input_tokens=4, output_tokens=1)
            },
        }

    def _redteam(_state: dict) -> dict:
        return {
            "redteam_metrics": "estimated-redteam",
            "estimated_costs": {
                "redteam": CostEstimate(cost=0.33, input_tokens=5, output_tokens=2)
            },
        }

    def _geval(_state: dict) -> dict:
        return {
            "geval_metrics": "estimated-geval",
            "estimated_costs": {"geval": CostEstimate(cost=0.44, input_tokens=6, output_tokens=2)},
        }

    def _reference(_state: dict) -> dict:
        return {
            "reference_metrics": "estimated-reference",
            "estimated_costs": {
                "reference": CostEstimate(cost=0.0, input_tokens=0, output_tokens=0)
            },
        }

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _ok)
    monkeypatch.setitem(runner_module.NODE_FNS, "chunk", _ok)
    monkeypatch.setitem(runner_module.NODE_FNS, "claims", _ok)
    monkeypatch.setitem(runner_module.NODE_FNS, "dedup", _ok)
    monkeypatch.setitem(runner_module.NODE_FNS, "geval_steps", _ok)
    monkeypatch.setitem(runner_module.NODE_FNS, "relevance", _relevance)
    monkeypatch.setitem(runner_module.NODE_FNS, "grounding", _grounding)
    monkeypatch.setitem(runner_module.NODE_FNS, "redteam", _redteam)
    monkeypatch.setitem(runner_module.NODE_FNS, "geval", _geval)
    monkeypatch.setitem(runner_module.NODE_FNS, "reference", _reference)
    monkeypatch.setitem(runner_module.NODE_FNS, "eval", _ok)
    monkeypatch.setitem(runner_module.NODE_FNS, "report", _ok)

    runner = CachedNodeRunner(cache_store=NoOpCacheStore())
    case = {"case_id": "merge-case", "generation": "hello world"}
    result = runner.run_case(case=case, node_name="eval", execution_mode="estimate", force=True)

    costs = result.final_state.get("estimated_costs", {})
    assert set(["relevance", "grounding", "redteam", "geval", "reference"]).issubset(set(costs))
    assert costs["relevance"].cost == 0.11
    assert costs["grounding"].cost == 0.22
    assert costs["redteam"].cost == 0.33
    assert costs["geval"].cost == 0.44


def test_estimate_mode_does_not_write_cache_by_default(monkeypatch, tmp_path) -> None:
    def _fake_scan(_state: dict) -> dict:
        return {"scan_marker": "ok"}

    def _fake_chunk(_state: dict) -> dict:
        return {
            "chunk_marker": "chunked",
            "estimated_costs": {"chunk": CostEstimate(cost=0.1, input_tokens=5, output_tokens=1)},
        }

    def _fake_claims(_state: dict) -> dict:
        return {
            "claims_marker": "claims",
            "estimated_costs": {"claims": CostEstimate(cost=0.2, input_tokens=8, output_tokens=2)},
        }

    def _fake_dedup(_state: dict) -> dict:
        return {
            "dedup_marker": "dedup",
            "estimated_costs": {"dedup": CostEstimate(cost=0.0, input_tokens=0, output_tokens=0)},
        }

    def _fake_grounding(_state: dict) -> dict:
        return {
            "grounding_marker": "estimated-grounding",
            "estimated_costs": {
                "grounding": CostEstimate(cost=0.3, input_tokens=12, output_tokens=3)
            },
        }

    monkeypatch.setitem(runner_module.NODE_FNS, "scan", _fake_scan)
    monkeypatch.setitem(runner_module.NODE_FNS, "chunk", _fake_chunk)
    monkeypatch.setitem(runner_module.NODE_FNS, "claims", _fake_claims)
    monkeypatch.setitem(runner_module.NODE_FNS, "dedup", _fake_dedup)
    monkeypatch.setitem(runner_module.NODE_FNS, "grounding", _fake_grounding)

    runner = CachedNodeRunner(cache_store=CacheStore(tmp_path))
    case = {"case_id": "case-estimate", "generation": "fresh text"}

    first = runner.run_case(
        case=case, node_name="grounding", execution_mode="estimate", force=False
    )
    second = runner.run_case(
        case=case, node_name="grounding", execution_mode="estimate", force=False
    )

    assert "grounding" in first.executed_nodes
    assert "grounding" in second.executed_nodes
    assert "grounding" not in second.cached_nodes
    assert second.final_state["estimated_costs"]["grounding"].cost == 0.3


def test_eval_estimate_reuses_cached_grounding_from_run_branch(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
) -> None:
    _install_fake_pipeline_nodes(monkeypatch)

    runner = CachedNodeRunner(cache_store=CacheStore(isolated_cache_dir))
    cases = [
        {"case_id": "case-0", "generation": "same text", "question": "q", "context": ["ctx-0"]},
        {"case_id": "case-1", "generation": "same text", "question": "q", "context": ["ctx-1"]},
        {"case_id": "case-2", "generation": "same text", "question": "q"},
    ]

    run_grounding = list(
        runner.run_cases_iter(
            cases=cases,
            node_name="grounding",
            execution_mode="run",
            max_workers=1,
            continue_on_error=True,
        )
    )
    assert all(outcome.result is not None for outcome in run_grounding)

    estimate_grounding = list(
        runner.run_cases_iter(
            cases=cases,
            node_name="grounding",
            execution_mode="estimate",
            max_workers=1,
            continue_on_error=True,
        )
    )
    grounding_exec, grounding_cached = _node_execution_counts(estimate_grounding, "grounding")
    assert grounding_exec == 0
    assert grounding_cached == 3

    estimate_eval = list(
        runner.run_cases_iter(
            cases=cases,
            node_name="eval",
            execution_mode="estimate",
            max_workers=1,
            continue_on_error=True,
        )
    )
    eval_grounding_exec, eval_grounding_cached = _node_execution_counts(estimate_eval, "grounding")
    assert eval_grounding_exec == 0
    assert eval_grounding_cached == 3
    for outcome in estimate_eval:
        assert outcome.result is not None
        assert "grounding" not in (outcome.result.final_state.get("estimated_costs") or {})


def test_eval_estimate_executes_grounding_when_case_content_changes(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
) -> None:
    _install_fake_pipeline_nodes(monkeypatch)

    runner = CachedNodeRunner(cache_store=CacheStore(isolated_cache_dir))
    run_case = {
        "case_id": "case-0",
        "generation": "same text",
        "question": "q",
        "context": ["ctx-0"],
    }
    changed_case = {
        "case_id": "case-0",
        "generation": "same text with update",
        "question": "q",
        "context": ["ctx-0"],
    }

    runner.run_case(case=run_case, node_name="grounding", execution_mode="run")
    eval_estimate = runner.run_case(case=changed_case, node_name="eval", execution_mode="estimate")

    assert "grounding" in eval_estimate.executed_nodes
    assert "grounding" not in eval_estimate.cached_nodes
    assert eval_estimate.final_state["estimated_costs"]["grounding"].cost == 0.3


def test_eval_and_report_are_never_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
) -> None:
    _install_fake_pipeline_nodes(monkeypatch)

    runner = CachedNodeRunner(cache_store=CacheStore(isolated_cache_dir))
    case = {
        "case_id": "case-eval-cache-policy",
        "generation": "hello",
        "question": "q",
        "context": ["ctx"],
    }

    first = runner.run_case(case=case, node_name="eval", execution_mode="run")
    second = runner.run_case(case=case, node_name="eval", execution_mode="run")

    assert "eval" in first.executed_nodes
    assert "report" in first.executed_nodes
    assert "eval" in second.executed_nodes
    assert "report" in second.executed_nodes
    assert "eval" not in second.cached_nodes
    assert "report" not in second.cached_nodes
