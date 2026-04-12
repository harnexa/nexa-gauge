from __future__ import annotations

import time

from lumiseval_core.cache import NoOpCacheStore
from lumiseval_graph import runner as runner_module
from lumiseval_graph.runner import CachedNodeRunner


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
