"""Cache-aware node runner aligned to graph.py as source of truth.

Design goals:
- Graph-first execution state: scanner node is the first step for every branch.
- Node-level key-value cache: each node patch is cached independently.
- Backend-agnostic keys: opaque cache keys can be reused for Redis/Dynamo.
- Safe concurrency: optional record-level parallelism plus existing eval metric fan-out.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Any, Iterable, Iterator, Mapping

from pydantic import BaseModel

from lumiseval_core.cache import NodeCacheBackend, build_node_cache_key, compute_case_hash
from lumiseval_core.config import config as cfg
from lumiseval_graph.llm.config import get_node_config
from lumiseval_graph.graph import EvalCase
from lumiseval_graph.registry import NODE_FNS
from lumiseval_graph.topology import METRIC_NODES, NODES_BY_NAME


class CachedNodeRunResult(BaseModel):
    """Result payload for cache-aware single-case execution."""

    node_name: str
    case_id: str
    executed_nodes: list[str]
    cached_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]
    elapsed_ms: float


class BatchRunResult(BaseModel):
    """Batch execution result for many records."""

    results: list[CachedNodeRunResult]
    failures: list[tuple[str, str]]


class CaseRunOutcome(BaseModel):
    """Ordered streaming outcome for one submitted case."""

    index: int
    case_id: str
    result: CachedNodeRunResult | None = None
    error: str | None = None


def _case_value(case: Any, key: str, default: Any = None) -> Any:
    if isinstance(case, dict):
        return case.get(key, default)
    return getattr(case, key, default)


def _case_id(case: Any) -> str:
    value = _case_value(case, "case_id", "")
    text = str(value).strip()
    return text or "unknown-case"


def _stable_json(obj: Any) -> str:
    def _default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return str(value)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _build_initial_state(case: dict[str, Any], *, execution_mode: str) -> EvalCase:
    record = {
        "case_id": _case_id(case),
        "generation": _case_value(case, "generation"),
        "question": _case_value(case, "question"),
        "reference": _case_value(case, "reference"),
        "context": _case_value(case, "context") or [],
        "geval": _case_value(case, "geval"),
        "redteam": _case_value(case, "redteam"),
    }
    return EvalCase(
        record=record,
        llm_overrides=_case_value(case, "llm_overrides"),
        execution_mode=execution_mode,
        estimated_costs={},
        reference_files=_case_value(case, "reference_files") or [],
    )


def _compute_case_fingerprint(case: dict[str, Any]) -> str:
    return compute_case_hash(
        generation=_case_value(case, "generation", ""),
        question=_case_value(case, "question"),
        reference=_case_value(case, "reference"),
        geval=_case_value(case, "geval"),
        redteam=_case_value(case, "redteam"),
        context=_case_value(case, "context") or [],
        reference_files=_case_value(case, "reference_files") or [],
    )


def _node_route_fingerprint(node_name: str, *, state: Mapping[str, Any], execution_mode: str) -> str:
    llm_overrides = state.get("llm_overrides")
    node_cfg = get_node_config(node_name, llm_overrides=llm_overrides)
    resolved_model = node_cfg.model or cfg.LLM_MODEL
    payload = {
        "execution_mode": execution_mode,
        "node": node_name,
        "model": resolved_model,
        "fallback_model": node_cfg.fallback_model,
        "temperature": node_cfg.temperature,
    }
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:16]


def _step_fingerprint(
    *,
    parent_fingerprint: str,
    node_name: str,
    state: Mapping[str, Any],
    execution_mode: str,
) -> str:
    route_fingerprint = _node_route_fingerprint(
        node_name,
        state=state,
        execution_mode=execution_mode,
    )
    raw = f"{parent_fingerprint}|{node_name}|{route_fingerprint}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_key_for_step(
    *,
    case_fingerprint: str,
    node_name: str,
    step_fingerprint: str,
    execution_mode: str,
) -> str:
    return build_node_cache_key(
        case_fingerprint=case_fingerprint,
        node_name=node_name,
        execution_mode=execution_mode,
        node_route_fingerprint=step_fingerprint,
    )


def _plan_nodes(node_name: str) -> list[str]:
    plan = list(NODES_BY_NAME[node_name].prerequisites) + [node_name]
    needs_report = node_name == "eval" or node_name in METRIC_NODES
    if needs_report and node_name != "report" and "report" not in plan and "report" in NODE_FNS:
        plan.append("report")
    return plan


class CachedNodeRunner:
    """Execute pipeline nodes with graph-aligned state and node-level cache reuse."""

    def __init__(self, cache_store: NodeCacheBackend) -> None:
        self._cache = cache_store

    @staticmethod
    def _normalize_max_in_flight(*, max_workers: int, max_in_flight: int | None) -> int:
        workers = max(1, max_workers)
        if max_in_flight is None:
            return max(1, workers * 2)
        return max(workers, max_in_flight)

    def _read_step_cache(self, cache_key: str) -> dict[str, Any] | None:
        entry = self._cache.get_entry_by_key(cache_key)
        if entry is None:
            return None
        return entry["node_output"]

    def _write_step_cache(
        self,
        *,
        cache_key: str,
        node_name: str,
        output: dict[str, Any],
        execution_mode: str,
        case_fingerprint: str,
    ) -> None:
        self._cache.put_by_key(
            cache_key,
            node_name,
            output,
            metadata={
                "execution_mode": execution_mode,
                "case_fingerprint": case_fingerprint,
                "cache_schema": "v2",
            },
        )

    def run_case(
        self,
        *,
        case: dict[str, Any],
        node_name: str,
        force: bool = False,
        execution_mode: str = "run",
    ) -> CachedNodeRunResult:
        if node_name not in NODE_FNS:
            valid = ", ".join(sorted(NODE_FNS))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        t0 = time.monotonic()
    
        state: EvalCase = _build_initial_state(case, execution_mode=execution_mode)
        
        case_fingerprint = _compute_case_fingerprint(case)

        plan = _plan_nodes(node_name)
        path_fingerprint = case_fingerprint

        executed: list[str] = []
        cached: list[str] = []
        node_output: dict[str, Any] = {}

        i = 0
        while i < len(plan):
            step = plan[i]

            if node_name == "eval" and METRIC_NODES and step == METRIC_NODES[0]:
                cached_group_outputs: dict[str, dict[str, Any]] = {}
                metric_fingerprints: dict[str, str] = {}
                to_run: list[tuple[str, str, str]] = []
                group_parent_fingerprint = path_fingerprint

                for metric_step in METRIC_NODES:
                    metric_fingerprint = _step_fingerprint(
                        parent_fingerprint=group_parent_fingerprint,
                        node_name=metric_step,
                        state=state,
                        execution_mode=execution_mode,
                    )
                    metric_fingerprints[metric_step] = metric_fingerprint
                    metric_key = _cache_key_for_step(
                        case_fingerprint=case_fingerprint,
                        node_name=metric_step,
                        step_fingerprint=metric_fingerprint,
                        execution_mode=execution_mode,
                    )
                    if not force:
                        cached_output = self._read_step_cache(metric_key)
                        if cached_output is not None:
                            cached_group_outputs[metric_step] = cached_output
                            cached.append(metric_step)
                            continue
                    to_run.append((metric_step, metric_key, metric_fingerprint))

                run_group_outputs: dict[str, dict[str, Any]] = {}
                if to_run:
                    with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
                        # Use an isolated snapshot per metric task to avoid shared
                        # nested object mutation races in parallel execution.
                        futures = {
                            pool.submit(NODE_FNS[metric_step], deepcopy(state)): (metric_step, metric_key)
                            for metric_step, metric_key, _ in to_run
                        }
                        for future in as_completed(futures):
                            metric_step, metric_key = futures[future]
                            output = future.result()
                            run_group_outputs[metric_step] = output
                            executed.append(metric_step)
                            self._write_step_cache(
                                cache_key=metric_key,
                                node_name=metric_step,
                                output=output,
                                execution_mode=execution_mode,
                                case_fingerprint=case_fingerprint,
                            )

                for metric_step in METRIC_NODES:
                    merged_output = cached_group_outputs.get(metric_step) or run_group_outputs.get(metric_step)
                    if merged_output is not None:
                        state.update(merged_output)

                group_signature_raw = "|".join(
                    [group_parent_fingerprint, "metrics", *[metric_fingerprints[n] for n in METRIC_NODES]]
                )
                path_fingerprint = hashlib.sha256(group_signature_raw.encode()).hexdigest()[:16]
                i += len(METRIC_NODES)
                continue

            step_fingerprint = _step_fingerprint(
                parent_fingerprint=path_fingerprint,
                node_name=step,
                state=state,
                execution_mode=execution_mode,
            )
            cache_key = _cache_key_for_step(
                case_fingerprint=case_fingerprint,
                node_name=step,
                step_fingerprint=step_fingerprint,
                execution_mode=execution_mode,
            )
            if not force:
                cached_output = self._read_step_cache(cache_key)
                if cached_output is not None:
                    state.update(cached_output)
                    cached.append(step)
                    if step == node_name:
                        node_output = cached_output
                    path_fingerprint = step_fingerprint
                    i += 1
                    continue

            output = NODE_FNS[step](state)
            state.update(output)
            self._write_step_cache(
                cache_key=cache_key,
                node_name=step,
                output=output,
                execution_mode=execution_mode,
                case_fingerprint=case_fingerprint,
            )
            executed.append(step)
            if step == node_name:
                node_output = output
            path_fingerprint = step_fingerprint
            i += 1

        elapsed_ms = (time.monotonic() - t0) * 1000

        return CachedNodeRunResult(
            node_name=node_name,
            case_id=_case_id(case),
            executed_nodes=executed,
            cached_nodes=cached,
            node_output=node_output,
            final_state=dict(state),
            elapsed_ms=elapsed_ms,
        )

    def run_cases_iter(
        self,
        *,
        cases: Iterable[dict[str, Any]],
        node_name: str,
        force: bool = False,
        execution_mode: str = "run",
        max_workers: int = 1,
        max_in_flight: int | None = None,
        continue_on_error: bool = True,
    ) -> Iterator[CaseRunOutcome]:
        workers = max(1, max_workers)
        in_flight_limit = self._normalize_max_in_flight(
            max_workers=workers,
            max_in_flight=max_in_flight,
        )

        if workers == 1:
            for idx, case in enumerate(cases):
                case_id = _case_id(case)
                try:
                    result = self.run_case(
                        case=case,
                        node_name=node_name,
                        force=force,
                        execution_mode=execution_mode,
                    )
                    yield CaseRunOutcome(index=idx, case_id=result.case_id, result=result)
                except Exception as exc:
                    yield CaseRunOutcome(index=idx, case_id=case_id, error=str(exc))
                    if not continue_on_error:
                        return
            return

        case_iter = iter(cases)
        submit_index = 0
        emit_index = 0
        stop_submitting = False
        source_exhausted = False
        first_failure_index: int | None = None

        pending: dict[Any, tuple[int, str]] = {}
        buffered_results: dict[int, CachedNodeRunResult] = {}
        buffered_failures: dict[int, tuple[str, str]] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while not stop_submitting and not source_exhausted and len(pending) < in_flight_limit:
                    try:
                        case = next(case_iter)
                    except StopIteration:
                        source_exhausted = True
                        break

                    idx = submit_index
                    submit_index += 1
                    case_id = _case_id(case)
                    future = pool.submit(
                        self.run_case,
                        case=case,
                        node_name=node_name,
                        force=force,
                        execution_mode=execution_mode,
                    )
                    pending[future] = (idx, case_id)

                if not pending:
                    break

                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, case_id = pending.pop(future)
                    try:
                        buffered_results[idx] = future.result()
                    except Exception as exc:
                        buffered_failures[idx] = (case_id, str(exc))
                        if first_failure_index is None:
                            first_failure_index = idx
                        if not continue_on_error:
                            stop_submitting = True

                while True:
                    if emit_index in buffered_results:
                        result = buffered_results.pop(emit_index)
                        yield CaseRunOutcome(index=emit_index, case_id=result.case_id, result=result)
                        emit_index += 1
                        continue

                    if emit_index in buffered_failures:
                        case_id, error = buffered_failures.pop(emit_index)
                        yield CaseRunOutcome(index=emit_index, case_id=case_id, error=error)
                        emit_index += 1

                        if not continue_on_error and first_failure_index is not None and emit_index > first_failure_index:
                            for pending_future in pending:
                                pending_future.cancel()
                            return
                        continue

                    break

            while True:
                if emit_index in buffered_results:
                    result = buffered_results.pop(emit_index)
                    yield CaseRunOutcome(index=emit_index, case_id=result.case_id, result=result)
                    emit_index += 1
                    continue

                if emit_index in buffered_failures:
                    case_id, error = buffered_failures.pop(emit_index)
                    yield CaseRunOutcome(index=emit_index, case_id=case_id, error=error)
                    emit_index += 1

                    if not continue_on_error:
                        return
                    continue

                break
