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
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from typing import Any, Iterable, Iterator, Mapping

from ng_core.cache import (
    NodeCacheBackend,
    build_node_cache_key,
    cache_read_allowed,
    cache_write_allowed,
    compute_case_hash,
)
from ng_core.config import config as cfg
from ng_core.types import EvalCase
from pydantic import BaseModel

from ng_graph.llm.config import get_node_config
from ng_graph.log import get_node_logger
from ng_graph.registry import NODE_FNS
from ng_graph.topology import (
    DEBUG_SKIP_NODES,
    METRIC_NODES,
    transitive_prerequisites,
)


def _debug_log_running(node_name: str, case_id: str) -> None:
    if node_name in DEBUG_SKIP_NODES:
        return
    get_node_logger(node_name).start(f"Running for case={case_id}")


class CachedNodeRunResult(BaseModel):
    """Outcome of running a single target node (and its prerequisites) for one case.

    Produced by :meth:`CachedNodeRunner.run_case` and consumed by the CLI
    (`nexagauge run …`) plus any callable that wraps the runner — notably the
    adapter in ``nexagauge`` that writes per-case JSON into ``--output-dir``.

    Fields:
        node_name: Target node requested by the caller (e.g. ``"geval_steps"``).
        case_id: Stable identifier pulled from the input record.
        executed_nodes: Nodes actually run for this case (cache miss).
        cached_nodes: Nodes served from the node-level cache (cache hit).
        node_output: Patch emitted by the *target* node only — what downstream
            reporting typically cares about when the target isn't ``report``.
        final_state: Merged :class:`EvalCase` state after the full plan ran;
            report aggregation reads from this.
        elapsed_ms: Wall-clock for the full per-case plan.
    """

    node_name: str
    case_id: str
    executed_nodes: list[str]
    cached_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]
    elapsed_ms: float


class BatchRunResult(BaseModel):
    """Aggregated result shape for a whole-batch (non-streaming) execution.

    Retained as a convenience container for callers that collect all outcomes
    before returning. The streaming path (:meth:`CachedNodeRunner.run_cases_iter`)
    yields :class:`CaseRunOutcome` instances instead and is preferred for CLI
    progress reporting.

    Fields:
        results: Successful per-case results in submission order.
        failures: ``(case_id, traceback_text)`` pairs for cases that raised.
    """

    results: list[CachedNodeRunResult]
    failures: list[tuple[str, str]]


class CaseRunOutcome(BaseModel):
    """One element yielded by :meth:`CachedNodeRunner.run_cases_iter`.

    Guarantees output in *submission order* even when workers finish out of
    order — the runner buffers completions and releases them contiguously from
    ``emit_index``. Exactly one of ``result`` / ``error`` is set.

    Fields:
        index: Zero-based position within the input iterable.
        case_id: Resolved case identifier (``"unknown-case"`` if missing).
        result: Populated on success.
        error: Formatted ``str(exc) + traceback`` on failure.
    """

    index: int
    case_id: str
    result: CachedNodeRunResult | None = None
    error: str | None = None


def _case_value(case: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a case whether it's a dict or an attribute-bearing object.

    The CLI feeds dicts loaded from JSON; tests and programmatic callers
    occasionally pass Pydantic models. One accessor keeps the rest of this
    module agnostic to that.
    """
    if isinstance(case, dict):
        return case.get(key, default)
    return getattr(case, key, default)


def _case_id(case: Any) -> str:
    """Return a non-empty, stripped case id, falling back to ``"unknown-case"``.

    Used for cache keys, log lines, and :class:`CaseRunOutcome` reporting, so
    it must never be the empty string.
    """
    value = _case_value(case, "case_id", "")
    text = str(value).strip()
    return text or "unknown-case"


def _stable_json(obj: Any) -> str:
    """Deterministic JSON encoding for hashing (sorted keys, compact, Pydantic-aware).

    Pydantic models are unwrapped via ``model_dump``/``dict`` so two
    structurally-equal states always produce byte-identical output — the whole
    basis of the node-level cache.
    """
    def _default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return str(value)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _build_initial_state(case: dict[str, Any], *, execution_mode: str, target_node: str) -> EvalCase:
    """Construct the initial :class:`EvalCase` state from a raw input record.

    This is the seed state every plan node receives; node outputs are merged in
    via :func:`_merge_state_patch`. Only the fields the scanner and downstream
    nodes actually read are extracted — extra keys on the input record are
    ignored by design.

    Called once per case at the top of :meth:`CachedNodeRunner.run_case`.
    """
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
        target_node=target_node,
        execution_mode=execution_mode,
        estimated_costs={},
        node_model_usage={},
        reference_files=_case_value(case, "reference_files") or [],
    )


def _compute_case_fingerprint(case: dict[str, Any]) -> str:
    """Hash the *input* of a case — the root of every cache key for that case.

    Excludes run-time concerns (target node, execution_mode, model routing);
    those enter via :func:`_step_fingerprint` further down. The delegate lives
    in ``ng_core.cache`` so core and graph packages agree on the hash.
    """
    return compute_case_hash(
        generation=_case_value(case, "generation", ""),
        question=_case_value(case, "question"),
        reference=_case_value(case, "reference"),
        geval=_case_value(case, "geval"),
        redteam=_case_value(case, "redteam"),
        context=_case_value(case, "context") or [],
        reference_files=_case_value(case, "reference_files") or [],
    )


def _node_route_fingerprint(
    node_name: str, *, state: Mapping[str, Any], execution_mode: str
) -> str:
    """Hash the LLM routing for a node — model, fallback, temperature, mode.

    This is the "did the model selection change?" component of a cache key.
    Swapping ``gpt-4o-mini`` for ``gpt-4o`` (or flipping ``run``↔``estimate``)
    must invalidate cached outputs; changing an unrelated node's routing must
    not. Called from :func:`_step_fingerprint`.
    """
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
    """Chain ``parent_fingerprint`` with this node's identity + routing.

    Produces the path-dependent fingerprint used to build this step's cache
    key. Chaining guarantees that a node's cache entry is only valid when the
    *entire upstream path* that produced its inputs is identical. Called once
    per plan step inside :meth:`CachedNodeRunner.run_case`.
    """
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
    """Assemble the opaque backend cache key for a single step execution.

    Delegates to ``ng_core.cache.build_node_cache_key`` so the exact
    key format stays in one place and is backend-agnostic (disk today,
    potentially Redis/Dynamo later).
    """
    return build_node_cache_key(
        case_fingerprint=case_fingerprint,
        node_name=node_name,
        execution_mode=execution_mode,
        node_route_fingerprint=step_fingerprint,
    )


def _step_fingerprint_for_node_branch(
    *,
    case_fingerprint: str,
    node_name: str,
    state: Mapping[str, Any],
    execution_mode: str,
) -> str:
    """Compute a node's fingerprint from *its own* prerequisite branch only.

    Used for the parallel metrics group in :meth:`CachedNodeRunner.run_case`
    (the ``eval`` target fans out to N metric nodes concurrently). Each metric
    must key itself against its intrinsic branch — not the order the scheduler
    happened to fan them out in — so ``run grounding`` and ``estimate eval``
    hit the same cache entry for grounding.
    """
    parent_fingerprint = case_fingerprint
    for prereq in transitive_prerequisites(node_name):
        parent_fingerprint = _step_fingerprint(
            parent_fingerprint=parent_fingerprint,
            node_name=prereq,
            state=state,
            execution_mode=execution_mode,
        )
    return _step_fingerprint(
        parent_fingerprint=parent_fingerprint,
        node_name=node_name,
        state=state,
        execution_mode=execution_mode,
    )


def _plan_nodes(node_name: str) -> list[str]:
    """Build the ordered list of nodes to execute for a requested target.

    Returns ``[*prerequisites, target, "eval", "report"]``. Every plan funnels
    through ``eval`` before ``report``: utility and metric nodes never reach
    ``report`` directly — ``eval`` is the sole subscriber of ``report``. This
    keeps the invariant that ``report.prerequisites == ("eval",)`` holds at
    runtime regardless of which target the CLI was invoked with.
    """
    plan = list(transitive_prerequisites(node_name)) + [node_name]
    if node_name not in ("eval", "report") and "eval" in NODE_FNS and "eval" not in plan:
        plan.append("eval")
    if node_name != "report" and "report" in NODE_FNS and "report" not in plan:
        plan.append("report")
    return plan


def _merge_state_patch(state: dict[str, Any], patch: Mapping[str, Any]) -> None:
    """Apply a node's output patch into ``state`` in place.

    Most keys are overwritten wholesale — node outputs are authoritative for
    the artifact they produce. Two keys are accumulators and get *shallow
    merged* so later nodes don't clobber earlier nodes' entries:

    - ``estimated_costs`` — per-node cost estimates, keyed by node name.
    - ``node_model_usage`` — per-node model usage tallies for fallback tracking.
    """
    for key, value in patch.items():
        if key == "estimated_costs" and isinstance(value, Mapping):
            existing = state.get("estimated_costs")
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update(dict(value))
            state["estimated_costs"] = merged
            continue
        if key == "node_model_usage" and isinstance(value, Mapping):
            existing = state.get("node_model_usage")
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update(dict(value))
            state["node_model_usage"] = merged
            continue
        state[key] = value


class CachedNodeRunner:
    """Execute a target node (plus its prerequisites + report) with per-node caching.

    This is the engine behind every ``nexagauge run <node>`` / ``nexagauge estimate <node>``
    invocation. Responsibilities:

    1. Expand a target into an ordered plan via :func:`_plan_nodes`.
    2. For each plan step, compute a path-chained fingerprint and look up a
       cached output; on hit, merge the cached patch into state and skip execution.
    3. On miss, invoke the node function from :data:`NODE_FNS`, merge its
       patch, and (when policy allows) write the output back to the cache.
    4. Fan out the metric group in parallel when the target is ``eval``.
    5. Provide ordered streaming over many cases via :meth:`run_cases_iter`.

    The backend is injected (:class:`NodeCacheBackend`) so tests can use an
    in-memory store while production uses the disk-backed implementation.
    """

    def __init__(self, cache_store: NodeCacheBackend) -> None:
        """Store the cache backend. Does no I/O."""
        self._cache = cache_store

    @staticmethod
    def _normalize_max_in_flight(*, max_workers: int, max_in_flight: int | None) -> int:
        """Pick a sane in-flight cap for streaming execution.

        Default is ``2 × max_workers`` so the pool stays fed while the consumer
        drains ordered results. Never lets the cap dip below ``max_workers``
        (which would starve the pool). Called once at the start of
        :meth:`run_cases_iter`.
        """
        workers = max(1, max_workers)
        if max_in_flight is None:
            return max(1, workers * 2)
        return max(workers, max_in_flight)

    def _read_step_cache(self, cache_key: str) -> dict[str, Any] | None:
        """Fetch a raw cached node output by opaque key, or ``None`` on miss.

        The backend stores ``{"node_output": ..., "metadata": ...}``; this
        returns only the output payload since the runner doesn't need metadata
        on read.
        """
        entry = self._cache.get_entry_by_key(cache_key)
        if entry is None:
            return None
        return entry["node_output"]

    def _read_step_cache_with_run_fallback(
        self,
        *,
        case_fingerprint: str,
        node_name: str,
        estimate_step_fingerprint: str,
        run_step_fingerprint: str | None,
        execution_mode: str,
        force: bool,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Look up a cached node output, with cross-mode fallback for ``estimate``.

        Precedence:

        1. Honor ``force`` and per-node/per-mode read policy. If reads are
           forbidden, return ``(None, None)`` immediately.
        2. Try the primary key (same ``execution_mode`` as the request).
        3. Only when the caller is running ``estimate`` and has supplied a
           ``run`` fingerprint, fall back to the matching ``run``-mode entry —
           a prior real run is a valid substitute for an estimate that would
           otherwise have to re-execute.

        Returns ``(output, source_mode)`` where ``source_mode`` is the mode
        the hit came from (``"run"`` or ``"estimate"``), or ``(None, None)``
        on miss.
        """
        if force:
            return None, None
        if not cache_read_allowed(execution_mode=execution_mode, node_name=node_name):
            return None, None

        primary_key = _cache_key_for_step(
            case_fingerprint=case_fingerprint,
            node_name=node_name,
            step_fingerprint=estimate_step_fingerprint,
            execution_mode=execution_mode,
        )
        cached_output = self._read_step_cache(primary_key)
        if cached_output is not None:
            return cached_output, execution_mode

        # Estimate mode can reuse prior run-mode node outputs for identical routes.
        if execution_mode == "estimate" and run_step_fingerprint is not None:
            run_key = _cache_key_for_step(
                case_fingerprint=case_fingerprint,
                node_name=node_name,
                step_fingerprint=run_step_fingerprint,
                execution_mode="run",
            )
            cached_output = self._read_step_cache(run_key)
            if cached_output is not None:
                return cached_output, "run"

        return None, None

    def _write_step_cache(
        self,
        *,
        cache_key: str,
        node_name: str,
        output: dict[str, Any],
        execution_mode: str,
        case_fingerprint: str,
    ) -> None:
        """Persist a freshly-computed node output under its cache key.

        Only called when :func:`cache_write_allowed` says this ``(mode, node)``
        pair is writable. Metadata stamps the entry with the schema version
        (``"v2"``) so future migrations can detect legacy payloads.
        """
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
        debug: bool = False,
    ) -> CachedNodeRunResult:
        """Execute the full plan for one case and return a :class:`CachedNodeRunResult`.

        This is the central per-case entry point. High-level shape:

        - Resolve target and build initial :class:`EvalCase` state.
        - Compute the case fingerprint (input hash) that roots every step key.
        - Walk the plan linearly, except when the target is ``eval``: at the
          start of the metric group, fan out all :data:`METRIC_NODES` across a
          :class:`ThreadPoolExecutor` on deep-copied state snapshots (avoids
          shared-mutation races on nested Pydantic objects), collect outputs
          in a stable order, merge them, then advance ``i`` past the group.
        - Each non-parallel step chains its fingerprint off the current
          ``path_fingerprint``, consults the cache (with run-mode fallback for
          estimate), executes on miss, merges the patch, and records timing.

        Parameters:
            case: Raw input record (dict, typically parsed from sample JSON).
            node_name: Target node. Must be a key in :data:`NODE_FNS`.
            force: Bypass all cache reads; still writes when policy allows.
            execution_mode: ``"run"`` for real execution, ``"estimate"`` for
                cost-only dry runs. Affects routing, cache namespacing, and
                cross-mode fallback.

        Raises:
            ValueError: If ``node_name`` isn't registered.
            Exception: Any exception raised by a node function propagates.
                :meth:`run_cases_iter` captures it as a :class:`CaseRunOutcome`.
        """
        if node_name not in NODE_FNS:
            valid = ", ".join(sorted(NODE_FNS))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        t0 = time.monotonic()
        case_id_for_log = _case_id(case)

        state: EvalCase = _build_initial_state(
            case,
            execution_mode=execution_mode,
            target_node=node_name,
        )
        # Under-dunder: runner-internal plumbing that nodes may read but never
        # emit in their output patches. Lets `GevalStepsNode` reuse the
        # runner's cache policy (e.g. NoOpCacheStore when --no-cache is set).
        state["__cache_store"] = self._cache

        case_fingerprint = _compute_case_fingerprint(case)

        plan = _plan_nodes(node_name)
        path_fingerprint = case_fingerprint
        run_path_fingerprint = case_fingerprint

        executed: list[str] = []
        cached: list[str] = []
        node_output: dict[str, Any] = {}

        i = 0
        while i < len(plan):
            step = plan[i]

            if node_name == "eval" and METRIC_NODES and step == METRIC_NODES[0]:
                cached_group_outputs: dict[str, dict[str, Any]] = {}
                metric_fingerprints: dict[str, str] = {}
                run_metric_fingerprints: dict[str, str] = {}
                to_run: list[tuple[str, str, str]] = []
                group_parent_fingerprint = path_fingerprint
                run_group_parent_fingerprint = run_path_fingerprint

                for metric_step in METRIC_NODES:
                    metric_fingerprint = _step_fingerprint_for_node_branch(
                        case_fingerprint=case_fingerprint,
                        node_name=metric_step,
                        state=state,
                        execution_mode=execution_mode,
                    )
                    metric_fingerprints[metric_step] = metric_fingerprint
                    run_metric_fingerprint = _step_fingerprint_for_node_branch(
                        case_fingerprint=case_fingerprint,
                        node_name=metric_step,
                        state=state,
                        execution_mode="run",
                    )
                    run_metric_fingerprints[metric_step] = run_metric_fingerprint
                    cached_output, _ = self._read_step_cache_with_run_fallback(
                        case_fingerprint=case_fingerprint,
                        node_name=metric_step,
                        estimate_step_fingerprint=metric_fingerprint,
                        run_step_fingerprint=run_metric_fingerprint,
                        execution_mode=execution_mode,
                        force=force,
                    )
                    metric_key = _cache_key_for_step(
                        case_fingerprint=case_fingerprint,
                        node_name=metric_step,
                        step_fingerprint=metric_fingerprint,
                        execution_mode=execution_mode,
                    )
                    if cached_output is not None:
                        cached_group_outputs[metric_step] = cached_output
                        cached.append(metric_step)
                        continue
                    to_run.append((metric_step, metric_key, metric_fingerprint))

                run_group_outputs: dict[str, dict[str, Any]] = {}
                if to_run:
                    if debug:
                        for metric_step, _, _ in to_run:
                            _debug_log_running(metric_step, case_id_for_log)
                    with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
                        # Use an isolated snapshot per metric task to avoid shared
                        # nested object mutation races in parallel execution.
                        futures = {
                            pool.submit(NODE_FNS[metric_step], deepcopy(state)): (
                                metric_step,
                                metric_key,
                            )
                            for metric_step, metric_key, _ in to_run
                        }
                        for future in as_completed(futures):
                            metric_step, metric_key = futures[future]
                            output = future.result()
                            run_group_outputs[metric_step] = output
                            executed.append(metric_step)
                            if cache_write_allowed(
                                execution_mode=execution_mode,
                                node_name=metric_step,
                            ):
                                self._write_step_cache(
                                    cache_key=metric_key,
                                    node_name=metric_step,
                                    output=output,
                                    execution_mode=execution_mode,
                                    case_fingerprint=case_fingerprint,
                                )

                for metric_step in METRIC_NODES:
                    merged_output = cached_group_outputs.get(metric_step) or run_group_outputs.get(
                        metric_step
                    )
                    if merged_output is not None:
                        _merge_state_patch(state, merged_output)

                group_signature_raw = "|".join(
                    [
                        group_parent_fingerprint,
                        "metrics",
                        *[metric_fingerprints[n] for n in METRIC_NODES],
                    ]
                )
                path_fingerprint = hashlib.sha256(group_signature_raw.encode()).hexdigest()[:16]
                run_group_signature_raw = "|".join(
                    [
                        run_group_parent_fingerprint,
                        "metrics",
                        *[run_metric_fingerprints[n] for n in METRIC_NODES],
                    ]
                )
                run_path_fingerprint = hashlib.sha256(run_group_signature_raw.encode()).hexdigest()[
                    :16
                ]
                i += len(METRIC_NODES)
                continue

            step_fingerprint = _step_fingerprint(
                parent_fingerprint=path_fingerprint,
                node_name=step,
                state=state,
                execution_mode=execution_mode,
            )
            run_step_fingerprint = _step_fingerprint(
                parent_fingerprint=run_path_fingerprint,
                node_name=step,
                state=state,
                execution_mode="run",
            )
            cached_output, _ = self._read_step_cache_with_run_fallback(
                case_fingerprint=case_fingerprint,
                node_name=step,
                estimate_step_fingerprint=step_fingerprint,
                run_step_fingerprint=run_step_fingerprint,
                execution_mode=execution_mode,
                force=force,
            )
            cache_key = _cache_key_for_step(
                case_fingerprint=case_fingerprint,
                node_name=step,
                step_fingerprint=step_fingerprint,
                execution_mode=execution_mode,
            )
            if cached_output is not None:
                _merge_state_patch(state, cached_output)
                cached.append(step)
                if step == node_name:
                    node_output = cached_output
                path_fingerprint = step_fingerprint
                run_path_fingerprint = run_step_fingerprint
                i += 1
                continue

            if debug:
                _debug_log_running(step, case_id_for_log)
            output = NODE_FNS[step](state)
            _merge_state_patch(state, output)
            if cache_write_allowed(execution_mode=execution_mode, node_name=step):
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
            run_path_fingerprint = run_step_fingerprint
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
        debug: bool = False,
    ) -> Iterator[CaseRunOutcome]:
        """Stream per-case outcomes in *submission order* with optional parallelism.

        Used by the CLI progress bar and the output-dir writer: both need
        stable ordering so record N's file always corresponds to input row N,
        even when worker N finishes before worker N-1.

        Behavior:

        - ``max_workers == 1``: simple serial loop, yields as each case
          completes.
        - ``max_workers > 1``: submits up to ``max_in_flight`` cases to a
          thread pool, buffers out-of-order completions by index, and only
          yields when ``emit_index`` is present in the buffer. This preserves
          input order without blocking the pool.

        Parameters:
            cases: Iterable of raw input records.
            node_name, force, execution_mode: Forwarded to :meth:`run_case`.
            max_workers: Pool size; 1 disables parallelism.
            max_in_flight: Submission cap. Defaults via
                :meth:`_normalize_max_in_flight` to ``2 × workers``.
            continue_on_error: If ``False``, stops submitting after the first
                failure and cancels remaining pending futures once that
                failure has been emitted. Already-buffered earlier results
                still get yielded so output ordering stays contiguous up to
                the failure point.

        Yields:
            :class:`CaseRunOutcome` — one per case, strictly increasing
            ``index``.
        """
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
                        debug=debug,
                    )
                    yield CaseRunOutcome(index=idx, case_id=result.case_id, result=result)
                except Exception as exc:
                    yield CaseRunOutcome(
                        index=idx,
                        case_id=case_id,
                        error=f"{exc}\n{traceback.format_exc()}",
                    )
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
                while (
                    not stop_submitting and not source_exhausted and len(pending) < in_flight_limit
                ):
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
                        debug=debug,
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
                        buffered_failures[idx] = (case_id, f"{exc}\n{traceback.format_exc()}")
                        if first_failure_index is None:
                            first_failure_index = idx
                        if not continue_on_error:
                            stop_submitting = True

                while True:
                    if emit_index in buffered_results:
                        result = buffered_results.pop(emit_index)
                        yield CaseRunOutcome(
                            index=emit_index, case_id=result.case_id, result=result
                        )
                        emit_index += 1
                        continue

                    if emit_index in buffered_failures:
                        case_id, error = buffered_failures.pop(emit_index)
                        yield CaseRunOutcome(index=emit_index, case_id=case_id, error=error)
                        emit_index += 1

                        if (
                            not continue_on_error
                            and first_failure_index is not None
                            and emit_index > first_failure_index
                        ):
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
