# Runner Observability PRD (Langfuse, Custom Runner)

## 1. Purpose

Define an implementation-ready observability design for the current `CachedNodeRunner` execution path so that:

- Every case run has a trace.
- Every node has observable metrics, including cache-hit paths.
- Parallel metric execution keeps parent/child trace continuity.
- The design remains compatible with current custom runner + node-level cache strategy.
- The same telemetry contract can be reused if/when an optional LangGraph executor is added.

This document is intentionally decision-complete so it can be implemented directly later.

## 2. Current State

- Node functions are decorated with `@observe(...)` in `graph.py`.
- CLI execution does **not** use `build_graph()` runtime; it uses `CachedNodeRunner` and direct `NODE_FNS` invocation.
- Executed nodes can be observed via decorators.
- Cache-hit nodes are currently **not** observed, because node functions are skipped on cache hit.
- Langfuse is optional and enabled only when `LANGFUSE_SECRET_KEY` is present (`observability.py`).
- Runner has parallelism at:
  - record level (`run_cases_iter`, thread pool)
  - metric fan-out level inside one case (`eval` branch)

## 3. Product Goals

### 3.1 Primary Goals

1. Provide per-case, per-node observability for all outcomes:
   - executed
   - cache_hit
   - failed
   - retried (future-ready, even if retries are introduced later)
2. Preserve current behavior:
   - cache semantics and cache key strategy
   - deterministic output order
   - current branch strategy and node dependency semantics
3. Keep overhead small and safe when Langfuse is disabled (no-op behavior).

### 3.2 Non-Goals (v1)

- Switching default executor from custom runner to LangGraph runtime.
- Building dashboard UI in repo.
- Implementing full distributed tracing backend abstraction beyond Langfuse.
- Changing cache key algorithm or cache storage layout.

## 4. Success Criteria

1. For one case run to `target_node`, Langfuse shows one root trace and child records for each planned node outcome.
2. Cache hits are visible as node-level records (not missing).
3. Failures include node name, case_id, and normalized error type.
4. Parallel metric node records remain associated with the correct case trace.
5. When Langfuse is disabled, execution behavior is unchanged and overhead is negligible.

## 5. Telemetry Contract (Canonical)

### 5.1 Root Trace (per case)

Name: `case_run`

Required metadata:

- `executor`: `"custom_runner"`
- `case_id`
- `target_node`
- `execution_mode`
- `force_cache_bypass` (bool)
- `max_workers`
- `max_in_flight`

Completion metadata:

- `elapsed_ms`
- `executed_nodes_count`
- `cached_nodes_count`
- `failed` (bool)
- `failure_node` (nullable)
- `failure_type` (nullable)

### 5.2 Node Record (per planned node)

Name: `node_step`

Required metadata:

- `case_id`
- `target_node`
- `node_name`
- `status`: `executed | cache_hit | failed`
- `execution_mode`
- `cache_key` (for executed/cache_hit paths)
- `attempt` (starts at 1)
- `max_attempts`
- `duration_ms`

Optional metadata:

- `error_type`
- `error_message`
- `llm_model`
- `llm_fallback_model`
- `temperature`
- `thread_name`

### 5.3 Retry Event (future-ready)

Name: `node_retry`

Metadata:

- `case_id`
- `node_name`
- `attempt`
- `backoff_ms`
- `error_type`
- `retryable` (bool)

## 6. Technical Design

### 6.1 Instrumentation Strategy

Add observability at **runner orchestration boundaries**, not only inside graph node functions.

Reason:

- `@observe` on node functions only covers executed nodes.
- Cache-hit paths bypass node function execution and currently have no telemetry.

### 6.2 Implementation Components

1. `observability.py` additions:
   - `observe_case_run(...)` helper (decorated function wrapper).
   - `observe_node_step(...)` helper for all node statuses.
   - `observe_node_retry(...)` helper (used once retry feature lands).
   - Helpers must be safe no-ops when Langfuse is disabled.

2. `runner.py` changes:
   - Wrap `run_case` with root case trace scope.
   - Emit `node_step` for:
     - cache hit path
     - executed success path
     - executed failure path
   - Include cache key and route metadata in event payload.
   - Preserve existing caching behavior (no cache writes on failure).

3. Thread context propagation:
   - In both thread pools (`run_cases_iter` and metric fan-out), propagate contextvars using `copy_context()`.
   - Submit tasks as `pool.submit(ctx.run, fn, ...)` to keep trace linkage stable.

4. Retry compatibility seam:
   - Introduce single execution helper:
     - `_execute_node_with_policy(node_name, state, cache_key, policy)`
   - v1 can use policy = single attempt.
   - Retry feature later plugs into same helper without changing telemetry contract.

### 6.3 Node Status Semantics

- `cache_hit`: node output loaded from cache, state updated, node fn not called.
- `executed`: node fn called and returned successfully.
- `failed`: node fn raised and execution stops per fail policy.

### 6.4 Error Normalization

Add runner-level utility:

- `normalize_error(exc) -> {error_type, error_message, retryable}`

Initial retryable classification:

- retryable: timeout, rate limit, transient network, provider 5xx
- non-retryable: validation/schema parse, logic errors, explicit user/data errors

## 7. File-Level Change Plan

### 7.1 `packages/lumiseval-graph/lumiseval_graph/observability.py`

- Add runner-oriented helper APIs:
  - `record_case_start(...)`
  - `record_case_end(...)`
  - `record_node_step(...)`
  - `record_node_retry(...)`
- Keep existing `observe` behavior and no-op fallback.

### 7.2 `packages/lumiseval-graph/lumiseval_graph/runner.py`

- Instrument `run_case` lifecycle:
  - case start
  - node-level step results
  - case end
- Add context propagation in both threadpool usage points.
- Add normalized error metadata for failures.
- Keep existing cache key/fingerprint logic unchanged.

### 7.3 `apps/lumiseval-cli/lumiseval_cli/main.py`

- Optional: pass lightweight run metadata into runner for root trace tags:
  - command mode
  - selected split
  - user provided concurrency settings

No CLI UX flag change required for v1.

## 8. Test Plan

### 8.1 Unit Tests

1. `Langfuse disabled`:
   - observability helpers no-op and do not throw.
2. `cache_hit telemetry`:
   - when cache returns value, `record_node_step(status=cache_hit)` called.
3. `executed telemetry`:
   - successful node execution emits `status=executed` with duration.
4. `failure telemetry`:
   - failing node emits `status=failed` with normalized error.
5. `parallel context`:
   - in metric fan-out, child node records map to same case trace id (integration-style test with mocked client).

### 8.2 Integration Tests

1. Run small dataset with mixed cache-hit and execution paths:
   - observed node count equals planned node count per case.
2. Run with `max_workers > 1` and `max_in_flight > 1`:
   - traces remain per-case correct.
3. Fail-fast mode:
   - first failure recorded with node metadata, run terminates per policy.

## 9. Performance / Safety Requirements

- Observability must never break core execution if provider/sdk fails.
- If telemetry submission errors happen, swallow and log at debug level.
- Added latency target:
  - < 3% median overhead with Langfuse enabled on local benchmark.
- No additional memory growth beyond existing bounded in-flight controls.

## 10. Rollout Plan

Phase 1:

- Add runner-level executed/cache-hit/failed node records.
- Validate no-op behavior without Langfuse.

Phase 2:

- Add context propagation for thread pools and verify trace continuity.

Phase 3:

- Add retry instrumentation hooks (policy can still be single-attempt).

Phase 4:

- Enable dashboards/queries in Langfuse using stable field names from this PRD.

## 11. Future Compatibility (Optional LangGraph Executor)

If a LangGraph runtime executor is added later:

- Keep **same telemetry contract** (`case_run`, `node_step`, `node_retry`).
- Implement executor-specific adapters that emit the same metadata fields.
- This keeps dashboards and alert rules unchanged across executors.

## 12. Final Decision

Proceed with runner-level observability on the custom runner now.  
Do not block on LangGraph runtime migration for node metrics.

