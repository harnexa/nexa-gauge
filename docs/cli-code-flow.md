# CLI Code Flow

This document maps the current CLI behavior to implementation.

Primary source files:
- `apps/nexagauge-apps/ng_cli/main.py`
- `apps/nexagauge-apps/ng_cli/run.py`
- `apps/nexagauge-apps/ng_cli/estimate.py`
- `apps/nexagauge-apps/ng_cli/cache.py`
- `packages/nexagauge-graph/ng_graph/runner.py`

## Command Entry

`nexagauge` exposes three top-level command entries:
- `nexagauge run <node_name>`
- `nexagauge estimate <node_name>`
- `nexagauge cache <command>`

Shared setup in both commands:
1. Validate target node name against topology.
2. Resolve global/per-node model routing.
3. Build cache store (`CacheStore` or `NoOpCacheStore`).
4. Resolve dataset adapter and stream selected rows.
5. Inject per-case runtime routing config (LLM overrides + `chunker`/`refiner`/`refiner_top_k`).

## `run` Flow

`run()` calls `CachedNodeRunner.run_cases_iter(..., execution_mode="run")`.

Per outcome:
- success: accumulate executed/cached step counters.
- success + `--output-dir`: write one `<case_id>.json` report file when `final_state["report"]` is present.
- failure: collect `(case_id, error)` and continue or exit based on `--continue-on-error`.

Behavior notes:
- `--debug` enables per-node debug logs.
- When `--debug` is off, a CLI progress bar is shown.
- `--web-search` and `--evidence-threshold` are currently accepted for compatibility but not used by current `run` implementation.

## `estimate` Flow

`estimate()` calls `CachedNodeRunner.run_cases_iter(..., execution_mode="estimate")`.

Per outcome:
- aggregate `estimated_costs` by node.
- aggregate node stats (executed, cached, estimated, uncached-eligible).
- render branch table + total estimate.

Behavior notes:
- `--debug` enables per-node debug logs.
- When `--debug` is off, a CLI progress bar is shown.

## `cache` Flow

`cache` exposes cache-management commands:
- `nexagauge cache dir` prints the resolved cache root.
- `nexagauge cache delete` wipes the node-level execution cache.

Cache root resolution:
1. `--cache-dir` for `cache delete`, when provided.
2. `NEXAGAUGE_CACHE_DIR`, when set.
3. The per-user platform default from `default_cache_dir()`.

## Runner Responsibilities

`CachedNodeRunner` is the CLI-to-graph bridge.

Key responsibilities:
- Build initial `EvalCase` state from each input row.
- Expand target plan from topology prerequisites.
- Read/write node-level cache entries.
- Reuse `run` cache entries from `estimate` mode when route matches.
- Execute metric nodes in parallel for `target=eval`.
- Stream outcomes in submission order.

Planning note:
- For non-`report` targets, runner appends `report` to the execution plan so final state can emit report JSON consistently.

## Adapter Resolution

`create_dataset_adapter(...)` behavior:
- `adapter=local` -> local file adapter.
- `adapter=huggingface` -> Hugging Face adapter.
- `adapter=auto`:
  - `hf://...` input -> Hugging Face adapter.
  - existing local path -> local adapter.
  - otherwise -> input parse error.
