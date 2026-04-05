# CLI Code Flow

This document traces the full execution path for the two CLI commands: `lumiseval estimate` and `lumiseval run`.

For the cross-cutting question of what runs sequentially vs in parallel across CLI and API paths, see `execution-model.md`.

---

## Entry Point

```
apps/lumiseval-cli/lumiseval_cli/main.py
```

Both commands are registered with a [Typer](https://typer.tiangolo.com/) app. Running `lumiseval estimate <node>` or `lumiseval run <node>` dispatches to the corresponding decorated function.

---

## `lumiseval estimate <node> --input <source>`

Scans the dataset, builds a cache-aware execution plan, and computes the **uncached cost** for the target branch — without running any LLM calls.

```
estimate()                               main.py
  │
  ├── _resolve_target_node(node_name)    validates against NODES_BY_NAME
  │
  ├── _load_cases(...)                   loads EvalCase list from dataset
  │     └── create_dataset_adapter()    lumiseval_ingest — picks local or HF adapter
  │           └── adapter.iter_cases()  yields EvalCase objects
  │
  ├── _build_job_config(...)             builds EvalJobConfig (model, flags, thresholds)
  │
  ├── CachedNodeRunner(cache_store)      node_runner.py — wraps CacheStore
  │
  └── estimate_preflight(...)            preflight.py
        │
        ├── scan_cases(cases)            scanner.py
        │     └── Scanner.build()
        │           ├── per-case: token counts (tiktoken cl100k_base)
        │           ├── per-case: chunk count, claim estimate
        │           ├── per-case: _node_eligibility() using NodeSpec flags
        │           └── returns InputMetadata (aggregated counts + per-record metas)
        │
        ├── runner.plan_dataset(...)     node_runner.py
        │     └── for each case: plan_case()
        │           ├── build_initial_state()
        │           ├── _compute_hashes() → case_hash + config_hash
        │           ├── _plan_nodes() → prerequisites + target node
        │           └── per step: is_case_eligible? → cached? → to_run?
        │           returns DatasetNodePlan (per-node: to_run / cached / skipped counts)
        │
        └── _estimate_delta_cost_report(...)  preflight.py
              └── for each node in NODE_ORDER:
                    ├── isolate cases that are "to_run" (not cached, not skipped)
                    ├── scan_cases(subset) → subset InputMetadata
                    ├── estimator._estimate_node(node, subset_meta) → NodeCostBreakdown
                    └── estimator.estimate(..., overrides) → CostReport

  Then CLI prints three tables + final cost line:
    _print_scan_table(metadata)
    _print_node_eligibility_table(metadata)
    _print_execution_plan_table(plan, total_records)
    _print_cost_table(cost_report, ...)
```

### Key data structures produced

| Object | Type | Where |
|---|---|---|
| `cases` | `list[EvalCase]` | lumiseval-ingest adapter |
| `job_config` | `EvalJobConfig` | main.py |
| `preflight.metadata` | `InputMetadata` | scanner.py |
| `preflight.plan` | `DatasetNodePlan` | node_runner.py |
| `preflight.cost_report` | `CostReport` | cost_estimator.py |

---

## `lumiseval run <node> --input <source>`

Executes all nodes up to and including the target node for each case. Uses cache to skip already-computed steps.

```
run()                                    main.py
  │
  ├── _resolve_target_node(node_name)    validates against NODES_BY_NAME
  ├── _load_cases(...)                   same as estimate
  ├── _build_job_config(...)             same as estimate
  ├── CachedNodeRunner(cache_store)      same as estimate
  │
  └── for case in cases:
        runner.run_case(case, node_name, job_config, force)   node_runner.py
          │
          ├── build_initial_state()      graph.py — builds the mutable state dict
          ├── _compute_hashes()          case_hash + config_hash (cache key)
          ├── _plan_nodes()              prerequisites list + target node
          │
          └── step loop over planned nodes:
                │
                ├── [eval target + metric nodes] → parallel fan-out
                │     ├── ThreadPoolExecutor over METRIC_NODES
                │     ├── each metric: eligibility check → cache check → NODE_FNS[m](state)
                │     └── results merged back into state in stable NODE_ORDER
                │
                └── [all other nodes] → sequential
                      ├── is_case_eligible_for_node? → skipped_output (cached as $0)
                      ├── cache hit?  → load cached output → state.update()
                      └── cache miss → NODE_FNS[step](state) → cache.put() → state.update()

  Returns CachedNodeRunResult:
    executed_nodes, cached_nodes, node_output, final_state, elapsed_ms

  Optionally writes final_state["report"] as JSON to --output-dir
```

### Node execution: `NODE_FNS[step](state)`

Every node function lives in `lumiseval_graph/nodes/` and is registered in `registry.py`:

| Node | Function | Purpose |
|---|---|---|
| `scan` | `node_metadata_scanner` | Tokenize, chunk-count, build `InputMetadata` |
| `chunk` | `node_chunk` | Split generation into `Chunk` objects |
| `claims` | `node_claims` | LLM: extract factual claims from chunks |
| `dedupe` | `node_dedupe` | MMR-based deduplication of extracted claims |
| `geval_steps` | `node_geval_steps` | LLM: generate reusable evaluation steps for criteria-only GEval metrics |
| `relevance` | `node_relevance` | LLM metric: answer relevancy vs. question |
| `grounding` | `node_grounding` | LLM metric: faithfulness vs. context passages |
| `redteam` | `node_adversarial` | LLM metric: safety / red-team evaluation |
| `geval` | `node_geval` | LLM metric: custom GEval scoring |
| `reference` | `node_reference` | ROUGE/BLEU/METEOR vs. reference answer |
| `eval` | `node_eval` | Aggregate all metric results into `EvalReport` |

---

## Supporting systems

### Pipeline topology (`lumiseval_core/pipeline.py`)

Single source of truth. Each `NodeSpec` declares:

- `prerequisites` — which nodes must have run first
- `requires_generation / requires_question / requires_context / requires_geval / requires_reference` — eligibility gates
- `is_metric` — whether the node participates in the parallel fan-out
- `skip_output` — state patch applied when a node is skipped
- `env_key_suffixes` — env var prefixes for LLM model selection

Three derived constants are computed at import time:

```python
NODES_BY_NAME  # dict[str, NodeSpec] — O(1) lookup
NODE_ORDER     # list[str]           — stable iteration order
METRIC_NODES   # list[str]           — parallel fan-out group
```

### Cache (`lumiseval_core/cache.py`)

`CacheStore` is a filesystem-backed key-value store:

- **Key**: `(case_hash, config_hash, node_name)`
- **case_hash**: SHA-256 of `generation + question + reference + context + geval`
- **config_hash**: SHA-256 of `EvalJobConfig` fields

Each cache entry stores `node_output` + `node_cost` (a `NodeCostBreakdown`).

`NoOpCacheStore` is used when `--no-cache` is passed; all reads miss and writes are discarded.

### Dataset adapters (`lumiseval_ingest/`)

`create_dataset_adapter(source, adapter, ...)` auto-detects:

- **local** — JSON/JSONL/CSV files; field mapping via schema inference
- **huggingface** — `hf://<dataset-id>` URIs; uses `datasets` library

Both yield `EvalCase` objects with a normalized schema.

### Scanner (`lumiseval_ingest/scanner.py`)

`scan_cases(cases)` builds `InputMetadata` without calling any LLM:

1. Tokenizes each field with `tiktoken` (`cl100k_base` encoding)
2. Chunks the generation text to estimate claim count
3. Calls `_node_eligibility()` — checks each `NodeSpec`'s eligibility flags against the case fields
4. Aggregates per-record `RecordMeta` into dataset-level token counts and `CostMetadata`

### Cost estimator (`lumiseval_graph/nodes/cost_estimator.py`)

`CostEstimator.estimate(metadata, job_config)` produces a `CostReport` with one `CostRow` per node:

- Formula-based: `input_tokens × calls × price_per_token`
- Cumulative: each row shows total cost up to that node
- `CostReport.row(node_name)` gives the delta cost for a specific branch

---

## Data flow summary

```
EvalCase list
     │
     ▼
  scan_cases()  ──────────────────► InputMetadata
     │                                    │
     ▼                                    ▼
plan_dataset()  ──────────────────► DatasetNodePlan
                                          │
                                          ▼
                              _estimate_delta_cost_report()
                                          │
                                          ▼
                                      CostReport          ← estimate command stops here


  run_case()  (per case)
     │
     ├── sequential nodes: scan → chunk → claims → dedupe
     │
     ├── parallel metric nodes for `eval` target only:
     │     relevance ║ grounding ║ redteam ║ geval ║ reference
     │
     └── eval: aggregate → EvalReport
```
