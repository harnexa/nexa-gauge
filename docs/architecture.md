# LumisEval Architecture (Current V1 Code)

This document reflects the current implementation across:
- `apps/lumiseval-cli`
- `apps/lumiseval-api`
- `packages/lumiseval-graph`
- `packages/lumiseval-ingest`
- `packages/lumiseval-evidence`
- `packages/lumiseval-core`

Core evaluation intent is split into two top-level quality dimensions:
- `retrieval_score`: did we retrieve the right evidence safely and sufficiently?
- `answer_score`: is the generated answer correct, relevant, and safe?

`composite_score` is currently the simple average of available top-level scores.

## 1) High-Level Architecture

```mermaid
flowchart LR
  U["User / Benchmark Source"] --> I["Interfaces: CLI, API, SDK"]
  I --> A["Adapter Layer: create_dataset_adapter()"]
  A --> C["Canonical Schema: EvalCase"]

  C --> CR["CachedNodeRunner (strict target execution)"]
  C --> NR["NodeRunner (single node correctness tests)"]
  I --> SG["Single Generation API: run_graph()"]

  CR --> NP["Node plan + cache reuse (per case)"]
  SG --> G["LangGraph Evaluation Pipeline"]
  NR --> NO["NodeRunResult: node_output + final_state"]

  NP --> O["Outputs: CLI node result / JSON report"]
  G --> R["EvalReport"]
  R --> O2["Outputs: API response / report"]

  subgraph EXT["External Engines & Services"]
    E1["RAGAS"]
    E2["DeepEval"]
    E3["Giskard"]
    E4["LiteLLM Judge Models"]
    E5["LanceDB + SentenceTransformers"]
    E6["Tavily Web Search (optional)"]
  end

  G -.uses.-> E1
  G -.uses.-> E2
  G -.uses.-> E3
  G -.uses.-> E4
  G -.uses.-> E5
  G -.optional.-> E6
```

## 2) Medium-Level Architecture

### 2.1 Graph Node Flow

```mermaid
flowchart TD
  N1["scan"] --> N2["estimate"]
  N2 --> N3["approve"]
  N3 --> N4["chunk"]
  N4 --> N5["claims"]
  N5 --> N6["dedupe"]
  
  N6 --> M1["relevance"]
  N6 --> M2["grounding"]
  N3 --> M3["redteam"]
  N3 --> M4["rubric"]

  M1 --> N8["eval"]
  M2 --> N8
  M3 --> N8
  M4 --> N8
  N8 --> N9["EvalReport"]

  subgraph CFG["Metric Activation from EvalJobConfig"]
    C1["enable_faithfulness + enable_answer_relevancy"]
    C2["enable_hallucination"]
    C3["enable_adversarial"]
    C4["enable_rubric + rubric_rules"]
  end

  C1 -.controls.-> M1
  C2 -.controls.-> M2
  C3 -.controls.-> M3
  C4 -.controls.-> M4
```

### 2.2 Evidence Routing Cascade

```mermaid
flowchart LR
  C["Claim"] --> L["Local LanceDB Query"]
  L -->|best score >= threshold| RL["Return LOCAL evidence + verdict"]
  L -->|best score < threshold| M["MCP LanceDB (stub placeholder)"]
  M --> W{"web_search enabled and Tavily key available?"}
  W -->|Yes| T["Tavily Search"]
  T --> IX["Index web passages into local LanceDB"]
  IX --> RQ["Re-query local LanceDB"]
  RQ --> RW["Return WEB evidence + verdict"]
  W -->|No| RN["Return NONE / UNVERIFIABLE"]
```

## 3) Low-Level Function Call Flow

### 3.1 Full Dataset Through CLI Strict Target

```mermaid
sequenceDiagram
  actor User
  participant CLI as "CLI run()"
  participant AD as "create_dataset_adapter()"
  participant DA as "DatasetAdapter.iter_cases()"
  participant SC as "scan_cases()"
  participant ES as "nodes.cost_estimator.estimate()"
  participant CN as "CachedNodeRunner.run_case()"
  participant NF as "node function map"

  User->>CLI: `lumiseval run <target_node> --input ...`
  CLI->>AD: resolve local / huggingface adapter
  AD-->>CLI: adapter instance
  CLI->>DA: iter_cases(split, limit)
  DA-->>CLI: EvalCase[]
  CLI->>SC: scan selected cases
  SC-->>CLI: InputMetadata
  CLI->>ES: estimate(scan_meta, job_config)
  ES-->>CLI: CostEstimate
  CLI->>User: confirm (unless --yes)

  loop each EvalCase
    CLI->>CN: run_case(case, node_name, job_config, force)
    CN->>NF: execute uncached prereqs + target
    NF-->>CN: per-node updates
    CN-->>CLI: CachedNodeRunResult
  end
```

### 3.2 Single-Node Correctness Testing Path

```mermaid
sequenceDiagram
  actor Dev
  participant NR as "NodeRunner.run_case()"
  participant IS as "build_initial_state()"
  participant NF as "node function map"
  participant ST as "in-memory EvalState"

  Dev->>NR: run_case(case, node_name, include_prerequisites=True)
  NR->>IS: create canonical EvalState
  IS-->>NR: initial state
  NR->>NR: resolve prerequisites[node_name]

  loop each planned step
    NR->>NF: call node fn(step)
    NF-->>NR: updates dict
    NR->>ST: state.update(updates)
  end

  NR-->>Dev: NodeRunResult(executed_nodes, node_output, final_state)
```

## Scoring Model (Current)

- Inputs to eval:
  - `relevance_metrics`
  - `grounding_metrics`
  - `redteam_metrics`
  - `rubric_metrics`
  - `claim_verdicts` from evidence routing
- Derived metric:
  - `evidence_support_rate` = proportion of claims with `SUPPORTED` verdict
- Partitioning:
  - Retrieval bucket: metrics with category `RETRIEVAL` + `evidence_support_rate`
  - Answer bucket: metrics with category `ANSWER`, excluding `vulnerability_*` marker metrics
- Warnings:
  - `vulnerability_*` results and metric errors are surfaced as warnings
- Composite:
  - `composite_score = avg([retrieval_score, answer_score] where present)`

## Canonical Dataset Contract for V1

All dataset sources are normalized to `EvalCase`:
- `case_id`
- `generation` (required)
- `question` (optional)
- `ground_truth` (optional)
- `context` (optional list)
- `reference_files` (optional list)
- `rubric_rules` (optional list)
- `metadata` (free-form)

This contract is what makes both flows modular:
- strict-target dataset execution (`CachedNodeRunner` via CLI `run`)
- isolated node validation (`NodeRunner`)

## Current Gaps (Known in Code)

- MCP retrieval is a stub in `retrieve`.
- API remains synchronous (`POST /jobs` executes inline).
- `cost_actual_usd` tracking is not fully wired from real LLM usage yet.
- Persisted jobs/reports storage is not implemented yet.
