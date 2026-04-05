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
    E2["DeepEval (GEval, BiasMetric, ToxicityMetric)"]
    E4["LiteLLM Judge Models"]
    E5["LanceDB + SentenceTransformers"]
    E6["Tavily Web Search (optional)"]
  end

  G -.uses.-> E2
  G -.uses.-> E4
  G -.uses.-> E5
  G -.optional.-> E6
```

## 2) Medium-Level Architecture

### 2.1 Graph Node Flow

```mermaid
flowchart TB
    subgraph CFG["EvalJobConfig toggles"]
        GRD["🔒 grounding\nfaithfulness per claim"]
        T1["enable_grounding"]
        REL["🔄 relevance\nanswer-relevancy"]
        T2["enable_relevance"]
        RDT["🛡️ redteam\nbias + toxicity"]
        T3["enable_redteam"]
        GVS["🧭 geval_steps\ncriteria -> eval steps"]
        GVT["🧠 geval\ncustom GEval metrics"]
        T4["enable_geval"]
        REF["📚 reference\nROUGE / BLEU / METEOR"]
        T5["enable_reference"]
    end

    SCAN["🔍 scan\ntokenise · count · eligibility"] --> CHK["📄 chunk\n~100-tok windows"]
    CHK --> CLM["🧩 claims\nextract atomic claims per chunk"]
    CLM --> DDP["🔀 dedupe\nMMR cosine dedup"]
    SCAN -- has_generation --> RDT
    SCAN -- has_geval --> GVS
    GVS --> GVT
    SCAN -- has_generation and has reference --> REF
    DDP -- has_question --> REL
    DDP -- has_context --> GRD
    REL --> EVL["⭐ eval\naggregate · score · report"]
    GRD --> EVL
    RDT --> EVL
    GVT --> EVL
    REF --> EVL
    EVL --> RPT(["📋 EvalReport"])
    T1 -. controls .-> GRD
    T2 -. controls .-> REL
    T3 -. controls .-> RDT
    T4 -. controls .-> GVS
    T4 -. controls .-> GVT
    T5 -. controls .-> REF

    GRD:::metric
    REL:::metric
    RDT:::metric
    GVS:::metric
    GVT:::metric
    REF:::metric
    SCAN:::preflight
    CHK:::ctxpath
    CLM:::ctxpath
    DDP:::ctxpath
    EVL:::agg
    RPT:::terminal

    classDef preflight fill:#1e3a5f,stroke:#4a9eff,color:#fff
    classDef ctxpath  fill:#1a3a1a,stroke:#4aaf4a,color:#fff
    classDef metric   fill:#3a1a3a,stroke:#af4aaf,color:#fff
    classDef agg      fill:#5a4000,stroke:#d4a017,color:#fff
    classDef terminal fill:#2a2a2a,stroke:#888,color:#fff
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
  participant CLIE as "CLI estimate()"
  participant CLI as "CLI run()"
  participant AD as "create_dataset_adapter()"
  participant DA as "DatasetAdapter.iter_cases()"
  participant SC as "scan_cases()"
  participant PL as "CachedNodeRunner.plan_dataset()"
  participant ES as "CostEstimator(job_config).estimate()"
  participant CN as "CachedNodeRunner.run_case()"
  participant NF as "node function map"

  User->>CLIE: `lumiseval estimate <target_node> --input ...`
  CLIE->>AD: resolve local / huggingface adapter
  AD-->>CLIE: adapter instance
  CLIE->>DA: iter_cases(split, limit)
  DA-->>CLIE: EvalCase[]
  CLIE->>SC: scan selected cases
  SC-->>CLIE: InputMetadata
  CLIE->>PL: plan_dataset(cases, target_node)
  PL-->>CLIE: to_run/cached/skipped by node
  CLIE->>ES: estimate(delta metadata overrides)
  ES-->>CLIE: CostReport (rich tables printed)

  User->>CLI: `lumiseval run <target_node> --input ...`
  CLI->>AD: resolve local / huggingface adapter
  AD-->>CLI: adapter instance
  CLI->>DA: iter_cases(split, limit)
  DA-->>CLI: EvalCase[]

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
  - `geval_metrics`
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
- `reference` (optional)
- `context` (optional list)
- `reference_files` (optional list)
- `geval` (optional object with `metrics[]`)
- `metadata` (free-form)

This contract is what makes both flows modular:
- strict-target dataset execution (`CachedNodeRunner` via CLI `run`)
- isolated node validation (`NodeRunner`)

## Current Gaps (Known in Code)

- MCP retrieval is a stub in `retrieve`.
- API handlers for `/estimate` and `/run` are documented but not implemented yet.
- `cost_actual_usd` tracking is not fully wired from real LLM usage yet.
- Persisted jobs/reports storage is not implemented yet.
