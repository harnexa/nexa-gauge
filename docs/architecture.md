# nexa-gauge Architecture

nexa-gauge is a CLI-first evaluation engine for LLM outputs. Execution is driven by a typed node topology, a cache-aware runner, and declarative report projection.

Source of truth for node dependencies and input gating: `packages/nexagauge-graph/ng_graph/topology.py`.

## Runtime Components

- `ng_cli` - command entrypoint (`run`, `estimate`), dataset selection, model routing overrides.
- `adapters` - input ingestion from local files and Hugging Face datasets.
- `ng_graph.runner` - ordered/parallel execution, node-level caching, streaming outcomes.
- `ng_graph.graph` + `ng_graph.nodes.*` - node implementations.
- `ng_graph.nodes.report` - declarative report projection from final state.

## Top-Down Pipeline Diagram

- Built from `PIPELINE` in `topology.py`, with a layout close to the original design style.
- Solid edges show primary graph flow.
- Dashed edges show gating and extra eval prerequisites.

Shape key:

- Circle — utility node (`is_utility`)
- Rounded rectangle — metric node (`is_metric`)
- Hexagon — `eval` (the single join every branch funnels through)
- Stadium — `report` (terminal aggregation)
- Rectangle — `scan` (preflight)

Edge labels encode the target node's `requires_*` input gates. Edges into `eval` are unlabeled because `eval` itself has no input requirements.

```mermaid
flowchart TD
    scan[scan]

    %% Utility nodes (circles)
    chunk((chunk))
    claims((claims))
    dedup((dedup))
    geval_steps((geval_steps))

    %% Metric nodes (rounded rectangles)
    relevance(relevance)
    grounding(grounding)
    redteam(redteam)
    geval(geval)
    reference(reference)

    %% Orchestration
    eval{{eval}}
    report([report])

    %% Utility chain
    scan -- "requires: generation" --> chunk
    chunk -- "requires: generation" --> claims
    claims -- "requires: generation" --> dedup
    scan -- "requires: generation + geval" --> geval_steps

    %% Metric fan-out
    dedup -- "requires: generation + question" --> relevance
    dedup -- "requires: generation + context" --> grounding
    scan -- "requires: generation" --> redteam
    geval_steps -- "requires: generation + geval" --> geval
    scan -- "requires: generation + reference" --> reference

    %% Join into eval
    chunk --> eval
    claims --> eval
    dedup --> eval
    geval_steps --> eval
    relevance --> eval
    grounding --> eval
    redteam --> eval
    geval --> eval
    reference --> eval

    %% Terminal
    eval --> report

    %% Node colors — muted mid-tone palette, hue-matched to NodeSpec.color in topology.py
    style scan        fill:#7FC7D1,stroke:#3F7A82,color:#fff
    style chunk       fill:#7FA8D8,stroke:#3F6A9C,color:#fff
    style claims      fill:#C07AA8,stroke:#7A4469,color:#fff
    style dedup       fill:#7FBF7F,stroke:#3F8A3F,color:#fff
    style geval_steps fill:#9FBF9F,stroke:#5F8A5F,color:#fff
    style relevance   fill:#8FCF7F,stroke:#4F8A3F,color:#fff
    style grounding   fill:#7FA8D8,stroke:#4F75A8,color:#fff
    style redteam     fill:#D8847A,stroke:#9A4A42,color:#fff
    style geval       fill:#A88FBF,stroke:#6F5A8A,color:#fff
    style reference   fill:#CF7FBF,stroke:#8F4F82,color:#fff
    style eval        fill:#E0C970,stroke:#A08F3F,color:#3A2F0F
    style report      fill:#C9A85C,stroke:#8F7A3A,color:#fff
```

## Execution Rules

- Dependencies and requirement labels are derived from `PIPELINE` node specs (direct-parent edges).
- `eval` aggregates metric branches plus utility prerequisites; `report` depends on `eval`.
- The eligibility subgraph mirrors `scan`-produced presence flags that gate node execution.
- At runtime, the CLI runner can append `report` for non-report targets; this diagram focuses on architecture dependency flow.

## Data and Output Contracts

Input normalization (`scan`) maps common aliases into canonical `inputs` fields:

- `case_id`, `generation`, `question`, `context`, `reference`, `geval`, `redteam`

Core runtime state includes:

- control: `target_node`, `execution_mode`, `llm_overrides`
- artifacts: `generation_chunk`, `generation_claims`, `generation_dedup_claims`, `geval_steps`, metric outputs
- bookkeeping: `estimated_costs`, `node_model_usage`

Report shape is controlled by `REPORT_VISIBILITY` and `SECTION_GATES` in `ng_graph.nodes.report`.
