# nexa-gauge Architecture

nexa-gauge is a CLI-first evaluation engine for LLM outputs. Execution is driven by a typed node topology, a cache-aware runner, and topology-driven report projection.

Source of truth for node dependencies and input gating: `packages/nexagauge-graph/ng_graph/topology.py`.

## Runtime Components

- `ng_cli` - command entrypoint (`run`, `estimate`), dataset selection, model routing overrides.
- `adapters` - input ingestion from local files and Hugging Face datasets.
- `ng_graph.runner` - ordered/parallel execution, node-level caching, streaming outcomes.
- `ng_graph.graph` + `ng_graph.nodes.*` - node implementations.
- `ng_graph.nodes.report` - topology-driven report projection from final state.

## Top-Down Pipeline Diagram

- Built from `PIPELINE` in `topology.py`, with strategy containers for `chunk` and `refiner`.
- Solid edges show primary graph flow.
- Inner links inside containers show available strategies (one selected at runtime).

Shape key:

- Rectangle group — strategy family (`chunk`, `refiner`)
- Circle — utility leaf node / strategy option
- Rounded rectangle — metric node (`is_metric`)
- Hexagon — `eval` (the single join every branch funnels through)
- Stadium — `report` (terminal aggregation)
- Rectangle — `scan` (preflight)

Edge labels encode the target node's `requires_*` input gates. Edges into `eval` are unlabeled because `eval` itself has no input requirements.

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 20}} }%%
flowchart TD
    scan[scan]

    %% Strategy container: chunk (wide, short box)
    subgraph chunk_box["chunk"]
      direction LR
      chunk_sem((sem))
      chunk_more((...))
      chunk_sem --- chunk_more
    end

    %% Strategy container: refiner (wide, short box)
    subgraph refiner_box["refiner"]
      direction LR
      refiner_mmr((mmr))
      refiner_rerank((rerank))
      refiner_more((...))
      refiner_mmr --- refiner_rerank --- refiner_more
    end

    %% Utility nodes (circles)
    claims((claims))
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
    scan -- "requires: generation" --> chunk_box
    chunk_box -- "requires: generation" --> refiner_box
    refiner_box -- "requires: generation" --> claims
    scan -- "requires: generation + geval" --> geval_steps

    %% Metric fan-out
    claims -- "requires: generation + question" --> relevance
    claims -- "requires: generation + context" --> grounding
    scan -- "requires: generation" --> redteam
    geval_steps -- "requires: generation + geval" --> geval
    scan -- "requires: generation + reference" --> reference

    %% Join into eval
    chunk_box --> eval
    refiner_box --> eval
    claims --> eval
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
    style chunk_box   fill:#BBD3EE,stroke:#3F6A9C,color:#173B61
    style refiner_box fill:#BFE3BF,stroke:#3F8A3F,color:#1B4C1B
    style chunk_sem        fill:#A7C2E4,stroke:#3F6A9C,color:#173B61,stroke-width:1px
    style chunk_more       fill:#A7C2E4,stroke:#3F6A9C,color:#173B61,stroke-width:1px
    style refiner_mmr      fill:#A6D7A6,stroke:#3F8A3F,color:#1B4C1B,stroke-width:1px
    style refiner_rerank   fill:#A6D7A6,stroke:#3F8A3F,color:#1B4C1B,stroke-width:1px
    style refiner_more     fill:#A6D7A6,stroke:#3F8A3F,color:#1B4C1B,stroke-width:1px
    classDef strategySmall font-size:10px,stroke-width:1px;
    class chunk_sem,chunk_more,refiner_mmr,refiner_rerank,refiner_more strategySmall
    style claims      fill:#C07AA8,stroke:#7A4469,color:#fff
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
- `chunk` and `refiner` are strategy families; one option is selected at runtime via CLI (`--chunker`, `--refiner`).
- `eval` aggregates metric branches plus utility prerequisites; `report` depends on `eval`.
- The eligibility subgraph mirrors `scan`-produced presence flags that gate node execution.
- At runtime, the CLI runner can append `report` for non-report targets; this diagram focuses on architecture dependency flow.

## Data and Output Contracts

Input normalization (`scan`) maps common aliases into canonical `inputs` fields:

- `case_id`, `generation`, `question`, `context`, `reference`, `geval`, `redteam`

Core runtime state includes:

- control: `target_node`, `execution_mode`, `llm_overrides`
- strategy control: `chunker`, `refiner`, `refiner_top_k`
- artifacts: `generation_chunk`, `generation_refined_chunks`, `generation_claims`, `geval_steps`, metric outputs
- bookkeeping: `estimated_costs`, `node_model_usage`

Report shape is topology-driven in `ng_graph.nodes.report`: sections are included by non-`None` `state_key` values from `PIPELINE`.
