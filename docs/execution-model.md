# Execution Model

This document describes the current GEval-only execution model after the legacy hard cut.

## Graph Topology

```text
scan
├─ chunk -> claims -> dedupe -> {relevance, grounding}
├─ geval_steps -> geval
├─ redteam
└─ reference

{relevance, grounding, geval, redteam, reference} -> eval
```

## Eligibility Gates

- `generation` required for all nodes
- `context` required for `chunk`, `claims`, `dedupe`, `relevance`, `grounding`
- `question` required for `relevance`
- `geval.metrics` required for `geval_steps`, `geval`
- `reference` required for `reference`

## CLI Behavior

- `lumiseval estimate <target> ...`
  - scans dataset
  - builds cache-aware run plan
  - computes uncached delta cost
  - does not execute graph nodes

- `lumiseval run <target> ...`
  - executes strict dependency chain up to `<target>`
  - uses node cache by `(case_hash, config_hash, node_name)`
  - for `target=eval`, metric siblings run in parallel when eligible

## Caching

- Case hash inputs: `generation`, `question`, `reference`, `context`, `geval`, `reference_files`
- Config hash inputs: `judge_model`, metric toggles, web search, evidence threshold
- GEval step artifacts are cached separately by signature:
  - `signature = sha256(model + prompt_version + parser_version + criteria)`

## Concurrency

- `CachedNodeRunner` executes prerequisites in order.
- For `target=eval`, these metric nodes can run concurrently:
  - `relevance`, `grounding`, `redteam`, `geval`, `reference`

## Removed Legacy Surfaces

- No legacy custom-judge input aliases
- No legacy config/API alias flags
- No legacy node-name aliases
- No legacy artifact/cache fallback paths
