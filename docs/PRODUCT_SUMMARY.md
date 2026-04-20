# neXa-gauge Product Summary

## What It Is

neXa-gauge is a Python package and CLI for evaluating LLM outputs using a cache-aware node graph.

Core commands:
- `nexagauge run <target_node>`
- `nexagauge estimate <target_node>`

Primary distribution target: `pip install nexa-gauge`.

## What It Solves

- Standardizes evaluation workflows across grounding, relevance, red-team, GEval, and reference metrics.
- Reduces repeated LLM spend with node-level caching and route-aware cache keys.
- Gives cost visibility before execution through estimate mode.
- Produces structured JSON reports suitable for CI pipelines and downstream analytics.

## Core Design

- Topology-driven execution (`topology.py`) with explicit prerequisites per node.
- Shared runner for both run and estimate modes.
- Parallel metric fan-out for `eval` target.
- Declarative report projection via `REPORT_VISIBILITY` and section gates.
- Per-node model routing (`LLM_{NODE}_MODEL`, fallback, temperature) plus runtime overrides.

## User Experience

- Local data support: `.json`, `.jsonl`, `.csv`, text fallback.
- Optional Hugging Face dataset ingestion via `hf://...`.
- Progress bar by default; per-node debug logs with `--debug`.
- Stable streaming result order even with concurrency.

## Packaging and Adoption

- Package name: `nexa-gauge`
- CLI entrypoint: `nexagauge`
- Python requirement: `>=3.10`
- Optional extra for HF adapters: `nexa-gauge[huggingface]`
- Repository includes contributor tooling (`uv`, `make`, tests, lint) while end users can start from PyPI only.
