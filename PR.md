<!-- pr-snapshot: b7a9af9f3e213a01aa407baf5c2e7cd9245d815f -->

# GEval split, unified cache, and package rename to `nexagauge`

**Branch:** `geval-split` → `main`
**Date:** 2026-04-19
Updated: 2026-04-21

## Summary

Splits the GEval metric node into `geval_steps`, `geval_score`, and `geval_weighted_score`, and consolidates GEval step-artifact caching onto the universal `NodeCacheBackend` so `--no-cache` is respected uniformly. Renames all packages from `lumos*`/`lumoseval*` to `nexagauge*` (CLI app now `nexagauge-api`), overhauls the report node, and expands CI, docs, and test coverage.

## What Changed

### Core (`nexagauge-core`)
- Extends `cache.py` with a comprehensive module docstring covering per-node and GEval-artifact key schemes, storage layout, deserialisation, and extension points; adds `"geval_artifact": GevalCacheArtifact` to `_FIELD_TYPE_MAP`.
- Adds `GevalCacheArtifact` and redteam/geval typed input models to `types.py`, with concise docstrings on every Pydantic model and enum.
- Deletes the stale v1 `geval_cache.py` and its test — functionality now subsumed by the universal cache.

### Graph (`nexagauge-graph`)
- Splits GEval: `geval_steps.py` now owns step generation + cache lookup via `NodeCacheBackend`; `score.py` is a pure per-metric scorer; new `weighted_score.py` aggregates across metrics.
- `nodes/metrics/geval/cache.py` shrinks to pure helpers (`compute_geval_signature`, `collect_geval_signatures`, `build_geval_artifact_cache_key`, version constants). The `GevalArtifactCache` class is gone.
- Runner stashes `self._cache` into `state["__cache_store"]` so `GevalStepsNode` picks up the same backend (including `NoOpCacheStore` under `--no-cache`).
- Report node (`nodes/report.py`) restructured for richer aggregate and per-case projections.
- Gateway, graph wiring, topology, and registry updated to match the new node boundaries.

### CLI app rename (`apps/nexagauge-cli` → `apps/nexagauge-api`)
- Directory rename plus entry-point, import, and test-path updates.
- CLI `run` / `estimate` / `util` refactored around the split graph and new cache plumbing.

### Tests
- Rewrites `test_cache.py` around the shared backend; adds `test_build_geval_artifact_cache_key_stable`, `test_roundtrip_via_cache_store`, `test_signature_ignores_evaluation_steps`.
- `test_steps.py` switches fixtures to `CacheStore` / `NoOpCacheStore`; adds `test_no_cache_store_always_generates` and `test_shared_criteria_reuses_generated_steps_across_cases`.
- New `test_weighted_score.py`, `test_report_aggregate.py`, `test_report_projection.py`.
- Scanner, end-metric, estimate, runner-streaming, and override tests updated for the renamed packages and new node signatures.

### Docs, CI, config
- `docs/get-started.md`, `architecture.md`, `cli-code-flow.md`, `execution-model.md` reworked.
- Adds `.github/workflows/ci.yml`, `publish.yml`, and `dependabot.yml`; adds `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`.
- Root `pyproject.toml`, `Makefile`, `setup.sh`, `sample.json` updated for the rename.

## Why These Changes

- The prior monolithic `geval` node coupled step generation, scoring, and aggregation, which blocked caching step artifacts independently of per-case state and forced O(cases) LLM calls even when criteria were shared.
- Three overlapping caches (universal, GEval artifact, stale v1) meant `--no-cache` silently leaked cache hits for GEval steps. Unifying on `NodeCacheBackend` gives one policy switch, one disk layout, one serialisation path.
- Keying the GEval artifact by `(model, prompt_version, parser_version, item_fields, criteria)` — never by case or by caller-supplied `evaluation_steps` — preserves cross-case reuse so N cases sharing a criterion pay for one LLM call.
- Splitting GEval into `steps` / `score` / `weighted_score` aligns with the rest of the node graph's single-responsibility shape and makes the per-metric cache fingerprint tractable.
- The `lumos*` / `lumoseval*` package names were inconsistent with the product; renaming now, before external consumers arrive, avoids a harder migration later.

## Test Plan

- [ ] `uv run pytest packages/ apps/ -q` — full suite.
- [ ] `uv run pytest packages/nexagauge-graph/test_ng_graph/test_nodes/test_metrics/test_geval/ -v` — GEval-focused.
- [ ] Smoke (no-cache leak regression):
  - `rm -rf ./data/geval_steps`
  - `uv run nexagauge run geval_steps --input sample.json --output-dir ./data/geval_steps --no-cache`
  - Confirm every output `steps_source` is `"generated"` or `"provided"`; zero `"cache_used"`.
- [ ] Smoke (cross-case reuse retained): rerun without `--no-cache` twice; second run shows `"cache_used"` for shared criteria.
- [ ] `grep -R "GevalArtifactCache\|ng_core.geval_cache\|lumos_\|lumoseval_" packages/ apps/` — zero hits.

---
### Updates since snapshot 0403b0e (2026-04-19)

**New Changes:**
- **Topology registry:** Introduces `packages/nexagauge-graph/ng_graph/topology.py` — a frozen `NodeSpec` / `PIPELINE` single source of truth for node ordering, direct-prerequisite edges, eligibility flags, colors, env-key suffixes, and skip-output shapes; adds `transitive_prerequisites()` helper. Node wiring now declares only *direct* parents; transitive ancestors are derived.
- **Apps layout:** Consolidates `apps/lumiseval-cli` and adapters under a single `apps/nexagauge-apps/` package. Splits surfaces into `ng_cli/` (existing CLI) and a new `ng_api/` FastAPI entry point (`ng_api/main.py`). CLI app rename in the prior snapshot is now **`nexagauge-apps`** (not `nexagauge-api`) — the earlier name was inaccurate.
- **Docs rewrite:** Rewrites `docs/get-started.md`, `docs/architecture.md`, `docs/cli-code-flow.md`; adds `docs/PRODUCT_SUMMARY.md`; removes `docs/execution-model.md` and root `nodes.md` (content folded into architecture + topology docstrings).
- **Branding:** Adds `nexagauge-banner.svg`; updates README and legacy `lumiseval-banner.svg` references.
- **Test reorganization:** Moves tests into package-scoped dirs (`test_apps/test_adapters`, `test_apps/test_ng_cli`, `test_ng_core`, `test_ng_graph`); removes the stale `packages/lumiseval-graph/test_lumiseval_graph/__init__.py` residue.
- **Residual rename cleanup:** Module-level imports, pyproject entry points, Makefile targets, `setup.sh`, and `.env.example` all settled onto the `nexagauge` / `ng_*` names. Grep for `lumos_` / `lumoseval_` is clean.

**Reason:**
- Prior snapshot still had the CLI app advertised as `nexagauge-api` and left node wiring implicit in `graph.py`; centralising topology removes a class of drift bugs where ordering, colors, and eligibility flags lived in three files.
- Adding `ng_api` alongside `ng_cli` under one app package preserves a single install surface while leaving room for the upcoming HTTP entry point without another rename.
- Docs had diverged from the renamed modules and split node boundaries; the rewrite aligns them with the shipped code before review.

**New Tests:**
- No new test files in this delta; existing suites were updated only for the package move (`test_apps/`, `test_ng_core/`, `test_ng_graph/` roots).
---

### Updates since snapshot 28018e7 (2026-04-21)

**New Changes:**
- **Intra-node LLM parallelism:** Adds node-local threadpool fan-out for high-cardinality LLM work:
  - `ClaimExtractorNode` now parallelizes chunk-level extraction.
  - `GevalStepsNode` now parallelizes missing step generation across metrics while preserving cache/provided ordering and adds a lock around shared usage accounting.
  - `RedteamNode` now parallelizes per-metric evaluations with thread-safe usage recording.
- **Global LLM backpressure control:** Introduces process-wide LLM concurrency gating in `ng_graph/llm/gateway.py` via `BoundedSemaphore`, plus `set_llm_concurrency()` / `get_llm_concurrency()`. Both structured-call paths (`invoke` and `invoke_logprobs`) now execute under the same global cap.
- **Config/runtime concurrency split:** Replaces generic `MAX_CONCURRENT_JOBS` with explicit node-local worker caps in `ng_core`:
  - `CLAIMS_MAX_WORKERS`
  - `GEVAL_STEPS_MAX_WORKERS`
  - `REDTEAM_MAX_WORKERS`
  This separates case-level concurrency from intra-node fan-out.
- **Runner refactor (no API break):** Replaces monolithic `ng_graph/runner.py` with `ng_graph/runner/` package:
  - `engine.py` (execution/caching pipeline)
  - `plan.py` (plan topology derivation)
  - `fingerprints.py` (cache key/fingerprint mechanics)
  - `types.py` (result/data contracts)
  `ng_graph.runner` import compatibility is preserved through `runner/__init__.py`.
- **CLI tuning + observability:** `nexagauge run` adds `--llm-concurrency`; debug mode now prints per-node timing summaries with run/cache counts and eligibility-aware totals for targeted paths.
- **Config cleanup:** Removes legacy web-search toggles from runtime config/CLI path (`TAVILY_API_KEY`, `WEB_SEARCH_ENABLED`, `--web-search`, `--evidence-threshold`) to keep execution settings focused on graph + LLM evaluation.
- **Naming/docs polish:** CLI and project descriptions were updated to the "graph-based toolkit" wording, and `docs/get-started.md` was rewritten as a dev-first bootstrap/testing guide.

**Reason:**
- Case-level parallelism alone did not address intra-case bottlenecks where a single case fans out into many LLM calls (`claims`, `geval_steps`, `redteam`). This change parallelizes within a case while adding a single global throttle to avoid provider overload.
- Splitting runner internals reduces coupling and makes cache-fingerprint logic, plan derivation, and execution flow independently testable/maintainable without changing the public runner surface.
- Debug output previously lacked per-node latency distributions and cache-hit visibility across runs; timing summary output now makes throughput tuning (`--max-workers`, `--max-in-flight`, `--llm-concurrency`) measurable.

**New Tests:**
- No new test modules added in this delta.
- Updated coverage across runner streaming, gateway behavior, claim extraction, GEval step generation, and redteam/reference metric tests to align with runner package split and new concurrency paths.
---

## Notes for Reviewer

- On-disk `{cache_dir}/geval_artifacts/*.json` from the old `GevalArtifactCache` layout become unreachable; no automatic migration. Recommend `rm -rf $LUMISEVAL_CACHE_DIR/geval_artifacts` after upgrade.
- `state["__cache_store"]` uses an under-dunder key deliberately — it is a runner-to-node internal channel, not user data, and `_merge_state_patch` does not touch it.
- `NON_CACHEABLE_NODES = {"eval", "report"}` and the GEval artifact key prefix `v2:geval_artifact:` are the two extension points most likely to matter for future caching work; both are documented in `cache.py`'s new module docstring.
- The stale `PR.md` that previously lived on this branch was from the already-merged `full-refactor` PR and has been regenerated from scratch for `geval-split`.
- `topology.py` is intentionally located in `nexagauge-graph` but contains no function references, so `nexagauge-core` and `nexagauge-apps` can import it without introducing a circular dependency. Treat this as the extension point for any future metric node.
- The new `ng_api/main.py` is a placeholder FastAPI entry and is not wired into CI or docs yet — flag in review if we should gate its pyproject entry point behind an extra.
