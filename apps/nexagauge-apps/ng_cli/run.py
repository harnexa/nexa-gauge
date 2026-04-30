"""Run CLI entrypoint and concurrency tuning guide.

Concurrency knobs:
- ``--max-workers``: case-level parallelism (how many records run at once).
- ``--max-in-flight``: cap on submitted-but-not-yet-emitted cases.
- ``--llm-concurrency``: global cap on in-flight LLM calls across all threads.

Suggested local profiles (16-core machine):

1. Debug / deterministic
- ``--max-workers 1``
- ``--llm-concurrency 8``
- keep ``--max-in-flight`` unset (ignored when max-workers=1)

2. Normal throughput (recommended default for daily runs)
- ``--max-workers 4``
- ``--llm-concurrency 16``
- keep ``--max-in-flight`` unset (defaults to ``2 * max-workers``)

3. High throughput (only if provider quota/rate limits allow)
- ``--max-workers 6`` (or 8)
- ``--llm-concurrency 24`` (or higher if stable)
- optionally ``--max-in-flight 18`` (about ``3 * max-workers``) for high latency/jitter

Tuning tips:
- If you see rate-limit/retry errors, reduce ``--llm-concurrency`` first.
- If workers idle during high latency variance, increase ``--max-in-flight``.
- For ``--debug``, prefer ``--max-workers 1`` for readable logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import typer
from adapters import create_dataset_adapter
from ng_core.cache import CacheStore, NoOpCacheStore
from ng_core.constants import DEFAULT_CHUNKER_STRATEGY, DEFAULT_REFINER_STRATEGY, REFINER_TOP_K
from ng_graph.llm.gateway import set_llm_concurrency
from ng_graph.log import set_node_logging_enabled
from ng_graph.nodes import eval as eval_node
from ng_graph.runner import CachedNodeRunner

from .util import (
    DEFAULT_FALLBACK_LLM,
    DEFAULT_PRIMARY_LLM,
    _case_progress,
    _is_case_eligible_for_target_path,
    _is_node_eligible_for_inputs,
    _plan_nodes_for_target,
    _print_llm_routing_summary,
    _print_node_timings_summary,
    _progress_total_from_bounds,
    _resolve_runtime_llm_overrides,
    _resolve_target_node,
    _scan_inputs_from_case,
    _set_case_llm_overrides,
    _write_report_json,
    console,
)

# Write me comments after each argument of what is does `llm_concurency`, `max_in_flight`, `max_workers`


@dataclass
class _RunEligibilityStats:
    submitted_cases: int
    eligible_counts_by_node: dict[str, int]


def _write_metric_breakdown_files(summary: dict[str, Any], metrics_dir: Path) -> None:
    by_node = summary.get("by_node")
    if not isinstance(by_node, dict) or not by_node:
        return

    by_metric = summary.get("by_metric")
    by_metric_dict = by_metric if isinstance(by_metric, dict) else {}

    for existing in metrics_dir.glob("*.json"):
        existing.unlink(missing_ok=True)

    for node_name, node_stats in sorted(by_node.items()):
        if not isinstance(node_stats, dict):
            continue
        payload = {
            "schema_version": summary.get("schema_version"),
            "cases_with_eval": summary.get("cases_with_eval"),
            "node": node_name,
            "summary": node_stats,
            "metrics": by_metric_dict.get(node_name, {}),
        }
        out_path = metrics_dir / f"{node_name}.json"
        out_path.write_text(json.dumps(payload, indent=2))


def _iter_eligible_cases_with_overrides(
    selected_cases: Iterable[Any],
    advance_progress: Callable[[], None],
    target_node: str,
    branch_nodes: list[str],
    llm_overrides: dict[str, dict[str, str]],
    chunker: str,
    refiner: str,
    refiner_top_k: int,
    stats: _RunEligibilityStats,
) -> Iterator[Any]:
    """Stream eligible cases while tracking per-node eligibility stats."""
    for case in selected_cases:
        advance_progress()
        if not _is_case_eligible_for_target_path(target_node=target_node, case=case):
            continue
        inputs = _scan_inputs_from_case(case)
        for node in branch_nodes:
            if _is_node_eligible_for_inputs(node, inputs):
                stats.eligible_counts_by_node[node] += 1
        stats.submitted_cases += 1
        yield _set_case_llm_overrides(
            case,
            llm_overrides,
            chunker=chunker,
            refiner=refiner,
            refiner_top_k=refiner_top_k,
        )


def run(
    node_name: str = typer.Argument(..., help="Target node name (e.g. claims, eval)."),
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Dataset source: local file path or hf://<dataset-id>.",
    ),
    start: int = typer.Option(
        0,
        "--start",
        min=0,
        help="Start index (inclusive) for selected rows.",
    ),
    end: Optional[int] = typer.Option(
        None,
        "--end",
        min=1,
        help="End index (exclusive) for selected rows.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        min=1,
        help="Maximum number of cases to run from the source.",
    ),
    adapter: str = typer.Option(
        "auto", "--adapter", help="Adapter to use: auto, local, huggingface."
    ),
    hf_config: Optional[str] = typer.Option(
        None, "--hf-config", help="Optional Hugging Face dataset config name."
    ),
    hf_revision: Optional[str] = typer.Option(
        None,
        "--hf-revision",
        help="Optional Hugging Face dataset revision/tag/commit.",
    ),
    judge_model: str = typer.Option(
        DEFAULT_PRIMARY_LLM,
        "--model",
        "-m",
        help="Global primary LLM model (backward-compatible alias of --llm-model MODEL).",
    ),
    llm_model: list[str] = typer.Option(
        (),
        "--llm-model",
        help=(
            "Repeatable LLM model override. Use MODEL for global default, or NODE=MODEL "
            f"(for example: --llm-model {DEFAULT_FALLBACK_LLM} "
            f"--llm-model grounding={DEFAULT_PRIMARY_LLM})."
        ),
    ),
    llm_fallback: list[str] = typer.Option(
        (),
        "--llm-fallback",
        help=(
            "Repeatable fallback override. Use MODEL for global fallback, or NODE=MODEL "
            f"(for example: --llm-fallback {DEFAULT_FALLBACK_LLM} "
            f"--llm-fallback grounding={DEFAULT_PRIMARY_LLM})."
        ),
    ),
    chunker: str = typer.Option(
        DEFAULT_CHUNKER_STRATEGY,
        "--chunker",
        help="Chunking strategy for the `chunk` utility node.",
    ),
    refiner: str = typer.Option(
        DEFAULT_REFINER_STRATEGY,
        "--refiner",
        help="Refiner strategy for selecting top-k chunks (default: mmr).",
    ),
    refiner_top_k: int = typer.Option(
        REFINER_TOP_K,
        "--refiner-top-k",
        min=1,
        help="Maximum number of chunks to keep after refinement.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue processing remaining cases if one case fails.",
    ),
    # Case-level parallelism: how many records are processed at the same time.
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        min=1,
        help=(
            "Case-level parallelism. Number of records processed concurrently. "
            "Default 1 means one case at a time."
        ),
    ),
    # Global cap across all threads for live LLM requests (run + fallback calls).
    # This is the main guardrail against rate limits/provider overload.
    llm_concurrency: int = typer.Option(
        16,
        "--llm-concurrency",
        min=1,
        help=(
            "Global cap on concurrent LLM calls across all worker threads. "
            "Lower this to reduce provider pressure/rate-limit risk."
        ),
    ),
    # Backpressure window for submitted cases waiting to be emitted in input order.
    # Caps submitted-but-not-yet-emitted case futures, Default is 2 * max_workers via normalization
    # Only matters when max_workers > 1.
    max_in_flight: Optional[int] = typer.Option(
        None,
        "--max-in-flight",
        min=1,
        help=(
            "Upper bound on submitted-but-not-yet-emitted cases. "
            "Only used when max_workers > 1. Defaults to max_workers * 2."
        ),
    ),
    force: bool = typer.Option(False, "--force", help="Ignore cache reads (still writes)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache reads and writes."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help=(
            "Write JSON output here. Creates `case_report/` for per-case report files and "
            "`metrics/` for aggregate metric breakdown files."
        ),
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help=(
            "Emit a colored per-node line (from the node's spec color) when each "
            "node starts running for a case. Excludes pure orchestration nodes "
            "(eval, report, chunk)."
        ),
    ),
) -> None:
    """Execute selected cases up to `node_name` without preflight prompts."""
    llm_concurrency = int(getattr(llm_concurrency, "default", llm_concurrency))
    chunker = str(getattr(chunker, "default", chunker))
    refiner = str(getattr(refiner, "default", refiner))
    refiner_top_k = int(getattr(refiner_top_k, "default", refiner_top_k))
    debug = bool(getattr(debug, "default", debug))

    target_node = _resolve_target_node(node_name)

    effective_judge_model, llm_overrides, llm_warnings = _resolve_runtime_llm_overrides(
        target_node=target_node,
        legacy_model=judge_model,
        llm_model_values=llm_model,
        llm_fallback_values=llm_fallback,
    )

    _print_llm_routing_summary(
        target_node=target_node,
        global_primary=effective_judge_model,
        llm_overrides=llm_overrides,
    )
    set_llm_concurrency(llm_concurrency)
    for warning in llm_warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    cache_store: CacheStore = NoOpCacheStore() if no_cache else CacheStore()
    runner = CachedNodeRunner(cache_store=cache_store)

    case_report_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        case_report_dir = output_dir / "case_report"
        metrics_dir = output_dir / "metrics"
        case_report_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

    total_executed = 0
    total_cached = 0
    succeeded = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    timings_by_case: list[dict[str, float]] = []

    # We create one eval_collator per run, to segregate multiple runs
    eval_collector = eval_node.EvalBatchCollector()

    branch_nodes = _plan_nodes_for_target(target_node)
    eligibility_stats = _RunEligibilityStats(
        submitted_cases=0,
        eligible_counts_by_node={node: 0 for node in branch_nodes},
    )

    effective_end = end
    if limit is not None:
        bounded_end = start + limit
        effective_end = bounded_end if effective_end is None else min(effective_end, bounded_end)

    ds_adapter = create_dataset_adapter(
        source=input,
        adapter=adapter,
        config_name=hf_config,
        revision=hf_revision,
    )

    selected_cases = islice(
        ds_adapter.iter_cases(limit=effective_end),
        start,
        effective_end,
    )
    set_node_logging_enabled(debug)
    try:
        with _case_progress(
            enabled=not debug,
            description=f"run:{target_node}",
            total=_progress_total_from_bounds(start=start, end=effective_end),
        ) as advance_progress:
            for outcome in runner.run_cases_iter(
                cases=_iter_eligible_cases_with_overrides(
                    selected_cases=selected_cases,
                    advance_progress=advance_progress,
                    target_node=target_node,
                    branch_nodes=branch_nodes,
                    llm_overrides=llm_overrides,
                    chunker=chunker,
                    refiner=refiner,
                    refiner_top_k=refiner_top_k,
                    stats=eligibility_stats,
                ),
                node_name=target_node,
                force=force,
                execution_mode="run",
                max_workers=max_workers,
                max_in_flight=max_in_flight,
                continue_on_error=continue_on_error,
                debug=debug,
                eval_collector=eval_collector,
            ):
                if outcome.result is not None:
                    result = outcome.result
                    succeeded += 1
                    total_executed += len(result.executed_nodes)
                    total_cached += len(result.cached_nodes)
                    if debug and result.node_timings:
                        timings_by_case.append(result.node_timings)

                    if case_report_dir:
                        report = result.final_state.get("report")
                        if report is not None:
                            _write_report_json(
                                report=report,
                                output_dir=case_report_dir,
                                case_id=result.case_id,
                            )
                    continue

                failed += 1
                failures.append((outcome.case_id, outcome.error or "unknown error"))
    finally:
        set_node_logging_enabled(False)
        if debug:
            _print_node_timings_summary(
                timings_by_case,
                eligible_counts_by_node=eligibility_stats.eligible_counts_by_node,
                total_cases=eligibility_stats.submitted_cases,
            )
        eval_summary_snapshot = eval_collector.snapshot()
        if metrics_dir is not None:
            _write_metric_breakdown_files(
                summary=eval_summary_snapshot,
                metrics_dir=metrics_dir,
            )
        for table in eval_node.build_eval_summary_tables(eval_summary_snapshot):
            console.print()
            console.print(table)

    console.print(
        f"\n[bold green]Done.[/bold green]  "
        f"cases={succeeded + failed}  succeeded={succeeded}  failed={failed}  "
        f"executed_steps={total_executed}  cached_steps={total_cached}"
    )

    if failures:
        console.print("\n[yellow]Failures:[/yellow]")
        for case_id, err in failures[:10]:
            console.print(f"  - {case_id}: {err}")
        if not continue_on_error:
            raise typer.Exit(1)
