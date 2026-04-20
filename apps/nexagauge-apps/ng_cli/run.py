from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Optional

import typer
from adapters import create_dataset_adapter
from ng_core.cache import CacheStore, NoOpCacheStore
from ng_graph.log import set_node_logging_enabled
from ng_graph.runner import CachedNodeRunner

from .util import (
    DEFAULT_PRIMARY_LLM,
    _case_progress,
    _is_case_eligible_for_target_path,
    _print_llm_routing_summary,
    _progress_total_from_bounds,
    _resolve_runtime_llm_overrides,
    _resolve_target_node,
    _set_case_llm_overrides,
    _write_report_json,
    console,
)


def run(
    node_name: str = typer.Argument(..., help="Target node name (e.g. claims, eval)."),
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Dataset source: local file path or hf://<dataset-id>.",
    ),
    split: str = typer.Option("train", "--split", help="Dataset split for adapter-backed sources."),
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
            "(for example: --llm-model openai/gpt-4o --llm-model grounding=openai/gpt-4o-mini)."
        ),
    ),
    llm_fallback: list[str] = typer.Option(
        (),
        "--llm-fallback",
        help=(
            "Repeatable fallback override. Use MODEL for global fallback, or NODE=MODEL "
            "(for example: --llm-fallback openai/gpt-4o --llm-fallback grounding=openai/gpt-4o-mini)."
        ),
    ),
    web_search: bool = typer.Option(False, "--web-search", help="Enable Tavily web search."),
    evidence_threshold: float = typer.Option(0.75, "--evidence-threshold"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Deprecated: run executes immediately and no longer needs confirmation.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue processing remaining cases if one case fails.",
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        min=1,
        help="Number of records to process concurrently.",
    ),
    max_in_flight: Optional[int] = typer.Option(
        None,
        "--max-in-flight",
        min=1,
        help=(
            "Maximum number of submitted-but-not-yet-emitted records. Defaults to max_workers * 2."
        ),
    ),
    force: bool = typer.Option(False, "--force", help="Ignore cache reads (still writes)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache reads and writes."),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="Cache directory path."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write per-case JSON report output here when a run produces `report`.",
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
    del web_search, evidence_threshold

    target_node = _resolve_target_node(node_name)

    if yes:
        console.print(
            "[dim]`--yes` is deprecated for `run`; execution now starts immediately.[/dim]"
        )

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
    for warning in llm_warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    cache_store: CacheStore = NoOpCacheStore() if no_cache else CacheStore(cache_dir)
    runner = CachedNodeRunner(cache_store=cache_store)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_executed = 0
    total_cached = 0
    succeeded = 0
    failed = 0
    failures: list[tuple[str, str]] = []

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
        ds_adapter.iter_cases(split=split, limit=effective_end),
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
            # Keep this fully streaming: no dataset materialization.
            def _iter_eligible_cases_with_overrides():
                for case in selected_cases:
                    advance_progress()
                    if not _is_case_eligible_for_target_path(target_node=target_node, case=case):
                        continue
                    yield _set_case_llm_overrides(case, llm_overrides)

            for outcome in runner.run_cases_iter(
                cases=_iter_eligible_cases_with_overrides(),
                node_name=target_node,
                force=force,
                execution_mode="run",
                max_workers=max_workers,
                max_in_flight=max_in_flight,
                continue_on_error=continue_on_error,
                debug=debug,
            ):
                if outcome.result is not None:
                    result = outcome.result
                    succeeded += 1
                    total_executed += len(result.executed_nodes)
                    total_cached += len(result.cached_nodes)

                    if output_dir:
                        report = result.final_state.get("report")
                        if report is not None:
                            _write_report_json(
                                report=report,
                                output_dir=output_dir,
                                case_id=result.case_id,
                            )
                    continue

                failed += 1
                failures.append((outcome.case_id, outcome.error or "unknown error"))
    finally:
        set_node_logging_enabled(False)

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
