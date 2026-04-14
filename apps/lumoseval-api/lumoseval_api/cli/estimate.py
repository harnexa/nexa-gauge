from __future__ import annotations

from itertools import islice
from typing import Optional

import typer
from lumoseval_api.adapters import create_dataset_adapter
from lumoseval_core.cache import CacheStore, NoOpCacheStore
from lumoseval_core.types import CostEstimate
from lumoseval_graph.runner import CachedNodeRunner
from rich.table import Table

from .util import (
    DEFAULT_PRIMARY_LLM,
    _collect_estimate_rows,
    _format_cost,
    _is_node_eligible_for_inputs,
    _plan_nodes_for_target,
    _print_llm_routing_summary,
    _resolve_runtime_llm_overrides,
    _resolve_target_node,
    _set_case_llm_overrides,
    console,
)


def estimate(
    node_name: str = typer.Argument(..., help="Target node name (e.g. grounding, relevance)."),
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
        help="Maximum number of cases to estimate from the source.",
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
) -> None:
    """Estimate uncached branch costs via graph estimate-mode execution."""
    # try:
    target_node = _resolve_target_node(node_name)
    # except ValueError as exc:
    #     console.print(f"[red]{exc}[/red]")
    #     raise typer.Exit(1)

    # try:
    effective_judge_model, llm_overrides, llm_warnings = _resolve_runtime_llm_overrides(
        target_node=target_node,
        legacy_model=judge_model,
        llm_model_values=llm_model,
        llm_fallback_values=llm_fallback,
    )
    # except ValueError as exc:
    #     console.print(f"[red]{exc}[/red]")
    #     raise typer.Exit(1)

    _print_llm_routing_summary(
        target_node=target_node,
        global_primary=effective_judge_model,
        llm_overrides=llm_overrides,
    )
    for warning in llm_warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    cache_store: CacheStore = NoOpCacheStore() if no_cache else CacheStore(cache_dir)
    runner = CachedNodeRunner(cache_store=cache_store)

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
    cases = (_set_case_llm_overrides(case, llm_overrides) for case in selected_cases)

    total_records = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    aggregated_costs: dict[str, CostEstimate] = {}
    branch_nodes = _plan_nodes_for_target(target_node)
    node_stats: dict[str, dict[str, int]] = {
        node: {"executed": 0, "cached": 0, "estimated": 0, "eligible_uncached": 0}
        for node in branch_nodes
    }

    # try:
    for outcome in runner.run_cases_iter(
        cases=cases,
        node_name=target_node,
        force=force,
        execution_mode="estimate",
        max_workers=max_workers,
        max_in_flight=max_in_flight,
        continue_on_error=continue_on_error,
    ):
        total_records += 1
        if outcome.result is None:
            failed += 1
            failures.append((outcome.case_id, outcome.error or "unknown error"))
            continue

        result = outcome.result
        estimated_costs = result.final_state.get("estimated_costs") or {}
        executed_nodes = set(getattr(result, "executed_nodes", []) or [])
        cached_nodes = set(getattr(result, "cached_nodes", []) or [])
        inputs = result.final_state.get("inputs")
        for node in branch_nodes:
            if node in executed_nodes:
                node_stats[node]["executed"] += 1
                if _is_node_eligible_for_inputs(node, inputs):
                    node_stats[node]["eligible_uncached"] += 1
            if node in cached_nodes:
                node_stats[node]["cached"] += 1
            if node in estimated_costs:
                node_stats[node]["estimated"] += 1

        for step, step_cost in estimated_costs.items():
            cost_obj = (
                step_cost
                if isinstance(step_cost, CostEstimate)
                else CostEstimate.model_validate(step_cost)
            )
            existing = aggregated_costs.get(step)
            if existing is None:
                aggregated_costs[step] = CostEstimate(
                    cost=float(cost_obj.cost or 0.0),
                    input_tokens=float(cost_obj.input_tokens)
                    if cost_obj.input_tokens is not None
                    else None,
                    output_tokens=float(cost_obj.output_tokens)
                    if cost_obj.output_tokens is not None
                    else None,
                )
                continue
            existing.cost = float(existing.cost or 0.0) + float(cost_obj.cost or 0.0)
            if cost_obj.input_tokens is not None:
                existing.input_tokens = float(existing.input_tokens or 0.0) + float(
                    cost_obj.input_tokens
                )
            if cost_obj.output_tokens is not None:
                existing.output_tokens = float(existing.output_tokens or 0.0) + float(
                    cost_obj.output_tokens
                )
    # except ValueError as exc:
    #     console.print(f"[red]{exc}[/red]")
    #     raise typer.Exit(1)
    # except Exception as exc:
    #     console.print(f"[red]{exc}[/red]")
    #     raise typer.Exit(1)

    successful_cases = total_records - failed
    rows = _collect_estimate_rows(
        target_node=target_node,
        cost_by_node=aggregated_costs,
        node_stats=node_stats,
        total_selected_cases=total_records,
        successful_cases=successful_cases,
        effective_judge_model=effective_judge_model,
        llm_overrides=llm_overrides,
    )
    total_cost = sum(cost for *_, cost in rows)

    table = Table(title=f"Estimate (target={target_node}, branch summary)", show_header=True)
    table.add_column("node_name", style="cyan")
    table.add_column("model", style="green")
    table.add_column("status", style="magenta")
    table.add_column("cached", justify="right")
    table.add_column("uncached", justify="right")
    table.add_column("uncached_eligible", justify="right")
    table.add_column("uncached_eligible_pct", justify="right")
    table.add_column("cost_estimate", justify="right", style="yellow")
    for node, model, status, executed, cached, eligible, eligible_pct, cost in rows:
        display_cost = "-" if float(cost) == 0.0 else _format_cost(cost)
        table.add_row(node, model, status, cached, executed, eligible, eligible_pct, display_cost)
    total_display_cost = "-" if float(total_cost) == 0.0 else _format_cost(total_cost)
    table.add_row("TOTAL", "-", "-", "-", "-", "-", "-", total_display_cost, style="bold")
    console.print(table)

    console.print(
        f"\n[bold green]Done.[/bold green]  "
        f"cases={total_records}  failed={failed}  total_cost_estimate={_format_cost(total_cost)}"
    )

    if failures:
        console.print("\n[yellow]Failures:[/yellow]")
        for case_id, err in failures[:10]:
            console.print(f"  - {case_id}: {err}")
        if not continue_on_error:
            raise typer.Exit(1)
