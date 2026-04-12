"""LumisEval CLI.

Public commands:
  - ``lumiseval estimate <node_name> --input <source>``: scan + plan + delta-cost only
  - ``lumiseval run <node_name> --input <source>``: execute branch directly
"""

from __future__ import annotations

import json
import re
import uuid
from itertools import islice
from pathlib import Path
from typing import Any, Optional

import typer
from lumiseval_core.cache import CacheStore, NoOpCacheStore
from lumiseval_graph.topology import NODE_ORDER, NODES_BY_NAME
from lumiseval_graph.llm.config import normalize_node_name
from lumiseval_graph.runner import CachedNodeRunner
from lumiseval_cli.adapters import create_dataset_adapter
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="lumiseval", help="Agentic LLM evaluation pipeline.")
console = Console()

DEFAULT_PRIMARY_LLM = "openai/gpt-4o-mini"
DEFAULT_FALLBACK_LLM = "openai/gpt-4o"


@app.callback()
def cli() -> None:
    """LumisEval command group."""


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned or "case"


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _resolve_target_node(node_name: str) -> str:
    if node_name not in NODES_BY_NAME:
        valid = ", ".join(NODE_ORDER)
        raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")
    return node_name


def _plan_nodes_for_target(target_node: str) -> list[str]:
    return [*NODES_BY_NAME[target_node].prerequisites, target_node]


def _parse_model_overrides(
    raw_values: list[str] | tuple[str, ...] | None,
    *,
    option_name: str,
) -> tuple[Optional[str], dict[str, str], list[str]]:
    """Parse repeated --llm-* values into (global_default, per_node_overrides, warnings)."""
    values = list(raw_values or [])
    global_model: Optional[str] = None
    node_models: dict[str, str] = {}
    warnings: list[str] = []

    for raw in values:
        token = str(raw).strip()
        if not token:
            continue

        if "=" not in token:
            if global_model is None:
                global_model = token
                continue
            if global_model != token:
                raise ValueError(
                    f"Conflicting global values for {option_name}: '{global_model}' and '{token}'."
                )
            continue

        raw_node, raw_model = token.split("=", 1)
        node = normalize_node_name(raw_node)
        model = raw_model.strip()
        if not node:
            raise ValueError(
                f"Invalid {option_name} value '{token}'. Expected 'NODE=MODEL' or 'MODEL'."
            )
        if not model:
            raise ValueError(
                f"Invalid {option_name} value '{token}'. Expected non-empty model after '='."
            )
        if node not in NODES_BY_NAME:
            valid = ", ".join(NODE_ORDER)
            raise ValueError(
                f"Unknown node '{raw_node}' in {option_name}. Valid options: {valid}."
            )
        if node in node_models and node_models[node] != model:
            warnings.append(
                f"{option_name}: duplicate override for '{node}', last value '{model}' wins."
            )
        node_models[node] = model

    return global_model, node_models, warnings


def _resolve_runtime_llm_overrides(
    *,
    target_node: str,
    legacy_model: str,
    llm_model_values: list[str] | tuple[str, ...] | None,
    llm_fallback_values: list[str] | tuple[str, ...] | None,
) -> tuple[str, dict[str, dict[str, str]], list[str]]:
    """Build canonical runtime overrides payload consumed by graph/LLM layer."""
    warnings: list[str] = []
    global_model_from_flag, model_overrides, model_warnings = _parse_model_overrides(
        llm_model_values,
        option_name="--llm-model",
    )
    global_fallback_from_flag, fallback_overrides, fallback_warnings = _parse_model_overrides(
        llm_fallback_values,
        option_name="--llm-fallback",
    )
    warnings.extend(model_warnings)
    warnings.extend(fallback_warnings)

    if (
        global_model_from_flag is not None
        and legacy_model != DEFAULT_PRIMARY_LLM
        and legacy_model != global_model_from_flag
    ):
        raise ValueError(
            "Conflicting global model values from --model and --llm-model. "
            "Use one global model or make them equal."
        )

    global_primary = global_model_from_flag or legacy_model or DEFAULT_PRIMARY_LLM
    global_fallback = global_fallback_from_flag or DEFAULT_FALLBACK_LLM

    branch_nodes = _plan_nodes_for_target(target_node)
    branch_set = set(branch_nodes)

    resolved_models = {node: global_primary for node in branch_nodes}
    resolved_fallbacks = {node: global_fallback for node in branch_nodes}

    for node, model in model_overrides.items():
        if node not in branch_set:
            warnings.append(
                f"Ignoring --llm-model for '{node}' because it is not in target branch '{target_node}'."
            )
            continue
        resolved_models[node] = model

    for node, model in fallback_overrides.items():
        if node not in branch_set:
            warnings.append(
                f"Ignoring --llm-fallback for '{node}' because it is not in target branch '{target_node}'."
            )
            continue
        resolved_fallbacks[node] = model

    return global_primary, {"models": resolved_models, "fallback_models": resolved_fallbacks}, warnings


def _set_case_llm_overrides(case: Any, llm_overrides: dict[str, dict[str, str]]) -> Any:
    """Attach runtime llm_overrides to an EvalCase-like object."""
    if isinstance(case, dict):
        updated = dict(case)
        updated["llm_overrides"] = llm_overrides
        return updated

    if hasattr(case, "model_copy"):
        try:
            return case.model_copy(update={"llm_overrides": llm_overrides})
        except Exception:
            pass

    try:
        setattr(case, "llm_overrides", llm_overrides)
    except Exception:
        return case
    return case


def _print_llm_routing_summary(
    *,
    target_node: str,
    global_primary: str,
    llm_overrides: dict[str, dict[str, str]],
) -> None:
    models = llm_overrides.get("models", {})
    fallbacks = llm_overrides.get("fallback_models", {})
    branch_nodes = _plan_nodes_for_target(target_node)

    table = Table(title=f"LLM Routing (target={target_node})", show_header=True)
    table.add_column("Node", style="cyan")
    table.add_column("Primary", style="green")
    table.add_column("Fallback", style="yellow")
    for node in branch_nodes:
        table.add_row(
            node,
            models.get(node, global_primary),
            fallbacks.get(node, DEFAULT_FALLBACK_LLM),
        )
    console.print(table)

def _print_cost_table(
    cost,
    total_records: int = 0,
    highlight_nodes: Optional[set[str]] = None,
    visible_nodes: Optional[set[str]] = None,
    target_node: Optional[str] = None,
) -> None:
    """Render cost table and optionally highlight the selected target branch."""
    cost.print_table(
        title="Cost Estimate (Uncached Work)",
        total_records=total_records,
        highlight_nodes=highlight_nodes,
        visible_nodes=visible_nodes,
        target_node=target_node,
    )


def _print_scan_table(meta) -> None:
    table = Table(title="Scan Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Source", style="dim")

    avg_tokens = (meta.total_tokens / meta.record_count) if meta.record_count else 0

    table.add_row("Records scanned", f"{meta.record_count:,}", "dataset adapter")
    table.add_row("Total tokens", f"{meta.total_tokens:,}", "tiktoken (cl100k_base)")
    table.add_row("  Question tokens", f"{meta.question_tokens:,}", "question field")
    table.add_row("  Generation tokens", f"{meta.generation_tokens:,}", "generation field")
    table.add_row("  Context tokens", f"{meta.context_tokens:,}", "context passages")
    table.add_row("  GEval tokens", f"{meta.geval_tokens:,}", "geval criteria text")
    table.add_row("Generation chunks", f"{meta.generation_chunk_count:,}", "chunker output")
    table.add_row("Avg tokens / record", f"{avg_tokens:.1f}", "total_tokens / records")
    console.print(table)


def _print_node_eligibility_table(meta) -> None:
    total = meta.record_count or 0
    if total == 0:
        return

    cm = meta.cost_meta
    node_eligible = {
        "claims": cm.claim.eligible_records,
        "grounding": cm.grounding.eligible_records,
        "relevance": cm.relevance.eligible_records,
        "geval_steps": cm.geval_steps.eligible_records,
        "geval": cm.geval.eligible_records,
        "redteam": cm.readteam.eligible_records,
        "reference": cm.reference.eligible_records,
    }

    table = Table(title="Node Eligibility (By Record)", show_header=True)
    table.add_column("Node", style="cyan")
    table.add_column("Eligible Records", justify="right")
    table.add_column("Coverage", justify="right")

    for node, eligible in node_eligible.items():
        pct = (eligible / total * 100.0) if total else 0.0
        table.add_row(node, f"{eligible:,} / {total:,}", f"{pct:.1f}%")
    console.print(table)


def _print_execution_plan_table(plan, total_records: int) -> None:
    """Render cache-aware execution plan for the selected target branch."""
    table = Table(title=f"Execution Plan (target={plan.target_node})", show_header=True)
    table.add_column("Node", style="cyan")
    table.add_column("To Run", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Skipped", justify="right")

    for node in plan.planned_nodes:
        to_run = plan.to_run_count(node)
        cached = plan.cached_count(node)
        skipped = plan.skipped_count(node)
        table.add_row(
            node,
            f"{to_run:,} / {total_records:,}",
            f"{cached:,} / {total_records:,}",
            f"{skipped:,} / {total_records:,}",
        )
    console.print(table)


@app.command()
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
            "Maximum number of submitted-but-not-yet-emitted records. "
            "Defaults to max_workers * 2."
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
) -> None:
    """Execute selected cases up to `node_name` without preflight prompts.

    Examples:
      # Default routing (primary=openai/gpt-4o-mini, fallback=openai/gpt-4o)
      lumiseval run grounding --input sample.json

      # Set one global primary for the branch
      lumiseval run grounding --input sample.json --llm-model openai/gpt-4o

      # Override one node primary model only
      lumiseval run grounding --input sample.json --llm-model grounding=openai/gpt-4o

      # Override one node primary + fallback
      lumiseval run grounding --input sample.json \
        --llm-model grounding=openai/gpt-4o \
        --llm-fallback grounding=openai/gpt-4o-mini

      # Global primary + per-node exception
      lumiseval run grounding --input sample.json \
        --llm-model openai/gpt-4o \
        --llm-model grounding=openai/gpt-4o-mini

      # Backward-compatible global primary alias
      lumiseval run eval --input sample.json --model openai/gpt-4o-mini

      # Bounded concurrent streaming over large datasets
      lumiseval run eval --input sample.json --max-workers 8 --max-in-flight 32
    """
    try:
        target_node = _resolve_target_node(node_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
        
    if yes:
        console.print(
            "[dim]`--yes` is deprecated for `run`; execution now starts immediately.[/dim]"
        )

    try:
        effective_judge_model, llm_overrides, llm_warnings = _resolve_runtime_llm_overrides(
            target_node=target_node,
            legacy_model=judge_model,
            llm_model_values=llm_model,
            llm_fallback_values=llm_fallback,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

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

    # console.print(f"\n[cyan]Running '{target_node}' on {len(cases)} case(s)...[/cyan]\n")
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
    cases = (_set_case_llm_overrides(case, llm_overrides) for case in selected_cases)

    try:
        for outcome in runner.run_cases_iter(
            cases=cases,
            node_name=target_node,
            force=force,
            execution_mode="run",
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            continue_on_error=continue_on_error,
        ):
            if outcome.result is not None:
                result = outcome.result
                succeeded += 1
                total_executed += len(result.executed_nodes)
                total_cached += len(result.cached_nodes)

                if output_dir:
                    report = result.final_state.get("report")
                    if report is not None:
                        out_path = output_dir / f"{_slug(result.case_id)}.json"
                        out_path.write_text(
                            report.model_dump_json(indent=2)
                            if hasattr(report, "model_dump_json")
                            else json.dumps(_to_jsonable(report), indent=2)
                        )
                continue

            failed += 1
            failures.append((outcome.case_id, outcome.error or "unknown error"))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
