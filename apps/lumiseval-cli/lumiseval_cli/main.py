"""LumisEval CLI.

Public commands:
  - ``lumiseval estimate <node_name> --input <source>``: scan + plan + delta-cost only
  - ``lumiseval run <node_name> --input <source>``: execute branch directly
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import typer
from lumiseval_core.cache import CacheStore, NoOpCacheStore
from lumiseval_core.errors import InputParseError
from lumiseval_core.pipeline import NODE_ORDER, NODES_BY_NAME
from lumiseval_core.types import EvalCase, EvalJobConfig
from lumiseval_graph import CachedNodeRunner, estimate_preflight
from lumiseval_ingest import create_dataset_adapter
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="lumiseval", help="Agentic LLM evaluation pipeline.")
console = Console()


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


def _load_cases(
    *,
    input_source: str,
    split: str,
    limit: Optional[int],
    adapter: str,
    hf_config: Optional[str],
    hf_revision: Optional[str],
) -> list[EvalCase]:
    """Load selected cases from local files or HF adapters."""
    ds_adapter = create_dataset_adapter(
        source=input_source,
        adapter=adapter,
        config_name=hf_config,
        revision=hf_revision,
    )
    return list(ds_adapter.iter_cases(split=split, limit=limit))


def _build_job_config(
    *,
    judge_model: str,
    web_search: bool,
    evidence_threshold: float,
) -> EvalJobConfig:
    """Create the CLI's default job config.

    All metric toggles are enabled so targeted branch runs and cost previews
    can include every metric path when eligible for the selected target.
    """
    return EvalJobConfig(
        job_id=str(uuid.uuid4()),
        judge_model=judge_model,
        web_search=web_search,
        evidence_threshold=evidence_threshold,
        enable_grounding=True,
        enable_relevance=True,
        enable_redteam=True,
        enable_geval=True,
        enable_reference=True,
    )


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
def estimate(
    node_name: str = typer.Argument(..., help="Target node name (e.g. relevance, eval)."),
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Dataset source: local file path or hf://<dataset-id>.",
    ),
    split: str = typer.Option("train", "--split", help="Dataset split for adapter-backed sources."),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        min=1,
        help="Maximum number of cases to include from the source.",
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
    judge_model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LiteLLM judge model."),
    web_search: bool = typer.Option(False, "--web-search", help="Enable Tavily web search."),
    evidence_threshold: float = typer.Option(0.75, "--evidence-threshold"),
    force: bool = typer.Option(False, "--force", help="Ignore cache reads in planning."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache reads and writes."),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="Cache directory path."),
) -> None:
    """Estimate branch cost without executing graph nodes."""
    try:
        target_node = _resolve_target_node(node_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Preparing source: {input}[/yellow]")
    try:
        cases: list[EvalCase] = _load_cases(
            input_source=input,
            split=split,
            limit=limit,
            adapter=adapter,
            hf_config=hf_config,
            hf_revision=hf_revision,
        )
    except InputParseError as exc:
        console.print(f"[red]Invalid dataset source: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Failed to load cases: {exc}[/red]")
        raise typer.Exit(1)

    if not cases:
        console.print("[yellow]No cases found after applying split/limit.[/yellow]")
        raise typer.Exit(0)

    job_config = _build_job_config(
        judge_model=judge_model,
        web_search=web_search,
        evidence_threshold=evidence_threshold,
    )
    cache_store: CacheStore = NoOpCacheStore() if no_cache else CacheStore(cache_dir)
    runner = CachedNodeRunner(cache_store=cache_store)

    console.print("[cyan]Scanning selected cases...[/cyan]")
    try:
        preflight = estimate_preflight(
            cases=cases,
            target_node=target_node,
            runner=runner,
            job_config=job_config,
            force=force,
            show_progress=True,
        )
    except Exception as exc:
        console.print(f"[red]Estimate failed: {exc}[/red]")
        raise typer.Exit(1)

    _print_scan_table(preflight.metadata)
    _print_node_eligibility_table(preflight.metadata)
    _print_execution_plan_table(preflight.plan, total_records=preflight.metadata.record_count)
    _print_cost_table(
        preflight.cost_report,
        total_records=preflight.metadata.record_count,
        highlight_nodes=set(preflight.plan.planned_nodes),
        visible_nodes=set(preflight.plan.planned_nodes),
        target_node=target_node,
    )
    target_row = preflight.cost_report.row(target_node)
    console.print(
        f"\n[bold green]Estimate complete.[/bold green] "
        f"target={target_node} delta_cost=${target_row.cost_usd:.6f}"
    )


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
    judge_model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LiteLLM judge model."),
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
    force: bool = typer.Option(False, "--force", help="Ignore cache reads (still writes)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache reads and writes."),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="Cache directory path."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write per-case JSON output here (eval only).",
    ),
) -> None:
    """Execute selected cases up to `node_name` without preflight prompts."""
    try:
        target_node = _resolve_target_node(node_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
        
    if yes:
        console.print(
            "[dim]`--yes` is deprecated for `run`; execution now starts immediately.[/dim]"
        )

    console.print(f"[yellow]Preparing source: {input}[/yellow]")
    try:
        cases: list[EvalCase] = _load_cases(
            input_source=input,
            split=split,
            limit=limit,
            adapter=adapter,
            hf_config=hf_config,
            hf_revision=hf_revision,
        )
    except InputParseError as exc:
        console.print(f"[red]Invalid dataset source: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Failed to load cases: {exc}[/red]")
        raise typer.Exit(1)

    if not cases:
        console.print("[yellow]No cases found after applying split/limit.[/yellow]")
        raise typer.Exit(0)

    job_config = _build_job_config(
        judge_model=judge_model,
        web_search=web_search,
        evidence_threshold=evidence_threshold,
    )
    cache_store: CacheStore = NoOpCacheStore() if no_cache else CacheStore(cache_dir)
    runner = CachedNodeRunner(cache_store=cache_store)

    if output_dir and target_node == "eval":
        output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[cyan]Running '{target_node}' on {len(cases)} case(s)...[/cyan]\n")
    total_executed = 0
    total_cached = 0
    succeeded = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for case in cases:
        try:
            result = runner.run_case(
                case=case,
                node_name=target_node,
                job_config=job_config,
                force=force,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        except Exception as exc:
            failed += 1
            failures.append((case.case_id, str(exc)))
            console.print(f"  [{case.case_id}] [red]failed[/red]  {exc}")
            if not continue_on_error:
                raise typer.Exit(1)
            continue

        succeeded += 1
        total_executed += len(result.executed_nodes)
        total_cached += len(result.cached_nodes)

        if output_dir and target_node == "eval":
            report = result.final_state.get("report")
            if report is not None:
                out_path = output_dir / f"{_slug(result.case_id)}.json"
                out_path.write_text(
                    report.model_dump_json(indent=2)
                    if hasattr(report, "model_dump_json")
                    else json.dumps(_to_jsonable(report), indent=2)
                )

    console.print(
        f"\n[bold green]Done.[/bold green]  "
        f"cases={len(cases)}  succeeded={succeeded}  failed={failed}  "
        f"executed_steps={total_executed}  cached_steps={total_cached}"
    )

    if failures:
        console.print("\n[yellow]Failures:[/yellow]")
        for case_id, err in failures[:10]:
            console.print(f"  - {case_id}: {err}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
