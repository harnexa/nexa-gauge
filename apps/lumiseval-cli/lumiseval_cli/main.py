"""
LumisEval CLI — Typer-based command-line interface.

Commands:
  lumiseval eval      — evaluate a single generation from a file or stdin
  lumiseval estimate  — compute a cost estimate without running evaluation
  lumiseval batch     — evaluate a JSON/JSONL/CSV dataset file

TODO:
  - lumiseval batch  — async dispatch via TaskIQ with streamed progress.
  - lumiseval index  — pre-index reference documents into local LanceDB.
  - Interactive cost confirmation prompt before long runs.
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from lumiseval_agent.graph import run_graph
from lumiseval_agent.nodes.cost_estimator import estimate as compute_estimate
from lumiseval_core.types import EvalJobConfig
from lumiseval_ingest.scanner import scan_file, scan_text

app = typer.Typer(name="lumiseval", help="Agentic LLM evaluation pipeline.")
console = Console()


@app.command()
def eval(
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to generation text file."),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Query that produced the generation."),
    web_search: bool = typer.Option(False, "--web-search", help="Enable Tavily web search."),
    adversarial: bool = typer.Option(False, "--adversarial", help="Run adversarial / guardrail probes."),
    judge_model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LiteLLM judge model."),
    budget_cap: Optional[float] = typer.Option(None, "--budget-cap", help="USD budget cap. Job rejected if estimate exceeds."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON report to this path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost confirmation prompt."),
) -> None:
    """Evaluate a single LLM generation."""
    if input:
        generation = input.read_text()
    else:
        console.print("[yellow]Reading generation from stdin (Ctrl+D to finish)...[/yellow]")
        generation = sys.stdin.read()

    if not generation.strip():
        console.print("[red]Error: empty generation.[/red]")
        raise typer.Exit(1)

    # Pre-run cost estimate
    meta = scan_text(generation)
    job_config = EvalJobConfig(
        job_id=str(uuid.uuid4()),
        judge_model=judge_model,
        web_search=web_search,
        adversarial=adversarial,
        budget_cap_usd=budget_cap,
    )

    try:
        cost = compute_estimate(meta, job_config)
        _print_cost_table(cost)
    except Exception as exc:
        console.print(f"[red]Budget exceeded: {exc}[/red]")
        raise typer.Exit(1)

    if not yes:
        proceed = typer.confirm(f"\nEstimated cost: ${cost.total_estimated_usd:.4f}. Proceed?")
        if not proceed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    console.print("\n[cyan]Running evaluation...[/cyan]")
    try:
        report = run_graph(
            generation=generation,
            job_config=job_config,
            question=question,
        )
    except RuntimeError as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        raise typer.Exit(1)

    _print_report_summary(report)

    if output:
        output.write_text(report.model_dump_json(indent=2))
        console.print(f"\n[green]Report written to {output}[/green]")


@app.command()
def estimate(
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Path to generation file or dataset."),
    web_search: bool = typer.Option(False, "--web-search"),
    judge_model: str = typer.Option("gpt-4o-mini", "--model", "-m"),
) -> None:
    """Compute a pre-run cost estimate without running any evaluation."""
    if input:
        meta = scan_file(input)
    else:
        console.print("[yellow]Reading from stdin...[/yellow]")
        generation = sys.stdin.read()
        meta = scan_text(generation)

    job_config = EvalJobConfig(
        job_id="estimate-only",
        judge_model=judge_model,
        web_search=web_search,
    )
    cost = compute_estimate(meta, job_config)
    _print_cost_table(cost)


@app.command()
def batch(
    dataset: Path = typer.Argument(..., help="Path to JSON/JSONL/CSV dataset file."),
    judge_model: str = typer.Option("gpt-4o-mini", "--model", "-m"),
    web_search: bool = typer.Option(False, "--web-search"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Evaluate a dataset file (JSON/JSONL/CSV). Each record must have a 'generation' field.

    TODO: Implement async batch dispatch via TaskIQ with streamed progress.
    """
    console.print(f"[yellow]Batch mode: scanning {dataset}...[/yellow]")
    meta = scan_file(dataset)
    console.print(
        f"Found [bold]{meta.record_count}[/bold] records, "
        f"~{meta.estimated_claim_count} total claims."
    )

    job_config = EvalJobConfig(
        job_id=str(uuid.uuid4()),
        judge_model=judge_model,
        web_search=web_search,
    )
    cost = compute_estimate(meta, job_config)
    _print_cost_table(cost)

    if not yes:
        proceed = typer.confirm(f"\nEstimated cost: ${cost.total_estimated_usd:.4f}. Proceed?")
        if not proceed:
            raise typer.Exit(0)

    console.print("[red]Batch execution not yet implemented. Coming soon.[/red]")
    raise typer.Exit(0)


def _print_cost_table(cost) -> None:
    table = Table(title="Cost Estimate", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_row("Judge calls", str(cost.estimated_judge_calls), f"${cost.judge_cost_usd:.6f}")
    table.add_row("Embedding calls", str(cost.estimated_embedding_calls), f"${cost.embedding_cost_usd:.6f}")
    table.add_row("Tavily searches", str(cost.estimated_tavily_calls), f"${cost.tavily_cost_usd:.6f}")
    table.add_row("[bold]Total[/bold]", "", f"[bold]${cost.total_estimated_usd:.6f}[/bold]")
    console.print(table)
    console.print(f"  Confidence range: ${cost.low_usd:.6f} – ${cost.high_usd:.6f} (±20%)")
    if cost.approximate_warning:
        console.print(f"  [yellow][ESTIMATE APPROXIMATE] {cost.approximate_warning}[/yellow]")


def _print_report_summary(report) -> None:
    console.print("\n[bold green]Evaluation Complete[/bold green]")
    console.print(f"  Job ID         : {report.job_id}")
    console.print(f"  Composite score: {report.composite_score}")
    console.print(f"  Confidence band: ±{report.confidence_band}")
    console.print(f"  Claims evaluated: {len(report.claim_verdicts)}")
    console.print(f"  Cost actual    : ${report.cost_actual_usd:.6f}")
    if report.warnings:
        for w in report.warnings:
            console.print(f"  [yellow]Warning: {w}[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
