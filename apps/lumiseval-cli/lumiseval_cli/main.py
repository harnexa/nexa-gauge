"""
LumisEval CLI — single-command interface.

Public command:
  lumiseval run <node_name> --input <source>

Flow:
  1) Load dataset cases from local file or hf:// source
  2) Scan selected cases and show scan statistics
  3) Estimate total cost and ask for confirmation (unless --yes)
  4) Execute each case up to target node with cache reuse
"""

import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import typer
from lumiseval_agent.nodes.cost_estimator import estimate as compute_estimate
from lumiseval_agent.runners.node_runner import CachedNodeRunner, NodeRunner
from lumiseval_core.cache import CacheStore, NoOpCacheStore
from lumiseval_core.errors import InputParseError
from lumiseval_core.types import EvalJobConfig
from lumiseval_ingest import create_dataset_adapter
from lumiseval_ingest.scanner import scan_cases
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


def _print_cost_table(cost) -> None:
    table = Table(title="Cost Estimate", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_row("Judge calls", str(cost.estimated_judge_calls), f"${cost.judge_cost_usd:.6f}")
    table.add_row(
        "Embedding calls", str(cost.estimated_embedding_calls), f"${cost.embedding_cost_usd:.6f}"
    )
    table.add_row(
        "Tavily searches", str(cost.estimated_tavily_calls), f"${cost.tavily_cost_usd:.6f}"
    )
    table.add_row("[bold]Total[/bold]", "", f"[bold]${cost.total_estimated_usd:.6f}[/bold]")
    console.print(table)
    console.print(f"  Confidence range: ${cost.low_usd:.6f} – ${cost.high_usd:.6f} (±20%)")
    if cost.approximate_warning:
        console.print(f"  [yellow][ESTIMATE APPROXIMATE] {cost.approximate_warning}[/yellow]")


def _print_scan_table(meta) -> None:
    table = Table(title="Scan Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    avg_tokens = (meta.total_tokens / meta.record_count) if meta.record_count else 0
    avg_chars = (meta.total_chars / meta.record_count) if meta.record_count else 0

    table.add_row("Records scanned", f"{meta.record_count:,}")
    table.add_row("Total tokens", f"{meta.total_tokens:,}")
    table.add_row("Total characters", f"{meta.total_chars:,}")
    table.add_row("Estimated chunks", f"{meta.estimated_chunk_count:,}")
    table.add_row("Estimated claims", f"{meta.estimated_claim_count:,}")
    table.add_row("Avg tokens / record", f"{avg_tokens:.1f}")
    table.add_row("Avg chars / record", f"{avg_chars:.1f}")
    console.print(table)


def _print_node_eligibility_table(meta) -> None:
    counts = meta.eligible_record_count or {}
    total = meta.record_count or 0
    if not counts or total == 0:
        return

    table = Table(title="Node Eligibility (By Record)", show_header=True)
    table.add_column("Node", style="cyan")
    table.add_column("Eligible Records", justify="right")
    table.add_column("Coverage", justify="right")

    node_order = [
        "scan",
        "estimate",
        "approve",
        "chunk",
        "claims",
        "dedupe",
        "relevance",
        "grounding",
        "redteam",
        "rubric",
        "eval",
    ]
    for node in node_order:
        eligible = int(counts.get(node, 0))
        pct = (eligible / total * 100.0) if total else 0.0
        table.add_row(node, f"{eligible:,} / {total:,}", f"{pct:.1f}%")
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
    enable_hallucination: bool = typer.Option(
        True,
        "--enable-hallucination/--disable-hallucination",
        help="Enable grounding node (hallucination metric).",
    ),
    enable_faithfulness: bool = typer.Option(
        True,
        "--enable-faithfulness/--disable-faithfulness",
        help="Enable faithfulness metric in relevance node.",
    ),
    enable_answer_relevancy: bool = typer.Option(
        True,
        "--enable-answer-relevancy/--disable-answer-relevancy",
        help="Enable answer relevancy metric in relevance node.",
    ),
    enable_adversarial: bool = typer.Option(
        False,
        "--enable-adversarial/--disable-adversarial",
        help="Enable redteam node.",
    ),
    enable_rubric: bool = typer.Option(
        False,
        "--enable-rubric/--disable-rubric",
        help="Enable rubric node.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
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
    """Run selected cases up to target node, with preflight scan/cost confirmation."""
    console.print(f"[yellow]Preparing source: {input}[/yellow]")
    target_node = NodeRunner.normalize_node_name(node_name)
    if target_node not in NodeRunner._node_fns:
        valid = ", ".join(sorted(NodeRunner._node_fns))
        console.print(f"[red]Unknown node '{node_name}'. Valid options: {valid}.[/red]")
        raise typer.Exit(1)

    try:
        ds_adapter = create_dataset_adapter(
            source=input,
            adapter=adapter,
            config_name=hf_config,
            revision=hf_revision,
        )
    except InputParseError as exc:
        console.print(f"[red]Invalid dataset source: {exc}[/red]")
        raise typer.Exit(1)

    try:
        cases = list(ds_adapter.iter_cases(split=split, limit=limit))
    except Exception as exc:
        console.print(f"[red]Failed to load cases: {exc}[/red]")
        raise typer.Exit(1)

    if not cases:
        console.print("[yellow]No cases found after applying split/limit.[/yellow]")
        raise typer.Exit(0)

    # ── Stage A: dataset-level preflight (scan + estimate + confirm) ───────
    console.print("[cyan]Scanning selected cases...[/cyan]")
    meta = scan_cases(cases, show_progress=True)
    _print_scan_table(meta)
    _print_node_eligibility_table(meta)

    base_job_config = EvalJobConfig(
        job_id=str(uuid.uuid4()),
        judge_model=judge_model,
        web_search=web_search,
        evidence_threshold=evidence_threshold,
        enable_hallucination=enable_hallucination,
        enable_faithfulness=enable_faithfulness,
        enable_answer_relevancy=enable_answer_relevancy,
        enable_adversarial=enable_adversarial,
        enable_rubric=enable_rubric,
    )

    console.print("\n[cyan]Estimating cost...[/cyan]")
    rubric_rule_count = sum(len(case.rubric_rules) for case in cases)
    try:
        cost = compute_estimate(
            meta,
            base_job_config,
            target_node=target_node,
            rubric_rule_count=rubric_rule_count,
        )
    except Exception as exc:
        console.print(f"[red]Cost estimation failed: {exc}[/red]")
        raise typer.Exit(1)
    _print_cost_table(cost)

    if not yes:
        proceed = typer.confirm(f"\nEstimated cost: ${cost.total_estimated_usd:.4f}. Proceed?")
        if not proceed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    # ── Stage B: per-case execution to strict target node ───────────────────
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
                job_config=base_job_config,
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

        # status = (
        #     f"[green]executed={len(result.executed_nodes)}[/green]  "
        #     f"[blue]cached={len(result.cached_nodes)}[/blue]  "
        #     f"({result.elapsed_ms:.0f}ms)"
        # )
        # console.print(f"  [{result.case_id}] {status}")

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
