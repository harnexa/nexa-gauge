from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional

from ng_core.constants import DEFAULT_FALLBACK_LLM, DEFAULT_PRIMARY_LLM
from ng_core.types import CostEstimate
from ng_graph.llm.config import get_judge_model, normalize_node_name
from ng_graph.nodes.scanner import scan
from ng_graph.topology import NODE_ORDER, NODES_BY_NAME, transitive_prerequisites
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()


def _progress_total_from_bounds(*, start: int, end: int | None) -> int | None:
    if end is None:
        return None
    return max(0, end - start)


@contextmanager
def _case_progress(
    *,
    enabled: bool,
    description: str,
    total: int | None,
) -> Iterator[Callable[[], None]]:
    """Render a lightweight CLI progress bar and yield an `advance()` callback."""
    if not enabled:
        yield lambda: None
        return

    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=24),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    )
    progress.start()
    task_id = progress.add_task(description, total=total)

    def _advance() -> None:
        progress.advance(task_id, 1)

    try:
        yield _advance
    finally:
        progress.stop()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned or "case"


def _print_node_timings_summary(timings_by_case: list[Mapping[str, float]]) -> None:
    """Render a per-node latency table (count/p50/p95/sum) across cases.

    Cache hits are recorded as ``0.0`` in the underlying timings dict; they
    are excluded here so the stats reflect real execution latency. A separate
    ``cached`` count shows how many cases hit cache for each node.
    """
    if not timings_by_case:
        return

    observed: set[str] = set()
    for t in timings_by_case:
        observed.update(t.keys())
    if not observed:
        return
    ordered = [n for n in NODE_ORDER if n in observed]

    def _percentile(samples: list[float], pct: float) -> float:
        # Sorted-input nearest-rank percentile. `samples` must be sorted.
        if not samples:
            return 0.0
        idx = max(0, min(len(samples) - 1, int(round(pct * (len(samples) - 1)))))
        return samples[idx]

    table = Table(
        title=f"per-node timings across {len(timings_by_case)} case(s)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("node", style="bold")
    table.add_column("ran", justify="right")
    table.add_column("cached", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("sum ms", justify="right")

    for node in ordered:
        raw = [float(t.get(node, -1.0)) for t in timings_by_case if node in t]
        ran = sorted(s for s in raw if s > 0.0)
        cached_count = sum(1 for s in raw if s == 0.0)
        if not ran and cached_count == 0:
            continue
        p50 = _percentile(ran, 0.50) if ran else 0.0
        p95 = _percentile(ran, 0.95) if ran else 0.0
        total = sum(ran)
        color = NODES_BY_NAME[node].color if node in NODES_BY_NAME else "white"
        table.add_row(
            f"[{color}]{node}[/{color}]",
            str(len(ran)),
            str(cached_count),
            f"{p50:.1f}" if ran else "—",
            f"{p95:.1f}" if ran else "—",
            f"{total:.1f}" if ran else "—",
        )

    console.print()
    console.print(table)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _resolve_target_node(node_name: str) -> str:
    valid_targets = ", ".join(
        n
        for n in NODE_ORDER
        if NODES_BY_NAME[n].is_utility or NODES_BY_NAME[n].is_metric or n == "eval"
    )
    if node_name not in NODES_BY_NAME:
        raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid_targets}.")
    spec = NODES_BY_NAME[node_name]
    if not (spec.is_utility or spec.is_metric or node_name == "eval"):
        raise ValueError(
            f"Node '{node_name}' is not directly invocable from the CLI. "
            f"Only utility, metric, and eval nodes are valid targets: {valid_targets}."
        )
    return node_name


def _plan_nodes_for_target(target_node: str) -> list[str]:
    return [*transitive_prerequisites(target_node), target_node]


def _parse_model_overrides(
    raw_values: list[str] | tuple[str, ...] | None,
    *,
    option_name: str,
) -> tuple[Optional[str], dict[str, str], list[str]]:
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
            raise ValueError(f"Unknown node '{raw_node}' in {option_name}. Valid options: {valid}.")
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

    return (
        global_primary,
        {"models": resolved_models, "fallback_models": resolved_fallbacks},
        warnings,
    )


def _set_case_llm_overrides(case: Any, llm_overrides: dict[str, dict[str, str]]) -> Any:
    if isinstance(case, dict):
        updated = dict(case)
        updated["llm_overrides"] = llm_overrides
        return updated

    if hasattr(case, "model_copy"):
        try:
            return case.model_copy(update={"llm_overrides": llm_overrides})
        except Exception:
            # pydantic may reject unknown fields on strict models; fall through to setattr.
            pass

    try:
        setattr(case, "llm_overrides", llm_overrides)
    except Exception:
        # Frozen/slots-only objects can't accept new attrs; return case untouched.
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


def _format_cost(value: float) -> str:
    return f"${value:,.6f}"


def _resolve_estimate_node_status(
    *,
    node_cost: float,
    executed: int,
    cached: int,
    estimated: int,
    successful_cases: int,
) -> str:
    if successful_cases <= 0:
        return "failed"
    if node_cost > 0.0:
        return "billable"
    if estimated > 0:
        return "zero_cost"
    if cached > 0 and executed == 0:
        return "cached_only"
    if executed > 0:
        return "skipped/ineligible"
    return "not_reached"


def _is_node_eligible_for_inputs(node_name: str, inputs: Any) -> bool:
    if node_name in {"scan", "eval"}:
        return True
    if inputs is None:
        return False

    spec = NODES_BY_NAME[node_name]
    if spec.requires_generation and not bool(getattr(inputs, "has_generation", False)):
        return False
    if spec.requires_context and not bool(getattr(inputs, "has_context", False)):
        return False
    if spec.requires_question and not bool(getattr(inputs, "has_question", False)):
        return False
    if spec.requires_geval and not bool(getattr(inputs, "has_geval", False)):
        return False
    if spec.requires_reference and not bool(getattr(inputs, "has_reference", False)):
        return False
    return True


def _scan_inputs_from_case(case: Any) -> Any:
    if isinstance(case, Mapping):
        raw_case = case
    elif hasattr(case, "model_dump"):
        dumped = case.model_dump()
        raw_case = dumped if isinstance(dumped, Mapping) else {}
    else:
        raw_case = {}

    scanned = scan(raw_case)
    return scanned.get("inputs") if isinstance(scanned, Mapping) else None


def _is_case_eligible_for_target_path(target_node: str, case: Any) -> bool:
    """Return whether a case satisfies every node requirement in target branch path.

    For eval/report targets, keep existing behavior (process all records), because
    requiring the full eval path would incorrectly force all metric-specific fields
    (question/context/reference/geval) to be present at once.
    """
    if target_node in {"eval", "report"}:
        return True

    inputs = _scan_inputs_from_case(case)
    if inputs is None:
        return False

    for node_name in _plan_nodes_for_target(target_node):
        if not _is_node_eligible_for_inputs(node_name, inputs):
            return False
    return True


def _collect_estimate_rows(
    *,
    target_node: str,
    cost_by_node: dict[str, CostEstimate],
    node_stats: dict[str, dict[str, int]],
    total_selected_cases: int,
    successful_cases: int,
    effective_judge_model: str,
    llm_overrides: dict[str, dict[str, str]],
) -> list[tuple[str, str, str, str, str, str, str, float]]:
    rows: list[tuple[str, str, str, str, str, str, str, float]] = []
    for node_name in _plan_nodes_for_target(target_node):
        spec = NODES_BY_NAME[node_name]
        if not (spec.is_metric or spec.is_utility or spec.is_preflight):
            continue
        est = cost_by_node.get(node_name) or CostEstimate(
            cost=0.0, input_tokens=None, output_tokens=None
        )
        node_cost = float(est.cost or 0.0)
        stats = node_stats.get(node_name) or {}
        executed = int(stats.get("executed", 0))
        cached = int(stats.get("cached", 0))
        estimated = int(stats.get("estimated", 0))
        eligible_uncached = int(stats.get("eligible_uncached", 0))
        eligible_pct = (
            (eligible_uncached / total_selected_cases * 100.0) if total_selected_cases > 0 else 0.0
        )
        status = _resolve_estimate_node_status(
            node_cost=node_cost,
            executed=executed,
            cached=cached,
            estimated=estimated,
            successful_cases=successful_cases,
        )
        resolved_model = get_judge_model(
            node_name=node_name,
            default=effective_judge_model,
            llm_overrides=llm_overrides,
        )
        rows.append(
            (
                node_name,
                resolved_model,
                status,
                f"{executed:,} / {total_selected_cases:,}",
                f"{cached:,} / {total_selected_cases:,}",
                f"{eligible_uncached:,} / {total_selected_cases:,}",
                f"{eligible_pct:.1f}%",
                node_cost,
            )
        )
    return rows


def _write_report_json(report: Any, output_dir: Path, case_id: str) -> None:
    out_path = output_dir / f"{_slug(case_id)}.json"
    out_path.write_text(
        report.model_dump_json(indent=2)
        if hasattr(report, "model_dump_json")
        else json.dumps(_to_jsonable(report), indent=2)
    )
