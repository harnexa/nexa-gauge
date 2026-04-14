from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from lumoseval_core.types import CostEstimate
from lumoseval_graph.llm.config import get_judge_model, normalize_node_name
from lumoseval_graph.topology import NODE_ORDER, NODES_BY_NAME
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_PRIMARY_LLM = "openai/gpt-4o-mini"
DEFAULT_FALLBACK_LLM = "openai/gpt-4o"


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
                f"{executed:,} / {successful_cases:,}",
                f"{cached:,} / {successful_cases:,}",
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
