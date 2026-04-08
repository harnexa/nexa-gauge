"""
Cost Estimator — computes a cumulative pre-run cost estimate for the pipeline.

Each row in the report shows the *total* cost up to and including that node.
For example, the ``grounding`` row reflects claims + grounding combined, because
grounding cannot run without the claims extraction step that precedes it.

Usage::

    from lumiseval_graph.nodes.cost_estimator import CostEstimator

    estimator = CostEstimator(job_config)
    report = estimator.estimate(metadata)
    for row in report.rows:
        print(row)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

from lumiseval_core.pipeline import METRIC_NODES, NODE_ORDER, NODES_BY_NAME
from lumiseval_core.types import (
    CostEstimate,
    EvalCase,
    EvalJobConfig,
    InputMetadata,
    NodeCostBreakdown,
)
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lumiseval_graph.llm.config import get_judge_model
from lumiseval_graph.nodes.dedup import DedupNode
from lumiseval_graph.nodes.claim_extractor import ClaimExtractorNode
from lumiseval_graph.nodes.metrics.geval import GevalNode, GevalStepsNode
from lumiseval_graph.nodes.metrics.grounding import GroundingNode
from lumiseval_graph.nodes.metrics.redteam import RedteamNode
from lumiseval_graph.nodes.metrics.reference import ReferenceNode
from lumiseval_graph.nodes.metrics.relevance import RelevanceNode

_METRIC_NODES = set(METRIC_NODES)

# ── Output types ──────────────────────────────────────────────────────────────


@dataclass
class NodeCostRow:
    """One row in the cost report, representing cumulative cost up to this node."""

    node_name: str
    model_calls: int  # cumulative across all contributing nodes
    cost_usd: float  # cumulative across all contributing nodes
    source: str  # e.g. "claims + grounding" or "—"
    individual_cost_usd: float = 0.0  # this node's isolated cost (not cumulative)
    eligible_records: int = 0  # records eligible for this node

    def __str__(self) -> str:
        source_lines = self.source.splitlines()
        first = source_lines[0] if source_lines else "—"
        indent = " " * 35
        rest = "".join(f"\n{indent}{line}" for line in source_lines[1:])
        return (
            f"{self.node_name:<12}  calls={self.model_calls:>5}  "
            f"cost=${self.cost_usd:.6f}  source=[{first}]{rest}"
        )


@dataclass
class CostReport:
    """Full pipeline cost report, one row per node in pipeline order."""

    rows: list[NodeCostRow]

    def row(self, node_name: str) -> NodeCostRow:
        """Look up a row by node name."""
        return next(r for r in self.rows if r.node_name == node_name)

    def __str__(self) -> str:
        header = f"{'node':<12}  {'calls':>5}  {'cost_usd':>12}  source"
        sep = "-" * 60
        lines = [header, sep] + [str(r) for r in self.rows]
        return "\n".join(lines)

    def print_table(
        self,
        title: Optional[str] = "Pipeline Cost Estimate",
        total_records: int = 0,
        highlight_nodes: Optional[set[str]] = None,
        visible_nodes: Optional[set[str]] = None,
        target_node: Optional[str] = None,
    ) -> None:
        """Render the report as a pretty Rich table in the terminal.

        Args:
            title: Table title.
            total_records: Dataset size used for optional coverage column.
            highlight_nodes: Optional set of node names to visually emphasize.
                Typical usage is highlighting the strict target branch selected
                by the CLI (for example, scan → chunk → claims → dedup → grounding).
            visible_nodes: Optional set of node names to render. When omitted,
                all rows are shown in pipeline order.
            target_node: Optional node name whose row should display the total
                cumulative cost. When provided, non-target rows in the
                ``Cost (USD)`` column show ``—``.
        """
        highlight_nodes = highlight_nodes or set()
        visible_nodes = visible_nodes or set()
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="bright_black",
            show_lines=True,
        )
        table.add_column("Node", style="bold", min_width=10, no_wrap=True)
        if total_records > 0:
            table.add_column("Coverage", justify="right", min_width=14, no_wrap=True)
        table.add_column("Calls", justify="right", min_width=6, no_wrap=True)
        table.add_column("Node Cost (USD)", justify="right", min_width=18, no_wrap=True)
        table.add_column("Cost (USD)", justify="right", min_width=12, no_wrap=True)
        table.add_column("Breakdown", min_width=55)

        total_cost = self.row("eval").cost_usd

        for r in self.rows:
            if visible_nodes and r.node_name not in visible_nodes:
                continue

            is_zero = r.model_calls == 0
            is_eval = r.node_name == "eval"
            is_metric = r.node_name in _METRIC_NODES
            is_highlighted = r.node_name in highlight_nodes

            if is_eval:
                row_style = "bold"
                cost_str = Text(f"${r.cost_usd:.6f}", style="bold green")
                node_cost_str = Text("—", style="dim")
                node_text = Text(r.node_name, style="bold")
            elif is_zero:
                row_style = "dim"
                cost_str = Text("—", style="dim")
                node_cost_str = Text("—", style="dim")
                node_text = Text(r.node_name, style="dim")
            elif is_metric:
                row_style = ""
                cost_str = Text(f"${r.cost_usd:.6f}", style="yellow")
                pct = (r.individual_cost_usd / total_cost * 100) if total_cost else 0
                node_cost_str = Text(f"${r.individual_cost_usd:.6f}  ({pct:.0f}%)", style="cyan")
                node_text = Text(r.node_name, style="cyan")
            else:
                row_style = ""
                cost_str = Text(f"${r.cost_usd:.6f}", style="green")
                pct = (r.individual_cost_usd / total_cost * 100) if total_cost else 0
                node_cost_str = (
                    Text(f"${r.individual_cost_usd:.6f}  ({pct:.0f}%)", style="green")
                    if r.individual_cost_usd > 0
                    else Text("—", style="dim")
                )
                node_text = Text(r.node_name)

            if is_highlighted:
                # Gentle branch highlight: soft cyan accent, no hard background.
                node_text = Text(f"> {r.node_name}", style="bold bright_cyan")
                row_style = ""
                cost_str = Text(str(cost_str), style="bright_cyan")
                node_cost_str = Text(str(node_cost_str), style="cyan")

            show_cumulative_cost = target_node is None or r.node_name == target_node
            if show_cumulative_cost:
                if target_node is not None:
                    # Explicit target run view: show only the target's total.
                    cost_str = Text(
                        f"${r.cost_usd:.6f}",
                        style="bright_cyan" if is_highlighted else ("bold green" if is_eval else "green"),
                    )
            else:
                cost_str = Text("—", style="cyan" if is_highlighted else "dim")

            calls_text = Text(
                str(r.model_calls) if r.model_calls else "—",
                style=(
                    "dim" if (is_zero and not is_highlighted) else ("cyan" if is_highlighted else "")
                ),
            )

            row_cells: list = [node_text]

            if total_records > 0:
                if r.eligible_records > 0:
                    cov_pct = r.eligible_records / total_records * 100
                    coverage_text = Text(
                        f"{r.eligible_records}/{total_records}  ({cov_pct:.0f}%)",
                        style=(
                            "dim"
                            if (is_zero and not is_highlighted)
                            else ("cyan" if is_highlighted else "")
                        ),
                    )
                else:
                    coverage_text = Text("—", style="cyan" if is_highlighted else "dim")
                row_cells.append(coverage_text)

            breakdown_cell = Text(r.source, style="cyan" if is_highlighted else "")
            row_cells.extend([calls_text, node_cost_str, cost_str, breakdown_cell])
            table.add_row(*row_cells, style=row_style)

        Console().print(table)


# ── Node registry ─────────────────────────────────────────────────────────────

# Maps node name → the class whose cost_estimate() to call.
_COST_NODES: dict[str, type] = {
    "claims": ClaimExtractorNode,
    "dedup": DedupNode,
    "geval_steps": GevalStepsNode,
    "grounding": GroundingNode,
    "relevance": RelevanceNode,
    "redteam": RedteamNode,
    "geval": GevalNode,
    "reference": ReferenceNode,
}

# Determines whether a node is active given the job config.
_NODE_ENABLED: dict[str, Callable[[EvalJobConfig], bool]] = {
    "claims": lambda c: c.enable_relevance or c.enable_grounding,
    "dedup": lambda c: c.enable_relevance or c.enable_grounding,
    "geval_steps": lambda c: c.enable_geval,
    "grounding": lambda c: c.enable_grounding,
    "relevance": lambda c: c.enable_relevance,
    "redteam": lambda c: c.enable_redteam,
    "geval": lambda c: c.enable_geval,
    "reference": lambda c: c.enable_reference,
}

_ZERO = NodeCostBreakdown(model_calls=0, cost_usd=0.0)


# ── CostEstimator ─────────────────────────────────────────────────────────────


class CostEstimator:
    """Estimates LLM cost for a full pipeline run given scanned input metadata.

    Args:
        job_config: Controls which nodes are active and which models to use.
    """

    def __init__(self, job_config: EvalJobConfig) -> None:
        self.job_config = job_config

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _eligible_records(self, node_name: str, metadata: InputMetadata) -> int:
        """Return the number of eligible records for a given node.

        Preferred source:
          - ``metadata.records[*].record_metadata.eligible_nodes`` from scanner output.

        Fallback:
          - synthetic metadata fixtures that omit ``records`` still use cost_meta
            aggregates for metric nodes and record_count for shared generation
            nodes (chunk/claims/dedup).
        """
        if metadata.records:
            return sum(
                1
                for record in metadata.records
                if node_name in record.record_metadata.eligible_nodes
            )

        mapping: dict[str, int] = {
            "chunk": int(metadata.cost_meta.claim.eligible_records),
            "claims": int(metadata.cost_meta.claim.eligible_records),
            "dedup": int(metadata.cost_meta.claim.eligible_records),
            "geval_steps": int(metadata.cost_meta.geval_steps.eligible_records),
            "grounding": int(metadata.cost_meta.grounding.eligible_records),
            "relevance": int(metadata.cost_meta.relevance.eligible_records),
            "redteam": int(metadata.cost_meta.readteam.eligible_records),
            "geval": int(metadata.cost_meta.geval.eligible_records),
            "reference": int(metadata.cost_meta.reference.eligible_records),
        }
        value = mapping.get(node_name)
        if value is not None:
            return value
        return int(metadata.record_count)

    def _resolve_cost_meta(self, node_name: str, metadata: InputMetadata):
        """Return the cost_meta object for a given node."""
        return {
            "claims": metadata.cost_meta.claim,
            "dedup": metadata.cost_meta.grounding,
            "geval_steps": metadata.cost_meta.geval_steps,
            "grounding": metadata.cost_meta.grounding,
            "relevance": metadata.cost_meta.relevance,
            "redteam": metadata.cost_meta.readteam,
            "geval": metadata.cost_meta.geval,
            "reference": metadata.cost_meta.reference,
        }[node_name]

    @staticmethod
    def _resolve_runtime_cost_kwargs(
        *,
        node_cls: type,
        cases: Optional[list[EvalCase]],
        model: str,
    ) -> dict[str, Any]:
        """Ask the node class for runtime-only cost kwargs when available."""
        resolver = getattr(node_cls, "resolve_cost_kwargs", None)
        if not callable(resolver):
            return {}
        resolved = resolver(cases=cases, model=model)
        return resolved if isinstance(resolved, dict) else {}

    @staticmethod
    def _call_cost_formula(
        formula_fn: Callable[..., Any],
        cost_meta: Any,
        runtime_kwargs: dict[str, Any],
    ) -> str:
        """Call node ``cost_formula`` with only supported runtime kwargs."""
        if not runtime_kwargs:
            return str(formula_fn(cost_meta))

        sig = inspect.signature(formula_fn)
        params = sig.parameters.values()
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if accepts_var_kwargs:
            return str(formula_fn(cost_meta, **runtime_kwargs))

        accepted_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in runtime_kwargs.items() if k in accepted_keys}
        if filtered_kwargs:
            return str(formula_fn(cost_meta, **filtered_kwargs))
        return str(formula_fn(cost_meta))

    def _estimate_node(
        self,
        node_name: str,
        metadata: InputMetadata,
        *,
        cases: Optional[list[EvalCase]] = None,
    ) -> NodeCostBreakdown:
        """Return the isolated cost for one node (not cumulative).

        Returns a zero breakdown for nodes with no LLM cost or that are disabled.
        """
        node_cls = _COST_NODES.get(node_name)
        if node_cls is None:
            return _ZERO

        enabled_fn = _NODE_ENABLED.get(node_name)
        if enabled_fn and not enabled_fn(self.job_config):
            return _ZERO

        model = get_judge_model(node_name, self.job_config.judge_model)
        cost_meta = self._resolve_cost_meta(node_name, metadata)
        runtime_kwargs = self._resolve_runtime_cost_kwargs(
            node_cls=node_cls,
            cases=cases,
            model=model,
        )

        if node_cls is ClaimExtractorNode:
            return cast(
                NodeCostBreakdown,
                node_cls(model=model).cost_estimate(cost_meta=cost_meta, **runtime_kwargs),
            )

        return cast(
            NodeCostBreakdown,
            node_cls(judge_model=model).cost_estimate(cost_meta=cost_meta, **runtime_kwargs),
        )

    def _node_formula(
        self,
        node_name: str,
        metadata: InputMetadata,
        *,
        cases: Optional[list[EvalCase]] = None,
    ) -> str:
        """Return the short cost formula string for a node, or '' if none."""
        node_cls = _COST_NODES.get(node_name)
        if node_cls is None or not hasattr(node_cls, "cost_formula"):
            return ""
        model = get_judge_model(node_name, self.job_config.judge_model)
        cost_meta = self._resolve_cost_meta(node_name, metadata)
        runtime_kwargs = self._resolve_runtime_cost_kwargs(
            node_cls=node_cls,
            cases=cases,
            model=model,
        )
        return self._call_cost_formula(node_cls.cost_formula, cost_meta, runtime_kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        metadata: InputMetadata,
        *,
        individual_overrides: Optional[dict[str, NodeCostBreakdown]] = None,
        eligible_overrides: Optional[dict[str, int]] = None,
        formula_overrides: Optional[dict[str, str]] = None,
        cases: Optional[list[EvalCase]] = None,
    ) -> CostReport:
        """Compute a cumulative cost report for all pipeline nodes.

        Each row's ``model_calls`` and ``cost_usd`` represent the *total* cost
        accrued by that node and all its cost-bearing prerequisites.  The
        ``source`` field lists the contributing node names so the breakdown
        is transparent.

        Args:
            metadata: Output of the Metadata Scanner (must include cost_meta).
            individual_overrides: Optional per-node isolated cost overrides.
                Used by cache-aware preflight to price only uncached work.
            eligible_overrides: Optional per-node eligible-record counts to
                display alongside overridden costs.
            formula_overrides: Optional per-node formula strings; keeps table
                explanations aligned with overridden subsets.
            cases: Optional case list used by nodes whose estimates depend on
                artifact-cache hit/miss state (for example, geval_steps).

        Returns:
            CostReport with one NodeCostRow per pipeline node in order.
        """
        # Step 1 — individual cost per node
        individual: dict[str, NodeCostBreakdown] = {}
        for node in NODE_ORDER:
            override = individual_overrides.get(node) if individual_overrides else None
            individual[node] = (
                override
                if override is not None
                else self._estimate_node(node, metadata, cases=cases)
            )

        # Step 2 — cumulative cost per node
        rows: list[NodeCostRow] = []
        for node in NODE_ORDER:
            prereqs_and_self = list(NODES_BY_NAME[node].prerequisites) + [node]
            contributing = [n for n in prereqs_and_self if individual[n].model_calls > 0]

            parts = []
            for n in contributing:
                formula = (
                    formula_overrides.get(n)
                    if formula_overrides and n in formula_overrides
                    else self._node_formula(n, metadata)
                )
                if n == node and formula:
                    indented = formula.replace("\n", "\n  ")
                    parts.append(f"{n}(\n  {indented}\n)")
                else:
                    parts.append(n)

            rows.append(
                NodeCostRow(
                    node_name=node,
                    model_calls=sum(individual[n].model_calls for n in contributing),
                    cost_usd=sum(individual[n].cost_usd for n in contributing),
                    source=" + ".join(parts) if parts else "—",
                    individual_cost_usd=individual[node].cost_usd,
                    eligible_records=(
                        eligible_overrides[node]
                        if eligible_overrides is not None and node in eligible_overrides
                        else self._eligible_records(node, metadata)
                    ),
                )
            )

        return CostReport(rows=rows)


# ── Module-level wrapper ──────────────────────────────────────────────────────


def estimate(
    metadata: InputMetadata,
    job_config: EvalJobConfig,
    target_node: Optional[str] = None,
    cases: Optional[list[EvalCase]] = None,
) -> CostEstimate:
    """Module-level wrapper for callers that need a typed ``CostEstimate``.

    Delegates to CostEstimator and converts the CostReport to a CostEstimate
    so the result can be stored in EvalState and used by the eval node.
    """
    report = CostEstimator(job_config).estimate(metadata, cases=cases)
    eval_row = report.row("eval")
    return CostEstimate(
        estimated_judge_calls=eval_row.model_calls,
        estimated_embedding_calls=0,
        estimated_tavily_calls=0,
        judge_cost_usd=eval_row.cost_usd,
        embedding_cost_usd=0.0,
        tavily_cost_usd=0.0,
        total_estimated_usd=eval_row.cost_usd,
        low_usd=eval_row.cost_usd,
        high_usd=eval_row.cost_usd,
        node_breakdown={
            r.node_name: NodeCostBreakdown(model_calls=r.model_calls, cost_usd=r.cost_usd)
            for r in report.rows
        },
    )


# ── Manual experiment ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick cost estimate experiment — runs the scanner on sample.json then
    prints the full CostReport for a representative EvalJobConfig.

    Run with:
        uv run python -m lumiseval_graph.nodes.cost_estimator
    """
    import sys
    from pathlib import Path

    from lumiseval_ingest.scanner import scan_file

    sample_path = Path(__file__).parents[4] / "sample.json"
    if not sample_path.exists():
        print(f"sample.json not found at {sample_path}", file=sys.stderr)
        sys.exit(1)

    metadata = scan_file(sample_path, show_progress=True)

    cfg = EvalJobConfig(
        job_id="experiment",
        enable_grounding=True,
        enable_relevance=True,
        enable_redteam=True,
        enable_geval=True,
    )

    report = CostEstimator(cfg).estimate(metadata)
    report.print_table()
