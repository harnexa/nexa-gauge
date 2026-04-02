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

from dataclasses import dataclass
from typing import Callable, Optional

from lumiseval_core.constants import GENERATION_CHUNK_SIZE_TOKENS
from lumiseval_core.pipeline import NODE_ORDER, NODE_PREREQUISITES
from lumiseval_core.types import (
    ClaimCostMeta,
    CostEstimate,
    EvalJobConfig,
    InputMetadata,
    NodeCostBreakdown,
)

from lumiseval_graph.llm.config import get_judge_model
from lumiseval_graph.nodes.claim_extractor import ClaimExtractorNode
from lumiseval_graph.nodes.metrics.reference import ReferenceMetricsNode
from lumiseval_graph.nodes.metrics.grounding import GroundingNode
from lumiseval_graph.nodes.metrics.redteam import RedteamNode
from lumiseval_graph.nodes.metrics.relevance import RelevanceNode
from lumiseval_graph.nodes.metrics.rubric import RubricNode

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


_METRIC_NODES = {"claims", "grounding", "relevance", "redteam", "rubric"}


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
    ) -> None:
        """Render the report as a pretty Rich table in the terminal."""
        from rich import box
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

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
            is_zero = r.model_calls == 0
            is_eval = r.node_name == "eval"
            is_metric = r.node_name in _METRIC_NODES

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

            calls_text = Text(
                str(r.model_calls) if r.model_calls else "—", style="dim" if is_zero else ""
            )

            row_cells: list = [node_text]

            if total_records > 0:
                if r.eligible_records > 0:
                    cov_pct = r.eligible_records / total_records * 100
                    coverage_text = Text(
                        f"{r.eligible_records}/{total_records}  ({cov_pct:.0f}%)",
                        style="dim" if is_zero else "",
                    )
                else:
                    coverage_text = Text("—", style="dim")
                row_cells.append(coverage_text)

            row_cells.extend([calls_text, node_cost_str, cost_str, r.source])
            table.add_row(*row_cells, style=row_style)

        Console().print(table)


# ── Node registry ─────────────────────────────────────────────────────────────

# Maps node name → the class whose cost_estimate() to call.
_COST_NODES: dict[str, type] = {
    "claims": ClaimExtractorNode,
    "grounding": GroundingNode,
    "relevance": RelevanceNode,
    "redteam": RedteamNode,
    "rubric": RubricNode,
    "reference": ReferenceMetricsNode,
}

# Determines whether a node is active given the job config.
_NODE_ENABLED: dict[str, Callable[[EvalJobConfig], bool]] = {
    "claims": lambda c: c.enable_relevance or c.enable_grounding,
    "grounding": lambda c: c.enable_grounding,
    "relevance": lambda c: c.enable_relevance,
    "redteam": lambda c: c.enable_redteam,
    "rubric": lambda c: c.enable_rubric,
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

    def _build_claim_cost_meta(self, metadata: InputMetadata) -> ClaimCostMeta:
        """Derive ClaimCostMeta from InputMetadata.

        Claims run on context-eligible records (same set as grounding/relevance).
        Chunk and token averages are computed from aggregate scanner output.
        """
        eligible_records = metadata.cost_meta.grounding.eligible_records
        avg_generation_chunks = metadata.generation_chunk_count / max(1, eligible_records)
        avg_generation_tokens = (
            metadata.generation_tokens // max(1, metadata.generation_chunk_count)
            if metadata.generation_chunk_count
            else GENERATION_CHUNK_SIZE_TOKENS
        )
        return ClaimCostMeta(
            eligible_records=eligible_records,
            avg_generation_chunks=avg_generation_chunks,
            avg_generation_tokens=avg_generation_tokens,
        )

    def _eligible_records(self, node_name: str, metadata: InputMetadata) -> int:
        """Return the number of eligible records for a given node."""
        mapping = {
            "chunk":  metadata.cost_meta.grounding.eligible_records,
            "claims": metadata.cost_meta.grounding.eligible_records,
            "dedupe": metadata.cost_meta.grounding.eligible_records,
            "grounding": metadata.cost_meta.grounding.eligible_records,
            "relevance": metadata.cost_meta.relevance.eligible_records,
            "redteam": metadata.cost_meta.readteam.eligible_records,
            "rubric": metadata.cost_meta.rubric.eligible_records,
            "reference": metadata.cost_meta.reference.eligible_records,
        }
        return mapping.get(node_name, metadata.record_count)

    def _resolve_cost_meta(self, node_name: str, metadata: InputMetadata):
        """Return the cost_meta object for a given node."""
        if node_name == "claims":
            return self._build_claim_cost_meta(metadata)
        return {
            "grounding": metadata.cost_meta.grounding,
            "relevance": metadata.cost_meta.relevance,
            "redteam": metadata.cost_meta.readteam,
            "rubric": metadata.cost_meta.rubric,
            "reference": metadata.cost_meta.reference,
        }[node_name]

    def _estimate_node(self, node_name: str, metadata: InputMetadata) -> NodeCostBreakdown:
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

        if node_cls is ClaimExtractorNode:
            return node_cls(model=model).cost_estimate(cost_meta=cost_meta)

        return node_cls(judge_model=model).cost_estimate(cost_meta=cost_meta)

    def _node_formula(self, node_name: str, metadata: InputMetadata) -> str:
        """Return the short cost formula string for a node, or '' if none."""
        node_cls = _COST_NODES.get(node_name)
        if node_cls is None or not hasattr(node_cls, "cost_formula"):
            return ""
        cost_meta = self._resolve_cost_meta(node_name, metadata)
        return node_cls.cost_formula(cost_meta)

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(self, metadata: InputMetadata) -> CostReport:
        """Compute a cumulative cost report for all pipeline nodes.

        Each row's ``model_calls`` and ``cost_usd`` represent the *total* cost
        accrued by that node and all its cost-bearing prerequisites.  The
        ``source`` field lists the contributing node names so the breakdown
        is transparent.

        Args:
            metadata: Output of the Metadata Scanner (must include cost_meta).

        Returns:
            CostReport with one NodeCostRow per pipeline node in order.
        """
        # Step 1 — individual cost per node
        individual: dict[str, NodeCostBreakdown] = {
            node: self._estimate_node(node, metadata) for node in NODE_ORDER
        }

        # Step 2 — cumulative cost per node
        rows: list[NodeCostRow] = []
        for node in NODE_ORDER:
            prereqs_and_self = list(NODE_PREREQUISITES[node]) + [node]
            contributing = [n for n in prereqs_and_self if individual[n].model_calls > 0]

            parts = []
            for n in contributing:
                formula = self._node_formula(n, metadata)
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
                    eligible_records=self._eligible_records(node, metadata),
                )
            )

        return CostReport(rows=rows)


# ── Module-level wrapper ──────────────────────────────────────────────────────


def estimate(
    metadata: InputMetadata,
    job_config: EvalJobConfig,
    target_node: Optional[str] = None,
    rubric_rule_count: int = 0,
) -> CostEstimate:
    """Module-level wrapper called by graph.node_cost_estimator.

    Delegates to CostEstimator and converts the CostReport to a CostEstimate
    so the result can be stored in EvalState and used by the eval node.
    """
    report = CostEstimator(job_config).estimate(metadata)
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
        enable_rubric=True,
    )

    report = CostEstimator(cfg).estimate(metadata)
    report.print_table()

"""
    claims + relevance[3 recs, 1 call/rec (all claims batched), 3 calls                   │
        input_tokens  = 108 (prompt) + 13 (question) + 35 (2.3 claims × 15t) = 156 tok/call │
        output_tokens = 23 (2.3 claims × 10t json_verdict) = 23 tok/call
    ] 

    This doesn't look nice to be displayed in the terminal. Can I have something more readable

    claims + relevance(
        number_rcords = 3 recs, 1 call/rec (all claims batched), 3 calls
        input_tokens  = 108 (prompt tokens) + 13 (question tokens) + 35 (2.3 claims/record × 15 token/claim) = 156 tok/call
        output_tokens = 23 (2.3 claims × 10 token_json_verdict) = 23 tok/call
        total_tokens = 3 * (156 + 23)= 179 tok/call
    )
│            │        │                  │   input_tokens  = 108 (prompt) + 13 (question) + 35 (2.3 claims × 15t) = 156 tok/call │
│            │        │                  │   output_tokens = 23 (2.3 claims × 10t json_verdict) = 23 tok/call] 

"""
