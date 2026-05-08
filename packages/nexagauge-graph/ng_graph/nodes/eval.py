"""Eval-node aggregation primitives for per-case and cross-case summaries.

This module intentionally supports two layers:

1. Per-case aggregation (``node_eval``)
   - Runs inside graph execution for one case.
   - Reads metric wrappers produced by metric nodes (grounding, relevance,
     redteam, GEval, reference).
   - Emits normalized ``eval_summary.metric_rows`` attached to that case's
     state. This is lightweight and stateless across cases.

2. Cross-case aggregation (``EvalBatchCollector``)
   - Used by the runner for multi-case executions (CLI batch, API batch).
   - Ingests each successful case's ``final_state`` and keeps running totals.
   - Produces a snapshot suitable for rendering (node-level and metric-level
     views).

Threading model:
- ``node_eval`` is pure and safe to run concurrently because it only reads
  input state and returns a new patch.
- ``EvalBatchCollector`` owns mutable shared counters and uses a ``Lock`` to
  make multi-threaded updates atomic.
- The lock protects in-memory consistency only; run isolation is achieved by
  creating one collector instance per run/session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from threading import Lock
from typing import Any, Mapping

from ng_core.types import CostEstimate
from rich.table import Table


@dataclass(frozen=True)
class EvalMetricSpec:
    """Declarative mapping for one metric source in ``node_eval`` output.

    Attributes:
        state_key: State key holding the wrapper model for this metric family
            (for example ``grounding_metrics``).
        aggregation: Label describing default rollup strategy metadata attached
            to each emitted row.
        weight: Default weight metadata attached to each emitted row.
    """

    state_key: str
    aggregation: str = "mean"
    weight: float = 1.0

# Single source of truth for eval-row extraction behavior.
# Future averaging/weighting behavior should be changed here only.
EVAL_METRIC_SPECS: dict[str, EvalMetricSpec] = {
    "grounding": EvalMetricSpec(state_key="grounding_metrics"),
    "geval": EvalMetricSpec(state_key="geval_metrics"),
    "relevance": EvalMetricSpec(state_key="relevance_metrics"),
    "reference": EvalMetricSpec(state_key="reference_metrics"),
    "redteam": EvalMetricSpec(state_key="redteam_metrics"),
}

@dataclass
class _AggregateStats:
    """Internal accumulator for one rollup bucket.

    A bucket may represent:
    - total across all rows,
    - one source node (for example ``grounding``),
    - one metric identity under a node (for example ``reference:rouge_l``).
    """

    metrics: int = 0
    scored: int = 0
    errors: int = 0
    passed: int = 0
    verdict_total: int = 0
    weighted_score_sum: float = 0.0
    weight_sum: float = 0.0
    score_values: list[float] = field(default_factory=list)

    def ingest(self, row: Mapping[str, Any]) -> None:
        """Update this bucket with one normalized metric row."""
        self.metrics += 1
        if row.get("error"):
            self.errors += 1
        normalized_verdict = _normalize_verdict(row.get("verdict"))
        if normalized_verdict is not None:
            self.verdict_total += 1
            if normalized_verdict == "PASSED":
                self.passed += 1

        score = _to_float(row.get("score"))
        if score is None:
            return
        weight = _to_float(row.get("weight"))
        safe_weight = weight if weight is not None and weight > 0 else 1.0
        self.scored += 1
        self.weighted_score_sum += score * safe_weight
        self.weight_sum += safe_weight
        self.score_values.append(score)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot including derived scores/pass counts."""
        avg_score = (self.weighted_score_sum / self.weight_sum) if self.weight_sum > 0 else None
        median_score = median(self.score_values) if self.score_values else None
        return {
            "metrics": self.metrics,
            "scored": self.scored,
            "errors": self.errors,
            "passed": self.passed,
            "verdict_total": self.verdict_total,
            "weighted_score_sum": self.weighted_score_sum,
            "weight_sum": self.weight_sum,
            "avg_score": avg_score,
            "median_score": median_score,
        }





def _to_float(value: Any) -> float | None:
    """Best-effort float conversion used for score/weight parsing."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_verdict(value: Any) -> str | None:
    """Normalize row verdict values to PASSED/FAILED when possible."""
    if value is None:
        return None
    verdict = str(value).strip().upper()
    if verdict not in {"PASSED", "FAILED"}:
        return None
    return verdict


def _unwrap_metrics(wrapper: Any) -> list[Any]:
    """Normalize a metric wrapper to a plain metrics list.

    Accepts:
    - ``None`` -> ``[]``
    - already-a-list -> as is
    - pydantic wrapper with ``.metrics`` -> that list
    """
    if wrapper is None:
        return []
    if isinstance(wrapper, list):
        return wrapper
    return list(getattr(wrapper, "metrics", None) or [])


def _cost_from_wrapper(wrapper: Any) -> CostEstimate | None:
    """Extract ``CostEstimate`` from a wrapper, tolerant to dict/model shapes."""
    raw = getattr(wrapper, "cost", None)
    if raw is None:
        return None
    if isinstance(raw, CostEstimate):
        return raw
    try:
        return CostEstimate.model_validate(raw)
    except Exception:
        return None


def _sum_costs(costs: list[CostEstimate]) -> CostEstimate:
    """Sum per-node costs for one case's eval fan-in view."""
    total_cost = 0.0
    total_input = 0.0
    total_output = 0.0
    saw_input = False
    saw_output = False

    for estimate in costs:
        total_cost += float(estimate.cost or 0.0)
        if estimate.input_tokens is not None:
            total_input += float(estimate.input_tokens)
            saw_input = True
        if estimate.output_tokens is not None:
            total_output += float(estimate.output_tokens)
            saw_output = True

    return CostEstimate(
        cost=total_cost,
        input_tokens=total_input if saw_input else None,
        output_tokens=total_output if saw_output else None,
    )


def iter_eval_metric_rows(final_state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract normalized rows from ``final_state.eval_summary.metric_rows``.

    This helper is intentionally tolerant so callers can safely pass any
    ``final_state`` payload and receive ``[]`` when the structure is missing.
    """
    eval_summary = final_state.get("eval_summary")
    if not isinstance(eval_summary, Mapping):
        return []
    metric_rows = eval_summary.get("metric_rows")
    if not isinstance(metric_rows, list):
        return []
    return [row for row in metric_rows if isinstance(row, Mapping)]


class EvalBatchCollector:
    """Thread-safe cross-case accumulator for eval summaries.

    Typical usage:
    - Batch run: runner ingests each successful case's ``final_state``.
    - Single-case run: collector is optional; callers can read that case's
      ``eval_summary`` directly without this class.

    Concurrency:
    - ``ingest_*`` and ``snapshot`` acquire ``self._lock``.
    - This ensures internal counters/maps stay consistent when many worker
      threads report results concurrently.
    - The lock does not provide run/session boundaries; create one collector
      instance per run to avoid mixing independent workloads.
    """

    def __init__(self) -> None:
        """Initialize empty counters for totals, per-node, and per-metric views."""
        self._lock = Lock()
        self._cases_with_eval = 0
        self._totals = _AggregateStats()
        self._by_node: dict[str, _AggregateStats] = {}
        self._by_metric: dict[tuple[str, str], _AggregateStats] = {}

    def reset(self) -> None:
        """Clear all accumulated state in place (thread-safe)."""
        with self._lock:
            self._cases_with_eval = 0
            self._totals = _AggregateStats()
            self._by_node.clear()
            self._by_metric.clear()

    def ingest_rows(self, metric_rows: list[Mapping[str, Any]]) -> bool:
        """Ingest one case worth of already-normalized metric rows.

        Returns:
            ``True`` when rows were ingested, ``False`` when input was empty.
        """
        if not metric_rows:
            return False

        with self._lock:
            self._cases_with_eval += 1
            for row in metric_rows:
                node = str(row.get("source_node") or "unknown")
                metric_name = str(row.get("metric_name") or "unknown")

                self._totals.ingest(row)
                node_stats = self._by_node.setdefault(node, _AggregateStats())
                node_stats.ingest(row)
                metric_stats = self._by_metric.setdefault((node, metric_name), _AggregateStats())
                metric_stats.ingest(row)
        return True

    def ingest_final_state(self, final_state: Mapping[str, Any]) -> bool:
        """Extract rows from one case ``final_state`` and ingest them."""
        return self.ingest_rows(iter_eval_metric_rows(final_state))

    def snapshot(self) -> dict[str, Any]:
        """Return immutable, JSON-friendly aggregate output for rendering/API."""
        with self._lock:
            by_node = {node: stats.to_dict() for node, stats in sorted(self._by_node.items())}
            by_metric: dict[str, dict[str, dict[str, Any]]] = {}
            for (node, metric_name), stats in sorted(self._by_metric.items()):
                by_metric.setdefault(node, {})[metric_name] = stats.to_dict()

            return {
                "schema_version": 1,
                "cases_with_eval": self._cases_with_eval,
                "total": self._totals.to_dict(),
                "by_node": by_node,
                "by_metric": by_metric,
            }


def _format_avg(value: Any) -> str:
    """Render avg values consistently for table output."""
    avg = _to_float(value)
    return f"{avg:.4f}" if avg is not None else "—"


def _format_passed(stats: Mapping[str, Any]) -> str:
    """Render pass counts as passed/total, or em-dash when unavailable."""
    passed = int(stats.get("passed") or 0)
    verdict_total = int(stats.get("verdict_total") or 0)
    if verdict_total <= 0:
        return "—"
    return f"{passed}/{verdict_total}"


def build_eval_summary_tables(summary: Mapping[str, Any]) -> list[Table]:
    """Build Rich tables from a collector snapshot.

    Produces:
    - table 1: per-node rollup
    - table 2: per-metric rollup (when present)

    Returns ``[]`` when no node-level data exists.
    """
    by_node = summary.get("by_node")
    if not isinstance(by_node, Mapping) or not by_node:
        return []

    cases_with_eval = int(summary.get("cases_with_eval") or 0)
    ordered_nodes = [n for n in ("grounding", "geval", "relevance", "reference", "redteam")]
    ordered_nodes.extend(sorted(n for n in by_node if n not in ordered_nodes))

    node_table = Table(
        title=f"eval metrics summary across {cases_with_eval} case(s)",
        show_header=True,
        header_style="bold cyan",
    )
    node_table.add_column("node", style="bold")
    node_table.add_column("metrics", justify="right")
    node_table.add_column("scored", justify="right")
    node_table.add_column("errors", justify="right")
    node_table.add_column("avg_score", justify="right")
    node_table.add_column("median_score", justify="right")
    node_table.add_column("passed", justify="right")

    for node in ordered_nodes:
        stats = by_node.get(node)
        if not isinstance(stats, Mapping):
            continue
        node_table.add_row(
            node,
            str(int(stats.get("metrics") or 0)),
            str(int(stats.get("scored") or 0)),
            str(int(stats.get("errors") or 0)),
            _format_avg(stats.get("avg_score")),
            _format_avg(stats.get("median_score")),
            _format_passed(stats),
        )

    tables: list[Table] = [node_table]

    by_metric = summary.get("by_metric")
    if not isinstance(by_metric, Mapping) or not by_metric:
        return tables

    metric_table = Table(
        title=f"eval metric breakdown across {cases_with_eval} case(s)",
        show_header=True,
        header_style="bold cyan",
    )
    metric_table.add_column("node", style="bold")
    metric_table.add_column("metric", style="cyan")
    metric_table.add_column("metrics", justify="right")
    metric_table.add_column("scored", justify="right")
    metric_table.add_column("errors", justify="right")
    metric_table.add_column("avg_score", justify="right")
    metric_table.add_column("median_score", justify="right")
    metric_table.add_column("passed", justify="right")

    for node in ordered_nodes:
        node_metrics = by_metric.get(node)
        if not isinstance(node_metrics, Mapping):
            continue
        for metric_name in sorted(node_metrics):
            stats = node_metrics.get(metric_name)
            if not isinstance(stats, Mapping):
                continue
            metric_table.add_row(
                node,
                metric_name,
                str(int(stats.get("metrics") or 0)),
                str(int(stats.get("scored") or 0)),
                str(int(stats.get("errors") or 0)),
                _format_avg(stats.get("avg_score")),
                _format_avg(stats.get("median_score")),
                _format_passed(stats),
            )
    tables.append(metric_table)
    return tables


def node_eval(state: Mapping[str, Any]) -> dict[str, Any]:
    """Collect per-case metric rows from all metric nodes.

    Execution semantics:
    - Called once per case by the graph runtime.
    - Does not mutate shared/global state.
    - Returns an ``eval_summary`` patch for that case only.

    Returns:
      {"eval_summary": {"metric_rows": [...], "cost": {...}, "schema_version": 1}}
      or {} if no metric rows exist for this case.
    """
    metric_rows: list[dict[str, Any]] = []
    costs: list[CostEstimate] = []

    for source_node, spec in EVAL_METRIC_SPECS.items():
        wrapper = state.get(spec.state_key)
        if wrapper is None:
            continue

        wrapper_cost = _cost_from_wrapper(wrapper)
        if wrapper_cost is not None:
            costs.append(wrapper_cost)

        for metric in _unwrap_metrics(wrapper):
            metric_rows.append(
                {
                    "source_node": source_node,
                    "metric_name": str(getattr(metric, "name", "")),
                    "score": _to_float(getattr(metric, "score", None)),
                    "verdict": getattr(metric, "verdict", None),
                    "error": getattr(metric, "error", None),
                    "aggregation": spec.aggregation,
                    "weight": spec.weight,
                }
            )

    if not metric_rows:
        return {}

    return {
        "eval_summary": {
            "metric_rows": metric_rows,
            "cost": _sum_costs(costs).model_dump(),
            "schema_version": 1,
        }
    }
