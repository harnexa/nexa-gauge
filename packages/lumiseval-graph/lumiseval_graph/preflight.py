"""Shared preflight estimation service for dataset-level branch planning.

This module centralizes the reusable preflight flow:

1. Scan selected cases (token + eligibility metadata).
2. Build cache-aware execution plan for a target branch.
3. Compute uncached-only (delta) cost for that branch.

Both CLI and future API `/estimate` handlers should call this module so they
produce identical planning and cost semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from lumiseval_core.pipeline import NODE_ORDER, NODES_BY_NAME
from lumiseval_core.types import EvalCase, EvalJobConfig, InputMetadata, NodeCostBreakdown
from lumiseval_ingest.scanner import scan_cases

from lumiseval_graph.node_runner import CachedNodeRunner, DatasetNodePlan
from lumiseval_graph.nodes.cost_estimator import CostEstimator, CostReport


@dataclass(frozen=True)
class PreflightEstimateResult:
    """Result payload returned by :func:`estimate_preflight`.

    Attributes:
        target_node: Canonical node selected by the caller.
        metadata: Full dataset scan output for the selected cases.
        plan: Cache-aware execution plan (to_run/cached/skipped maps).
        cost_report: Delta cost report built from uncached work only.
    """

    target_node: str
    metadata: InputMetadata
    plan: DatasetNodePlan
    cost_report: CostReport


def _estimate_delta_cost_report(
    *,
    metadata: InputMetadata,
    cases: list[EvalCase],
    plan: DatasetNodePlan,
    job_config: EvalJobConfig,
) -> CostReport:
    """Estimate only uncached cost for the selected branch.

    For each node row, the method isolates case indices that still need fresh
    execution, rescans just that subset to derive accurate per-node cost meta,
    and then asks CostEstimator for a cumulative report from those overrides.
    """

    estimator = CostEstimator(job_config)
    individual: dict[str, NodeCostBreakdown] = {
        node: NodeCostBreakdown(model_calls=0, cost_usd=0.0) for node in NODE_ORDER
    }
    eligible_overrides: dict[str, int] = {node: 0 for node in NODE_ORDER}
    formula_overrides: dict[str, str] = {node: "" for node in NODE_ORDER}

    for node in NODE_ORDER:
        case_indices = plan.to_run_case_indices_by_node.get(node, [])
        eligible_overrides[node] = len(case_indices)
        if not case_indices:
            continue
        subset_cases = [cases[i] for i in case_indices]
        subset_meta = scan_cases(subset_cases, show_progress=False)
        individual[node] = estimator._estimate_node(node, subset_meta, cases=subset_cases)
        formula_overrides[node] = estimator._node_formula(node, subset_meta, cases=subset_cases)

    return estimator.estimate(
        metadata,
        individual_overrides=individual,
        eligible_overrides=eligible_overrides,
        formula_overrides=formula_overrides,
    )


def estimate_preflight(
    *,
    cases: list[EvalCase],
    target_node: str,
    runner: CachedNodeRunner,
    job_config: EvalJobConfig,
    force: bool = False,
    show_progress: bool = True,
) -> PreflightEstimateResult:
    """Run shared preflight for one target branch over a selected dataset.

    Args:
        cases: Selected dataset records.
        target_node: Canonical node name.
        runner: Cache-aware runner used to compute the execution plan.
        job_config: Job configuration used for both planning and costing.
        force: When True, ignore cache reads in planning.
        show_progress: Pass-through flag for scanner progress rendering.
    """

    if target_node not in NODES_BY_NAME:
        valid = ", ".join(NODE_ORDER)
        raise ValueError(f"Unknown node '{target_node}'. Valid options: {valid}.")

    metadata = scan_cases(cases, show_progress=show_progress)
    plan = runner.plan_dataset(
        cases=cases,
        node_name=target_node,
        job_config=job_config,
        force=force,
    )
    cost_report = _estimate_delta_cost_report(
        metadata=metadata,
        cases=cases,
        plan=plan,
        job_config=job_config,
    )
    return PreflightEstimateResult(
        target_node=target_node,
        metadata=metadata,
        plan=plan,
        cost_report=cost_report,
    )



"""


scan: scans all the cases


"""