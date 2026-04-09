"""Utility for running individual graph nodes with optional caching.

Two runners are provided:

  NodeRunner        — simple, no-cache runner (runs prerequisites + target node
                      sequentially from a fresh state).

  CachedNodeRunner  — cache-aware runner that checks the cache before each step;
                      only uncached nodes are executed.  Enables three scenarios:
                        1. Partial execution — stop after any named node.
                        2. Incremental extension — extend a cached run to a
                           deeper node without re-running upstream steps.
                        3. Incremental datasets — new cases run fully; previously-
                           seen cases are served entirely from cache.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from lumiseval_core.cache import CacheEntry, CacheStore, compute_case_hash, compute_config_hash
from lumiseval_core.pipeline import (
    METRIC_NODES,
    NODE_ORDER,
    NODES_BY_NAME,
)
from lumiseval_core.types import EvalCase, EvalJobConfig, NodeCostBreakdown
from pydantic import BaseModel

from lumiseval_graph import graph as graph_module
from lumiseval_graph.nodes.cost_estimator import CostEstimator
from lumiseval_graph.nodes.cost_estimator import estimate as build_cost_estimate
from lumiseval_graph.registry import NODE_FNS


class NodeRunResult(BaseModel):
    """Result payload for non-cached single-case execution."""

    node_name: str
    executed_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]


class CachedNodeRunResult(BaseModel):
    """Result payload for cache-aware single-case execution."""

    node_name: str
    case_id: str
    executed_nodes: list[str]
    cached_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]
    elapsed_ms: float


class CaseNodePlan(BaseModel):
    """Per-case execution planning result for a specific target node.

    This structure is computed without executing node functions and captures:
      - the strict prerequisite path
      - which nodes will run vs be loaded from cache vs be skipped by eligibility
      - stable hashes used for cache lookup
    """

    target_node: str
    case_index: int
    case_id: str
    case_hash: str
    config_hash: str
    planned_nodes: list[str]
    to_run_nodes: list[str]
    cached_nodes: list[str]
    skipped_nodes: list[str]
    node_status: dict[str, str]


class DatasetNodePlan(BaseModel):
    """Aggregated execution plan across a dataset for one target node.

    Node-level maps are keyed by canonical node name and list either case ids
    or case indices so callers can produce UX summaries and delta estimates
    without re-running planning logic.
    """

    target_node: str
    planned_nodes: list[str]
    case_plans: list[CaseNodePlan]
    to_run_case_ids_by_node: dict[str, list[str]]
    cached_case_ids_by_node: dict[str, list[str]]
    skipped_case_ids_by_node: dict[str, list[str]]
    to_run_case_indices_by_node: dict[str, list[int]]
    cached_case_indices_by_node: dict[str, list[int]]
    skipped_case_indices_by_node: dict[str, list[int]]

    def to_run_count(self, node_name: str) -> int:
        """Return how many cases require fresh compute for `node_name`."""
        return len(self.to_run_case_ids_by_node.get(node_name, []))

    def cached_count(self, node_name: str) -> int:
        """Return how many cases can reuse cached output for `node_name`."""
        return len(self.cached_case_ids_by_node.get(node_name, []))

    def skipped_count(self, node_name: str) -> int:
        """Return how many cases are ineligible and therefore skipped."""
        return len(self.skipped_case_ids_by_node.get(node_name, []))


class NodeRunner:
    """Execute a single node for a case, optionally running required prior nodes."""

    @staticmethod
    def _has_generation(case: EvalCase) -> bool:
        return bool(case.generation and case.generation.strip())

    @staticmethod
    def _has_question(case: EvalCase) -> bool:
        return bool(case.question and case.question.strip())

    @staticmethod
    def _has_context(case: EvalCase) -> bool:
        return any(
            (isinstance(item, str) and item.strip()) or (item is not None and str(item).strip())
            for item in (case.context or [])
        )

    @staticmethod
    def _has_geval(case: EvalCase) -> bool:
        return case.geval is not None and len(case.geval.metrics) > 0

    @staticmethod
    def _has_reference(case: EvalCase) -> bool:
        """True when a non-empty reference answer exists on the case."""
        return bool(case.reference and case.reference.strip())

    @classmethod
    def is_case_eligible_for_node(cls, case: EvalCase, node_name: str) -> bool:
        """Apply field-based eligibility gates used by both planning and execution.

        Why centralize this:
          - planners and runners must agree or preflight cost/execution diverges
          - all route gating (question/context/geval/reference) stays in one location
        """
        if not cls._has_generation(case):
            return False
        spec = NODES_BY_NAME.get(node_name)
        if spec is None:
            return True

        if spec.requires_question and not cls._has_question(case):
            return False
        if spec.requires_context and not cls._has_context(case):
            return False
        if spec.requires_geval and not cls._has_geval(case):
            return False
        if spec.requires_reference and not cls._has_reference(case):
            return False
        return True

    @classmethod
    def skipped_output_for_node(cls, node_name: str) -> dict[str, Any]:
        """Return the canonical no-op state patch for an ineligible node."""
        spec = NODES_BY_NAME.get(node_name)
        return dict(spec.skip_output) if spec else {}

    def run_case(
        self,
        *,
        case: EvalCase,
        node_name: str,
        job_config: EvalJobConfig | None = None,
        include_prerequisites: bool = True,
    ) -> NodeRunResult:
        if node_name not in NODE_FNS:
            valid = ", ".join(sorted(NODE_FNS))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        state: dict[str, Any] = dict(
            graph_module.build_initial_state(
            generation=case.generation,
            job_config=job_config,
            question=case.question,
            reference=case.reference,
            context=case.context,
            target_node=node_name,
            geval=case.geval,
            reference_files=case.reference_files,
            )
        )

        plan = list(NODES_BY_NAME[node_name].prerequisites) if include_prerequisites else []
        plan.append(node_name)

        node_output: dict[str, Any] = {}
        executed_nodes: list[str] = []
        for step in plan:
            if not self.is_case_eligible_for_node(case, step):
                updates = self.skipped_output_for_node(step)
            else:
                updates = NODE_FNS[step](state)
                executed_nodes.append(step)
            state.update(updates)
            if step == node_name:
                node_output = updates

        return NodeRunResult(
            node_name=node_name,
            executed_nodes=executed_nodes,
            node_output=node_output,
            final_state=dict(state),
        )


class CachedNodeRunner:
    """Execute a pipeline node for a case, loading any upstream results from cache."""

    def __init__(self, cache_store: CacheStore) -> None:
        """Create a cache-aware runner with a filesystem or no-op cache backend."""
        self._cache = cache_store

    @staticmethod
    def _compute_hashes(case: EvalCase, job_config: EvalJobConfig) -> tuple[str, str]:
        """Compute stable content/config hashes used as cache keys."""
        case_hash = compute_case_hash(
            generation=case.generation,
            question=case.question,
            reference=case.reference,
            geval=case.geval,
            context=case.context or [],
            reference_files=case.reference_files or [],
        )
        config_hash = compute_config_hash(job_config)
        return case_hash, config_hash

    @staticmethod
    def _plan_nodes(node_name: str) -> list[str]:
        """Return strict prerequisite chain plus target node.

        Metric targets and ``eval`` always finish with ``report`` so users
        receive an aggregated report for both partial metric branches and
        full-eval runs.
        """
        plan = list(NODES_BY_NAME[node_name].prerequisites) + [node_name]

        needs_report = node_name == "eval" or node_name in METRIC_NODES
        if needs_report and node_name != "report" and "report" not in plan and "report" in NODE_FNS:
            plan.append("report")

        return plan

    def _get_cached_entry(
        self,
        *,
        case_hash: str,
        config_hash: str,
        step: str,
    ) -> CacheEntry | None:
        """Read canonical cache entry for one node step."""
        return self._cache.get_entry(case_hash, config_hash, step)

    @staticmethod
    def _estimate_step_cost(
        estimator: CostEstimator,
        step: str,
        state: dict[str, Any],
    ) -> NodeCostBreakdown:
        """Compute isolated node cost from current state metadata.

        Metadata is only available after scan runs; pre-scan nodes therefore
        receive a zero-cost placeholder.
        """
        metadata = state.get("metadata")
        if metadata is None:
            return NodeCostBreakdown(model_calls=0, cost_usd=0.0)
        return estimator._estimate_node(step, metadata)

    def plan_case(
        self,
        *,
        case: EvalCase,
        case_index: int = -1,
        node_name: str,
        job_config: EvalJobConfig | None = None,
        force: bool = False,
    ) -> CaseNodePlan:
        """Plan one case without executing any node function.

        The method checks node eligibility and cache presence for each step in
        the strict path to `node_name`. The output is later reused for:
          - preflight UX tables
          - delta cost estimation
          - debugging what work is expected to run
        """
        if node_name not in NODE_FNS:
            valid = ", ".join(sorted(NODE_FNS))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        state: dict[str, Any] = dict(
            graph_module.build_initial_state(
            generation=case.generation,
            job_config=job_config,
            question=case.question,
            reference=case.reference,
            context=case.context,
            target_node=node_name,
            geval=case.geval,
            reference_files=case.reference_files,
            )
        )
        case_hash, config_hash = self._compute_hashes(case, state["job_config"])
        planned_nodes = self._plan_nodes(node_name)

        to_run_nodes: list[str] = []
        cached_nodes: list[str] = []
        skipped_nodes: list[str] = []
        node_status: dict[str, str] = {}

        for step in planned_nodes:
            if not NodeRunner.is_case_eligible_for_node(case, step):
                skipped_nodes.append(step)
                node_status[step] = "skipped"
                continue
            if not force and self._get_cached_entry(
                case_hash=case_hash,
                config_hash=config_hash,
                step=step,
            ) is not None:
                cached_nodes.append(step)
                node_status[step] = "cached"
                continue
            to_run_nodes.append(step)
            node_status[step] = "to_run"

        return CaseNodePlan(
            target_node=node_name,
            case_index=case_index,
            case_id=case.case_id,
            case_hash=case_hash,
            config_hash=config_hash,
            planned_nodes=planned_nodes,
            to_run_nodes=to_run_nodes,
            cached_nodes=cached_nodes,
            skipped_nodes=skipped_nodes,
            node_status=node_status,
        )

    def plan_dataset(
        self,
        *,
        cases: list[EvalCase],
        node_name: str,
        job_config: EvalJobConfig | None = None,
        force: bool = False,
    ) -> DatasetNodePlan:
        """Plan an entire dataset run for a target node.

        Aggregates `plan_case` outputs into per-node maps so callers can quickly
        answer:
          - how many cases will run per node
          - how many are cache hits
          - how many are skipped by eligibility
        """
        if node_name not in NODE_FNS:
            valid = ", ".join(sorted(NODE_FNS))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        planned_nodes = self._plan_nodes(node_name)
        to_run_case_ids_by_node: dict[str, list[str]] = {n: [] for n in NODE_ORDER}
        cached_case_ids_by_node: dict[str, list[str]] = {n: [] for n in NODE_ORDER}
        skipped_case_ids_by_node: dict[str, list[str]] = {n: [] for n in NODE_ORDER}
        to_run_case_indices_by_node: dict[str, list[int]] = {n: [] for n in NODE_ORDER}
        cached_case_indices_by_node: dict[str, list[int]] = {n: [] for n in NODE_ORDER}
        skipped_case_indices_by_node: dict[str, list[int]] = {n: [] for n in NODE_ORDER}

        case_plans: list[CaseNodePlan] = []
        for i, case in enumerate(cases):
            plan = self.plan_case(
                case=case,
                case_index=i,
                node_name=node_name,
                job_config=job_config,
                force=force,
            )
            case_plans.append(plan)
            for step in plan.to_run_nodes:
                to_run_case_ids_by_node[step].append(plan.case_id)
                to_run_case_indices_by_node[step].append(plan.case_index)
            for step in plan.cached_nodes:
                cached_case_ids_by_node[step].append(plan.case_id)
                cached_case_indices_by_node[step].append(plan.case_index)
            for step in plan.skipped_nodes:
                skipped_case_ids_by_node[step].append(plan.case_id)
                skipped_case_indices_by_node[step].append(plan.case_index)

        return DatasetNodePlan(
            target_node=node_name,
            planned_nodes=planned_nodes,
            case_plans=case_plans,
            to_run_case_ids_by_node=to_run_case_ids_by_node,
            cached_case_ids_by_node=cached_case_ids_by_node,
            skipped_case_ids_by_node=skipped_case_ids_by_node,
            to_run_case_indices_by_node=to_run_case_indices_by_node,
            cached_case_indices_by_node=cached_case_indices_by_node,
            skipped_case_indices_by_node=skipped_case_indices_by_node,
        )

    def run_case(
        self,
        *,
        case: EvalCase,
        node_name: str,
        job_config: EvalJobConfig | None = None,
        force: bool = False,
    ) -> CachedNodeRunResult:
        """Execute one case to `node_name` with cache reuse and per-node cache writes.

        Behavior:
          - Reads cached outputs when available (unless `force=True`).
          - Writes outputs for executed *and skipped* nodes so future runs can
            short-circuit consistently.
          - Persists `node_cost` metadata alongside each cached node output.
          - For `eval` target, executes metric nodes in parallel when possible.
        """
        if node_name not in NODE_FNS:
            valid = ", ".join(sorted(NODE_FNS))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        t0 = time.monotonic()

        state: dict[str, Any] = dict(
            graph_module.build_initial_state(
            generation=case.generation,
            job_config=job_config,
            question=case.question,
            reference=case.reference,
            context=case.context,
            target_node=node_name,
            geval=case.geval,
            reference_files=case.reference_files,
            )
        )

        case_hash, config_hash = self._compute_hashes(case, state["job_config"])
        plan = self._plan_nodes(node_name)
        executed: list[str] = []
        cached: list[str] = []
        node_output: dict[str, Any] = {}
        metric_group = METRIC_NODES
        estimator = CostEstimator(state["job_config"])

        # A plan may look like below
        # plan: ['scan', 'chunk', 'claims', 'dedup', 'relevance']
        i = 0
        while i < len(plan):
            step = plan[i]

            # `cost_estimate` is no longer a graph node. For eval runs we inject
            # it here once scan metadata is available so the terminal `report`
            # node can include report.cost_estimate.
            if node_name == "eval" and step == "eval" and state.get("cost_estimate") is None:
                metadata = state.get("metadata")
                if metadata is not None:
                    state["cost_estimate"] = build_cost_estimate(
                        metadata,
                        state["job_config"],
                        cases=[case],
                    )

            # For eval target, metric nodes are independent and can run in parallel.
            if node_name == "eval" and metric_group and step == metric_group[0]:
                cached_group_outputs: dict[str, dict[str, Any]] = {}
                skipped_group_outputs: dict[str, dict[str, Any]] = {}
                to_run: list[str] = []

                for metric_step in metric_group:
                    if not NodeRunner.is_case_eligible_for_node(case, metric_step):
                        output = NodeRunner.skipped_output_for_node(metric_step)
                        skipped_group_outputs[metric_step] = output
                        self._cache.put(
                            case_hash,
                            config_hash,
                            metric_step,
                            output,
                            node_cost=NodeCostBreakdown(model_calls=0, cost_usd=0.0),
                        )
                        continue

                    if not force:
                        cached_entry = self._get_cached_entry(
                            case_hash=case_hash,
                            config_hash=config_hash,
                            step=metric_step,
                        )
                        if cached_entry is not None:
                            cached_group_outputs[metric_step] = cached_entry["node_output"]
                            cached.append(metric_step)
                            continue
                    to_run.append(metric_step)

                run_group_outputs: dict[str, dict[str, Any]] = {}
                if to_run:
                    with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
                        futures = {pool.submit(NODE_FNS[m], dict(state)): m for m in to_run}
                        for future in as_completed(futures):
                            metric_step = futures[future]
                            output = future.result()
                            run_group_outputs[metric_step] = output
                            executed.append(metric_step)
                            self._cache.put(
                                case_hash,
                                config_hash,
                                metric_step,
                                output,
                                node_cost=self._estimate_step_cost(estimator, metric_step, state),
                            )

                # Merge outputs in stable node order.
                for metric_step in metric_group:
                    merged_output = (
                        skipped_group_outputs.get(metric_step)
                        or cached_group_outputs.get(metric_step)
                        or run_group_outputs.get(metric_step)
                    )
                    if merged_output is not None:
                        state.update(merged_output)

                i += len(metric_group)
                continue

            if not NodeRunner.is_case_eligible_for_node(case, step):
                output = NodeRunner.skipped_output_for_node(step)
                state.update(output)
                self._cache.put(
                    case_hash,
                    config_hash,
                    step,
                    output,
                    node_cost=NodeCostBreakdown(model_calls=0, cost_usd=0.0),
                )
                if step == node_name:
                    node_output = output
                i += 1
                continue

            if not force:
                cached_entry = self._get_cached_entry(
                    case_hash=case_hash,
                    config_hash=config_hash,
                    step=step,
                )
                if cached_entry is not None:
                    cached_output = cached_entry["node_output"]
                    state.update(cached_output)
                    cached.append(step)
                    if step == node_name:
                        node_output = cached_output
                    i += 1
                    continue

            output = NODE_FNS[step](state)
            state.update(output)
            self._cache.put(
                case_hash,
                config_hash,
                step,
                output,
                node_cost=self._estimate_step_cost(estimator, step, state),
            )
            executed.append(step)
            if step == node_name:
                node_output = output
            i += 1

        elapsed_ms = (time.monotonic() - t0) * 1000

        return CachedNodeRunResult(
            node_name=node_name,
            case_id=case.case_id,
            executed_nodes=executed,
            cached_nodes=cached,
            node_output=node_output,
            final_state=dict(state),
            elapsed_ms=elapsed_ms,
        )
