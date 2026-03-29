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

from lumiseval_core.cache import CacheStore, compute_case_hash, compute_config_hash
from lumiseval_core.types import EvalCase, EvalJobConfig
from pydantic import BaseModel

from lumiseval_agent import graph as graph_module


class NodeRunResult(BaseModel):
    node_name: str
    executed_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]


class CachedNodeRunResult(BaseModel):
    node_name: str
    case_id: str
    executed_nodes: list[str]
    cached_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]
    elapsed_ms: float


class NodeRunner:
    """Execute a single node for a case, optionally running required prior nodes."""

    _legacy_aliases = {
        "metadata_scanner": "scan",
        "cost_estimator": "estimate",
        "confirm_gate": "approve",
        "chunker": "chunk",
        "claim_extractor": "claims",
        "mmr_deduplicator": "dedupe",
        "ragas": "relevance",
        "hallucination": "grounding",
        "adversarial": "redteam",
        "eval": "eval",
    }

    _node_fns = {
        "scan": graph_module.node_metadata_scanner,
        "estimate": graph_module.node_cost_estimator,
        "approve": graph_module.node_confirm_gate,
        "chunk": graph_module.node_chunk,
        "claims": graph_module.node_claims,
        "dedupe": graph_module.node_dedupe,
        "relevance": graph_module.node_relevance,
        "grounding": graph_module.node_grounding,
        "redteam": graph_module.node_adversarial,
        "rubric": graph_module.node_rubric,
        "eval": graph_module.node_eval,
    }

    _prerequisites = {
        "scan": [],
        "estimate": ["scan"],
        "approve": ["scan", "estimate"],
        "chunk": ["scan", "estimate", "approve"],
        "claims": [
            "scan",
            "estimate",
            "approve",
            "chunk",
        ],
        "dedupe": [
            "scan",
            "estimate",
            "approve",
            "chunk",
            "claims",
        ],
        "relevance": [
            "scan",
            "estimate",
            "approve",
            "chunk",
            "claims",
            "dedupe",
        ],
        "grounding": [
            "scan",
            "estimate",
            "approve",
            "chunk",
            "claims",
            "dedupe",
        ],
        "redteam": [
            "scan",
            "estimate",
            "approve",
        ],
        "rubric": [
            "scan",
            "estimate",
            "approve",
        ],
        "eval": [
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
        ],
    }
    _context_required_nodes = {"chunk", "claims", "dedupe", "relevance", "grounding"}
    _rubric_required_nodes = {"rubric"}
    _skip_outputs: dict[str, dict[str, Any]] = {
        "chunk": {"chunks": []},
        "claims": {"raw_claims": []},
        "dedupe": {"unique_claims": []},
        "relevance": {"relevance_metrics": []},
        "grounding": {"grounding_metrics": []},
        "redteam": {"redteam_metrics": []},
        "rubric": {"rubric_metrics": []},
    }

    @classmethod
    def normalize_node_name(cls, node_name: str) -> str:
        """Map legacy node names to canonical one-word names."""
        return cls._legacy_aliases.get(node_name, node_name)

    @staticmethod
    def _has_generation(case: EvalCase) -> bool:
        return bool(case.generation and case.generation.strip())

    @staticmethod
    def _has_context(case: EvalCase) -> bool:
        return any(
            (isinstance(item, str) and item.strip()) or (item is not None and str(item).strip())
            for item in (case.context or [])
        )

    @staticmethod
    def _has_rubric(case: EvalCase) -> bool:
        return len(case.rubric_rules or []) > 0

    @classmethod
    def is_case_eligible_for_node(cls, case: EvalCase, node_name: str) -> bool:
        if not cls._has_generation(case):
            return False
        if node_name in cls._context_required_nodes:
            return cls._has_context(case)
        if node_name in cls._rubric_required_nodes:
            return cls._has_rubric(case)
        return True

    @classmethod
    def skipped_output_for_node(cls, node_name: str) -> dict[str, Any]:
        return dict(cls._skip_outputs.get(node_name, {}))

    def run_case(
        self,
        *,
        case: EvalCase,
        node_name: str,
        job_config: EvalJobConfig | None = None,
        include_prerequisites: bool = True,
    ) -> NodeRunResult:
        node_name = self.normalize_node_name(node_name)
        if node_name not in self._node_fns:
            valid = ", ".join(sorted(self._node_fns))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        state = graph_module.build_initial_state(
            generation=case.generation,
            job_config=job_config,
            question=case.question,
            ground_truth=case.ground_truth,
            context=case.context,
            target_node=node_name,
            rubric_rules=case.rubric_rules,
            reference_files=case.reference_files,
        )

        plan = list(self._prerequisites[node_name]) if include_prerequisites else []
        plan.append(node_name)

        node_output: dict[str, Any] = {}
        executed_nodes: list[str] = []
        for step in plan:
            if not self.is_case_eligible_for_node(case, step):
                updates = self.skipped_output_for_node(step)
            else:
                updates = self._node_fns[step](state)
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
        self._cache = cache_store

    def run_case(
        self,
        *,
        case: EvalCase,
        node_name: str,
        job_config: EvalJobConfig | None = None,
        force: bool = False,
    ) -> CachedNodeRunResult:
        """Run *node_name* for *case*, using the cache for already-computed steps.

        # Note: This function runs for each record
        Args:
            case: The evaluation case to run.
            node_name: Target node name (must be in NodeRunner._node_fns).
            job_config: Evaluation config; a default is created if not provided.
            force: If True, skip cache reads (still writes new outputs to cache).

        Returns:
            CachedNodeRunResult with lists of executed vs cached nodes.
        """
        node_name = NodeRunner.normalize_node_name(node_name)
        if node_name not in NodeRunner._node_fns:
            valid = ", ".join(sorted(NodeRunner._node_fns))
            raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

        t0 = time.monotonic()

        state = graph_module.build_initial_state(
            generation=case.generation,
            job_config=job_config,
            question=case.question,
            ground_truth=case.ground_truth,
            context=case.context,
            target_node=node_name,
            rubric_rules=case.rubric_rules,
            reference_files=case.reference_files,
        )

        # Compute an HashID for Caching
        case_hash = compute_case_hash(
            generation=case.generation,
            question=case.question,
            ground_truth=case.ground_truth,
            rubric_rules=case.rubric_rules or [],
            context=case.context or [],
            reference_files=case.reference_files or [],
        )
        config_hash = compute_config_hash(state["job_config"])

        plan: list[str] = list(NodeRunner._prerequisites[node_name]) + [node_name]
        executed: list[str] = []
        cached: list[str] = []
        node_output: dict[str, Any] = {}
        metric_group = ["relevance", "grounding", "redteam", "rubric"]

        # A plan may look like below
        # plan:  ['scan', 'estimate']
        i = 0
        while i < len(plan):
            step = plan[i]

            # For eval target, metric nodes are independent and can run in parallel.
            if node_name == "eval" and step == "relevance":
                cached_group_outputs: dict[str, dict[str, Any]] = {}
                skipped_group_outputs: dict[str, dict[str, Any]] = {}
                to_run: list[str] = []

                for metric_step in metric_group:
                    if not NodeRunner.is_case_eligible_for_node(case, metric_step):
                        output = NodeRunner.skipped_output_for_node(metric_step)
                        skipped_group_outputs[metric_step] = output
                        self._cache.put(case_hash, config_hash, metric_step, output)
                        continue
                    if not force and self._cache.has(case_hash, config_hash, metric_step):
                        cached_output = self._cache.get(case_hash, config_hash, metric_step)
                        if cached_output is not None:
                            cached_group_outputs[metric_step] = cached_output
                            cached.append(metric_step)
                            continue
                    to_run.append(metric_step)

                run_group_outputs: dict[str, dict[str, Any]] = {}
                if to_run:
                    with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
                        futures = {
                            pool.submit(NodeRunner._node_fns[m], dict(state)): m for m in to_run
                        }
                        for future in as_completed(futures):
                            metric_step = futures[future]
                            output = future.result()
                            run_group_outputs[metric_step] = output
                            executed.append(metric_step)
                            self._cache.put(case_hash, config_hash, metric_step, output)

                # Merge outputs in stable node order.
                for metric_step in metric_group:
                    output = (
                        skipped_group_outputs.get(metric_step)
                        or cached_group_outputs.get(metric_step)
                        or run_group_outputs.get(metric_step)
                    )
                    if output is not None:
                        state.update(output)

                i += len(metric_group)
                continue

            if not NodeRunner.is_case_eligible_for_node(case, step):
                output = NodeRunner.skipped_output_for_node(step)
                state.update(output)
                self._cache.put(case_hash, config_hash, step, output)
                if step == node_name:
                    node_output = output
                i += 1
                continue

            if not force and self._cache.has(case_hash, config_hash, step):
                cached_output = self._cache.get(case_hash, config_hash, step)
                if cached_output is not None:
                    state.update(cached_output)
                    cached.append(step)
                    if step == node_name:
                        node_output = cached_output
                    i += 1
                    continue

            output = NodeRunner._node_fns[step](state)
            state.update(output)
            self._cache.put(case_hash, config_hash, step, output)
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
