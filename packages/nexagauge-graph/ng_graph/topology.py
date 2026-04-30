"""
Centralized pipeline topology registry.

Single source of truth for node ordering, prerequisite chains, eligibility
flags, display colors, environment key prefixes, and skip-output shapes.

Function objects are NOT stored here — they live in nexagauge-graph so this
module can be imported by nexagauge-core, nexagauge-graph,
and nexagauge-cli without creating circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_ARTIFACT_KINDS: frozenset[str] = frozenset(
    {
        "none",
        "inputs",
        "chunks",
        "claims",
        "geval_steps",
        "metric_results",
        "eval_summary",
        "report",
    }
)


@dataclass(frozen=True)
class NodeSpec:
    """Immutable specification for one pipeline node."""

    # ── Identity ──────────────────────────────────────────────────────────
    name: str

    # ── Topology ──────────────────────────────────────────────────────────
    prerequisites: tuple[str, ...] = ()
    """Direct parent edges for this node in the pipeline DAG.

    List only the *immediate* predecessors — transitive ancestors are expanded
    on demand via :func:`transitive_prerequisites`. For example, ``grounding``
    declares ``("claims",)``; the full ``scan → chunk → refiner → claims`` chain
    is derived, not repeated here.
    """

    # ── Eligibility ───────────────────────────────────────────────────────
    requires_generation: bool = False

    # ── Eligibility ───────────────────────────────────────────────────────
    requires_context: bool = False
    """Skip this node when the case has no context passages."""

    # ── Eligibility ───────────────────────────────────────────────────────
    requires_question: bool = False
    """Skip this node when the case has no question text."""

    requires_geval: bool = False
    """Skip this node when the case has no GEval metrics."""

    requires_reference: bool = False
    """Skip this node when the case has no reference answer."""

    is_metric: bool = False
    """Node produces metric results and participates in the parallel fan-out."""

    is_utility: bool = False
    """Node produces direct/indirect artifacts for metric nodes but is not a metric itself (e.g. claim extraction)."""

    is_preflight: bool = False
    """Node runs before any LLM calls and is part of lightweight preflight passes."""

    # ── Artifact contract ───────────────────────────────────────────────────
    stream: str | None = None
    """Namespace prefix for state keys, e.g. ``generation`` or ``context``."""

    state_key: str | None = None
    """Primary output state key for this node."""

    artifact_in_kind: str = "none"
    """Expected upstream artifact kind consumed by this node.

    Current values: ``none`` | ``inputs`` | ``chunks`` | ``claims`` |
    ``geval_steps`` | ``metric_results`` | ``eval_summary`` | ``report``.
    """

    artifact_out_kind: str = "none"
    """Artifact kind emitted by this node.

    Current values: ``none`` | ``inputs`` | ``chunks`` | ``claims`` |
    ``geval_steps`` | ``metric_results`` | ``eval_summary`` | ``report``.
    """

    is_transform: bool = False
    """True when node transforms one artifact kind to the same kind."""

    route_via_artifact_graph: bool = False
    """Resolve ``artifact_in_kind`` by walking topology ancestors when True."""

    # ── Display ───────────────────────────────────────────────────────────
    color: str = "white"
    """Rich color string used by NodeLogger."""

    # ── Naming ────────────────────────────────────────────────────────────
    env_key_suffixes: tuple[str, ...] = ()
    """Upper-case env-key suffixes tried when reading LLM_*_MODEL env vars."""

    # ── Skip output ───────────────────────────────────────────────────────
    skip_output: dict[str, Any] = field(default_factory=dict)
    """State patch applied when a node is skipped (ineligible case)."""


# ── Pipeline definition ────────────────────────────────────────────────────────
# To add a new metric node:
#   1. Append a NodeSpec here.
#   2. Add one entry to NODE_FNS in ng_graph/registry.py.
#   3. Write the node function in ng_graph/nodes/metrics/<name>.py.
#   4. Add g.add_edge() calls in ng_graph/graph.py build_graph().

PIPELINE: list[NodeSpec] = [
    NodeSpec(
        name="scan",
        prerequisites=(),
        artifact_in_kind="none",
        artifact_out_kind="inputs",
        is_preflight=True,
        color="cyan",
        env_key_suffixes=("SCAN",),
    ),
    NodeSpec(
        name="chunk",
        prerequisites=("scan",),
        requires_generation=True,
        stream="generation",
        state_key="generation_chunk",
        artifact_in_kind="inputs",
        artifact_out_kind="chunks",
        color="blue",
        is_utility=True,
        env_key_suffixes=("CHUNK",),
    ),
    NodeSpec(
        name="refiner",
        prerequisites=("chunk",),
        requires_generation=True,
        stream="generation",
        state_key="generation_refined_chunks",
        artifact_in_kind="chunks",
        artifact_out_kind="chunks",
        is_transform=True,
        route_via_artifact_graph=True,
        color="green",
        is_utility=True,
        env_key_suffixes=("REFINER",),
    ),
    NodeSpec(
        name="claims",
        prerequisites=("refiner",),
        requires_generation=True,
        stream="generation",
        state_key="generation_claims",
        artifact_in_kind="chunks",
        artifact_out_kind="claims",
        route_via_artifact_graph=True,
        color="magenta",
        is_utility=True,
        env_key_suffixes=("CLAIMS",),
    ),
    NodeSpec(
        name="geval_steps",
        prerequisites=("scan",),
        requires_generation=True,
        requires_geval=True,
        state_key="geval_steps",
        artifact_in_kind="inputs",
        artifact_out_kind="geval_steps",
        is_utility=True,
        color="dark_sea_green4",
        env_key_suffixes=("GEVAL_STEPS",),
    ),
    NodeSpec(
        name="relevance",
        prerequisites=("claims",),
        requires_generation=True,
        requires_question=True,
        state_key="relevance_metrics",
        artifact_in_kind="claims",
        artifact_out_kind="metric_results",
        route_via_artifact_graph=True,
        is_metric=True,
        color="bright_green",
        env_key_suffixes=("RELEVANCE",),
    ),
    NodeSpec(
        name="grounding",
        prerequisites=("claims",),
        requires_generation=True,
        requires_context=True,
        state_key="grounding_metrics",
        artifact_in_kind="claims",
        artifact_out_kind="metric_results",
        route_via_artifact_graph=True,
        is_metric=True,
        color="cornflower_blue",
        env_key_suffixes=("GROUNDING",),
    ),
    NodeSpec(
        name="redteam",
        prerequisites=("scan",),
        requires_generation=True,
        state_key="redteam_metrics",
        artifact_in_kind="inputs",
        artifact_out_kind="metric_results",
        is_metric=True,
        color="red",
        env_key_suffixes=("REDTEAM",),
    ),
    NodeSpec(
        name="geval",
        prerequisites=("geval_steps",),
        requires_generation=True,
        requires_geval=True,
        state_key="geval_metrics",
        artifact_in_kind="geval_steps",
        artifact_out_kind="metric_results",
        is_metric=True,
        color="medium_purple3",
        env_key_suffixes=("GEVAL",),
    ),
    NodeSpec(
        name="reference",
        prerequisites=("scan",),
        requires_generation=True,
        requires_reference=True,
        state_key="reference_metrics",
        artifact_in_kind="inputs",
        artifact_out_kind="metric_results",
        is_metric=True,
        color="bright_magenta",
        env_key_suffixes=("REFERENCE",),
    ),
    NodeSpec(
        name="eval",
        prerequisites=(
            "chunk",
            "refiner",
            "claims",
            "geval_steps",
            "relevance",
            "grounding",
            "redteam",
            "geval",
            "reference",
        ),
        state_key="eval_summary",
        artifact_in_kind="metric_results",
        artifact_out_kind="eval_summary",
        color="gold1",
        env_key_suffixes=("EVAL",),
    ),
    NodeSpec(
        name="report",
        prerequisites=("eval",),
        state_key="report",
        artifact_in_kind="eval_summary",
        artifact_out_kind="report",
        color="gold3",
        env_key_suffixes=("REPORT",),
    ),
]

# ── Derived constants (computed once at import time) ──────────────────────────
# Adding a new node? Just append a NodeSpec to PIPELINE above — nothing else here needs updating.

NODES_BY_NAME: dict[str, NodeSpec] = {s.name: s for s in PIPELINE}

NODE_ORDER: list[str] = [s.name for s in PIPELINE]
"""Stable ordered list of all canonical node names."""

METRIC_NODES: list[str] = [s.name for s in PIPELINE if s.is_metric]
"""Ordered list of metric node names (the parallel fan-out group)."""
UTILITY_NODES: list[str] = [s.name for s in PIPELINE if s.is_utility]

# Nodes excluded from --debug per-case logging. These are either pure
# orchestration joins or declarative aggregation — no LLM work to narrate.
DEBUG_SKIP_NODES: frozenset[str] = frozenset({"eval", "report", "scan"})


def transitive_prerequisites(node_name: str) -> tuple[str, ...]:
    """Return all ancestors of ``node_name`` in topological (PIPELINE) order.

    Walks direct ``prerequisites`` edges recursively and deduplicates. The
    result does not include ``node_name`` itself. Emission order follows
    ``NODE_ORDER`` so callers (plan builders, cache fingerprinting) see a
    stable, deterministic sequence independent of DFS traversal order.
    """
    seen: set[str] = set()

    def _visit(name: str) -> None:
        for parent in NODES_BY_NAME[name].prerequisites:
            if parent in seen:
                continue
            seen.add(parent)
            _visit(parent)

    _visit(node_name)
    return tuple(n for n in NODE_ORDER if n in seen)


def _topology_validation_errors() -> list[str]:
    errors: list[str] = []
    known = set(NODES_BY_NAME)

    for spec in PIPELINE:
        for parent in spec.prerequisites:
            if parent not in known:
                errors.append(f"Node '{spec.name}' has unknown prerequisite '{parent}'.")
        if spec.state_key and not spec.state_key.startswith("_") and " " in spec.state_key:
            errors.append(f"Node '{spec.name}' has invalid state_key '{spec.state_key}'.")
        if spec.artifact_in_kind not in _ARTIFACT_KINDS:
            errors.append(
                f"Node '{spec.name}' has unsupported artifact_in_kind '{spec.artifact_in_kind}'."
            )
        if spec.artifact_out_kind not in _ARTIFACT_KINDS:
            errors.append(
                f"Node '{spec.name}' has unsupported artifact_out_kind '{spec.artifact_out_kind}'."
            )
        if not spec.artifact_in_kind:
            errors.append(f"Node '{spec.name}' must define artifact_in_kind.")
        if not spec.artifact_out_kind:
            errors.append(f"Node '{spec.name}' must define artifact_out_kind.")
        if spec.is_transform and spec.artifact_in_kind != spec.artifact_out_kind:
            errors.append(
                f"Node '{spec.name}' is_transform requires matching artifact kinds, got "
                f"'{spec.artifact_in_kind}' -> '{spec.artifact_out_kind}'."
            )
        if spec.is_transform and spec.artifact_in_kind == "none":
            errors.append(f"Node '{spec.name}' is_transform cannot use artifact kind 'none'.")

    # Validate artifact-routed nodes can resolve a unique upstream producer.
    for spec in PIPELINE:
        if not spec.route_via_artifact_graph:
            continue

        producers_by_depth: dict[str, int] = {}

        def _visit(name: str, seen: set[str], depth: int) -> None:
            for parent in NODES_BY_NAME[name].prerequisites:
                if parent in seen:
                    continue
                seen.add(parent)
                p_spec = NODES_BY_NAME[parent]
                if p_spec.artifact_out_kind == spec.artifact_in_kind and p_spec.state_key:
                    existing = producers_by_depth.get(parent)
                    producers_by_depth[parent] = depth if existing is None else min(existing, depth)
                _visit(parent, seen, depth + 1)

        _visit(spec.name, set(), 1)
        if not producers_by_depth:
            errors.append(
                f"Node '{spec.name}' expects artifact kind '{spec.artifact_in_kind}' but has no "
                "upstream producer."
            )
            continue

        min_depth = min(producers_by_depth.values())
        nearest = sorted(node for node, depth in producers_by_depth.items() if depth == min_depth)
        if len(nearest) > 1:
            errors.append(
                f"Node '{spec.name}' has ambiguous upstream producers for "
                f"'{spec.artifact_in_kind}': {', '.join(nearest)}."
            )

    return errors


_TOPOLOGY_ERRORS = _topology_validation_errors()
if _TOPOLOGY_ERRORS:
    raise RuntimeError("Invalid topology:\n- " + "\n- ".join(_TOPOLOGY_ERRORS))
