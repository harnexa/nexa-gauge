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
    declares ``("dedup",)``; the full ``scan → chunk → claims → dedup`` chain
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

    # ── Cost routing ──────────────────────────────────────────────────────
    cost_category: str = ""
    """Logical cost group.  Values: 'claim_extraction' | '' (none)."""

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
        is_preflight=True,
        color="cyan",
        env_key_suffixes=("SCAN",),
    ),
    NodeSpec(
        name="chunk",
        prerequisites=("scan",),
        requires_generation=True,
        cost_category="claim_extraction",
        color="blue",
        is_utility=True,
        env_key_suffixes=("CHUNK",),
        skip_output={"generation_chunk": None},
    ),
    NodeSpec(
        name="claims",
        prerequisites=("chunk",),
        requires_generation=True,
        cost_category="claim_extraction",
        color="magenta",
        is_utility=True,
        env_key_suffixes=("CLAIMS",),
        skip_output={"generation_claims": None},
    ),
    NodeSpec(
        name="dedup",
        prerequisites=("claims",),
        requires_generation=True,
        color="green",
        is_utility=True,
        env_key_suffixes=("DEDUP",),
        skip_output={"generation_dedup_claims": None},
    ),
    NodeSpec(
        name="geval_steps",
        prerequisites=("scan",),
        requires_generation=True,
        requires_geval=True,
        is_utility=True,
        color="dark_sea_green4",
        env_key_suffixes=("GEVAL_STEPS",),
        skip_output={
            "geval_steps_by_signature": {},
        },
    ),
    NodeSpec(
        name="relevance",
        prerequisites=("dedup",),
        requires_generation=True,
        requires_question=True,
        is_metric=True,
        color="bright_green",
        env_key_suffixes=("RELEVANCE",),
        skip_output={"relevance_metrics": None},
    ),
    NodeSpec(
        name="grounding",
        prerequisites=("dedup",),
        requires_generation=True,
        requires_context=True,
        is_metric=True,
        color="cornflower_blue",
        env_key_suffixes=("GROUNDING",),
        skip_output={"grounding_metrics": None},
    ),
    NodeSpec(
        name="redteam",
        prerequisites=("scan",),
        requires_generation=True,
        is_metric=True,
        color="red",
        env_key_suffixes=("REDTEAM",),
        skip_output={"redteam_metrics": None},
    ),
    NodeSpec(
        name="geval",
        prerequisites=("geval_steps",),
        requires_generation=True,
        requires_geval=True,
        is_metric=True,
        color="medium_purple3",
        env_key_suffixes=("GEVAL",),
        skip_output={"geval_metrics": None},
    ),
    NodeSpec(
        name="reference",
        prerequisites=("scan",),
        requires_generation=True,
        requires_reference=True,
        is_metric=True,
        color="bright_magenta",
        env_key_suffixes=("REFERENCE",),
        skip_output={"reference_metrics": None},
    ),
    NodeSpec(
        name="eval",
        prerequisites=(
            "chunk",
            "claims",
            "dedup",
            "geval_steps",
            "relevance",
            "grounding",
            "redteam",
            "geval",
            "reference",
        ),
        color="gold1",
        env_key_suffixes=("EVAL",),
    ),
    NodeSpec(
        name="report",
        prerequisites=("eval",),
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
