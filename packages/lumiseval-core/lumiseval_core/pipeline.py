"""
Centralized pipeline topology registry.

Single source of truth for node ordering, prerequisite chains, eligibility
flags, display colors, legacy name aliases, environment key prefixes, and
skip-output shapes.

Function objects are NOT stored here — they live in lumiseval-graph so this
module can be imported by lumiseval-core, lumiseval-ingest, lumiseval-graph,
and lumiseval-cli without creating circular imports.
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
    """Ordered names of all nodes that must have run before this node."""

    # ── Eligibility ───────────────────────────────────────────────────────
    requires_context: bool = False
    """Skip this node when the case has no context passages."""

    requires_rubric: bool = False
    """Skip this node when the case has no rubric rules."""

    requires_reference: bool = False
    """Skip this node when the case has no reference reference."""

    is_metric: bool = False
    """Node produces metric results and participates in the parallel fan-out."""

    is_preflight: bool = False
    """Node runs before any LLM calls; CLI skips cost confirmation when target is preflight-only."""

    # ── Cost routing ──────────────────────────────────────────────────────
    cost_category: str = ""
    """Logical cost group.  Values: 'claim_extraction' | '' (none)."""

    # ── Display ───────────────────────────────────────────────────────────
    color: str = "white"
    """Rich color string used by NodeLogger."""

    # ── Naming ────────────────────────────────────────────────────────────
    legacy_aliases: tuple[str, ...] = ()
    """Historical node names that normalize to `name`."""

    env_key_suffixes: tuple[str, ...] = ()
    """Upper-case env-key suffixes tried when reading LLM_*_MODEL env vars."""

    # ── Skip output ───────────────────────────────────────────────────────
    skip_output: dict[str, Any] = field(default_factory=dict)
    """State patch applied when a node is skipped (ineligible case)."""


# ── Pipeline definition ────────────────────────────────────────────────────────
# To add a new metric node:
#   1. Append a NodeSpec here.
#   2. Add one entry to NODE_FNS in lumiseval_graph/registry.py.
#   3. Write the node function in lumiseval_graph/nodes/metrics/<name>.py.
#   4. Add g.add_edge() calls in lumiseval_graph/graph.py build_graph().

PIPELINE: list[NodeSpec] = [
    NodeSpec(
        name="scan",
        prerequisites=(),
        is_preflight=True,
        color="cyan",
        legacy_aliases=("metadata_scanner",),
        env_key_suffixes=("SCAN", "METADATA_SCANNER"),
    ),
    NodeSpec(
        name="estimate",
        prerequisites=("scan",),
        is_preflight=True,
        color="yellow",
        legacy_aliases=("cost_estimator",),
        env_key_suffixes=("ESTIMATE", "COST_ESTIMATOR"),
    ),
    NodeSpec(
        name="approve",
        prerequisites=("scan", "estimate"),
        is_preflight=True,
        color="white",
        legacy_aliases=("confirm_gate",),
        env_key_suffixes=("APPROVE", "CONFIRM_GATE"),
    ),
    NodeSpec(
        name="chunk",
        prerequisites=("scan", "estimate", "approve"),
        requires_context=True,
        cost_category="claim_extraction",
        color="blue",
        legacy_aliases=("chunker",),
        env_key_suffixes=("CHUNK", "CHUNKER"),
        skip_output={"chunks": []},
    ),
    NodeSpec(
        name="claims",
        prerequisites=("scan", "estimate", "approve", "chunk"),
        requires_context=True,
        cost_category="claim_extraction",
        color="magenta",
        legacy_aliases=("claim_extractor",),
        env_key_suffixes=("CLAIMS", "CLAIM_EXTRACTOR"),
        skip_output={"raw_claims": []},
    ),
    NodeSpec(
        name="dedupe",
        prerequisites=("scan", "estimate", "approve", "chunk", "claims"),
        requires_context=True,
        color="green",
        legacy_aliases=("mmr_deduplicator",),
        env_key_suffixes=("DEDUPE", "MMR_DEDUPLICATOR"),
        skip_output={"unique_claims": []},
    ),
    NodeSpec(
        name="relevance",
        prerequisites=("scan", "estimate", "approve", "chunk", "claims", "dedupe"),
        requires_context=True,
        is_metric=True,
        color="bright_green",
        legacy_aliases=("ragas",),
        env_key_suffixes=("RELEVANCE", "RAGAS"),
        skip_output={"relevance_metrics": []},
    ),
    NodeSpec(
        name="grounding",
        prerequisites=("scan", "estimate", "approve", "chunk", "claims", "dedupe"),
        requires_context=True,
        is_metric=True,
        color="cornflower_blue",
        legacy_aliases=("hallucination",),
        env_key_suffixes=("GROUNDING", "HALLUCINATION"),
        skip_output={"grounding_metrics": []},
    ),
    NodeSpec(
        name="redteam",
        prerequisites=("scan", "estimate", "approve"),
        is_metric=True,
        color="red",
        legacy_aliases=("adversarial",),
        env_key_suffixes=("REDTEAM", "ADVERSARIAL"),
        skip_output={"redteam_metrics": []},
    ),
    NodeSpec(
        name="rubric",
        prerequisites=("scan", "estimate", "approve"),
        requires_rubric=True,
        is_metric=True,
        color="orchid",
        env_key_suffixes=("RUBRIC", "RUBRIC_EVAL"),
        skip_output={"rubric_metrics": []},
    ),
    NodeSpec(
        name="reference",
        prerequisites=("scan", "estimate", "approve"),
        requires_reference=True,
        is_metric=True,
        color="bright_magenta",
        env_key_suffixes=("REFERENCE",),
        skip_output={"reference_metrics": []},
    ),
    NodeSpec(
        name="eval",
        prerequisites=(
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
            "reference",
        ),
        color="gold1",
        env_key_suffixes=("EVAL",),
    ),
]

# ── Derived constants (computed once at import time) ──────────────────────────

NODES_BY_NAME: dict[str, NodeSpec] = {s.name: s for s in PIPELINE}

NODE_ORDER: list[str] = [s.name for s in PIPELINE]
"""Stable ordered list of all canonical node names."""

LEGACY_ALIASES: dict[str, str] = {alias: s.name for s in PIPELINE for alias in s.legacy_aliases}
"""Flat map from every legacy alias to its canonical node name."""

CONTEXT_REQUIRED_NODES: frozenset[str] = frozenset(s.name for s in PIPELINE if s.requires_context)

RUBRIC_REQUIRED_NODES: frozenset[str] = frozenset(s.name for s in PIPELINE if s.requires_rubric)

REFERENCE_REQUIRED_NODES: frozenset[str] = frozenset(
    s.name for s in PIPELINE if s.requires_reference
)

PREFLIGHT_NODES: frozenset[str] = frozenset(s.name for s in PIPELINE if s.is_preflight)

METRIC_NODES: list[str] = [s.name for s in PIPELINE if s.is_metric]
"""Ordered list of metric node names (the parallel fan-out group)."""

NODE_COLORS: dict[str, str] = {s.name: s.color for s in PIPELINE}

SKIP_OUTPUTS: dict[str, dict[str, Any]] = {s.name: s.skip_output for s in PIPELINE if s.skip_output}

NODE_PREREQUISITES: dict[str, list[str]] = {s.name: list(s.prerequisites) for s in PIPELINE}

NODE_ENV_KEY_SUFFIXES: dict[str, list[str]] = {s.name: list(s.env_key_suffixes) for s in PIPELINE}


def normalize_node_name(node_name: str) -> str:
    """Map any legacy alias or canonical name to the canonical node name."""
    return LEGACY_ALIASES.get(node_name, node_name)
