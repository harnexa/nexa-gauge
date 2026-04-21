"""
Registry for graph node handlers.

This module maps canonical pipeline node names (defined in ``PIPELINE``) to
their executable functions in ``ng_graph.graph``. It is the central binding
layer between topology (node order/definitions) and runtime execution
(callable implementations), and includes a startup guard that fails fast when
a pipeline node is missing from ``NODE_FNS``.

To add a new node:
1. Implement the node function in the graph/nodes layer.
2. Add the node spec to ``ng_graph.topology.PIPELINE``.
3. Register the node name and function in ``NODE_FNS`` below.
4. Add required graph edges in ``ng_graph.graph.build_graph``.
"""

from typing import Any, Callable

from ng_graph import graph as _graph
from ng_graph.topology import PIPELINE

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

NODE_FNS: dict[str, NodeFn] = {
    "scan": _graph.node_metadata_scanner,
    "chunk": _graph.node_generation_chunk,
    "claims": _graph.node_generation_claims,
    "dedup": _graph.node_generation_claims_dedup,
    "geval_steps": _graph.node_geval_steps,
    "relevance": _graph.node_relevance,
    "grounding": _graph.node_grounding,
    "redteam": _graph.node_redteam,
    "geval": _graph.node_geval,
    "reference": _graph.node_reference,
    "eval": _graph.node_eval,
    "report": _graph.node_report,
}

# Guard: every pipeline node must have a registered function
_missing = [s.name for s in PIPELINE if s.name not in NODE_FNS]
if _missing:
    raise RuntimeError(f"registry.py is missing NODE_FNS entries: {_missing}")
