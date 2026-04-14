"""
Node function registry — maps canonical node names to their callable implementations.

This is the only file that imports both lumos_core.pipeline (pure data) and
the graph module (callables), keeping circular imports out of graph.py and node_runner.py.

To add a new metric node:
  1. Write the node function in lumos_graph/nodes/metrics/<name>.py
  2. Add a NodeSpec entry to PIPELINE in lumos_graph/topology.py
  3. Add one entry to NODE_FNS here
  4. Add g.add_edge() calls in lumos_graph/graph.py build_graph()
"""

from typing import Any, Callable

from lumoseval_graph import graph as _graph
from lumoseval_graph.topology import PIPELINE

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
