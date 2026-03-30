"""
Node function registry — maps canonical node names to their callable implementations.

This is the only file that imports both lumiseval_core.pipeline (pure data) and
the graph module (callables), keeping circular imports out of graph.py and node_runner.py.

To add a new metric node:
  1. Write the node function in lumiseval_graph/nodes/metrics/<name>.py
  2. Add a NodeSpec entry to PIPELINE in lumiseval_core/pipeline.py
  3. Add one entry to NODE_FNS here
  4. Add g.add_edge() calls in lumiseval_graph/graph.py build_graph()
"""

from typing import Any, Callable

from lumiseval_core.pipeline import PIPELINE

from lumiseval_graph import graph as _graph

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

NODE_FNS: dict[str, NodeFn] = {
    "scan": _graph.node_metadata_scanner,
    "estimate": _graph.node_cost_estimator,
    "approve": _graph.node_confirm_gate,
    "chunk": _graph.node_chunk,
    "claims": _graph.node_claims,
    "dedupe": _graph.node_dedupe,
    "relevance": _graph.node_relevance,
    "grounding": _graph.node_grounding,
    "redteam": _graph.node_adversarial,
    "rubric": _graph.node_rubric,
    "eval": _graph.node_eval,
}

# Guard: every pipeline node must have a registered function
_missing = [s.name for s in PIPELINE if s.name not in NODE_FNS]
if _missing:
    raise RuntimeError(f"registry.py is missing NODE_FNS entries: {_missing}")
