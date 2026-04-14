"""GEval metric branch nodes.

GEval metrics should stay focused on one dimension per metric so scoring
remains stable and interpretable. Keep evaluation steps aligned to the same
dimension instead of mixing unrelated checks in a single metric.
"""

from lumoseval_graph.nodes.metrics.geval.steps import GevalStepsNode

__all__ = ["GevalNode", "GevalStepsNode"]


def __getattr__(name: str):
    if name == "GevalNode":
        from lumoseval_graph.nodes.metrics.geval.score import GevalNode

        return GevalNode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
