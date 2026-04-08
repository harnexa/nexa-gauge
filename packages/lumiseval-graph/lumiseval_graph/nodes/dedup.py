"""Dedup Node — removes near-duplicate items using MMR."""

from lumiseval_core.dedup.mmr import deduplicate
from lumiseval_core.types import CostEstimate, DedupArtifacts, Item
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseNode

log = get_node_logger("dedup")


class DedupNode(BaseNode):
    node_name = "dedup"

    def run(self, items: list[Item]) -> DedupArtifacts:  # type: ignore[override]
        unique_items, dedup_map = deduplicate(items)
        dropped = len(items) - len(unique_items)
        log.success(f"{len(unique_items)} unique item(s) kept ({dropped} duplicate(s) removed)")
        return DedupArtifacts(
            items=unique_items,
            dropped=dropped,
            dedup_map=dedup_map,
            cost=self.estimate(0.0, 0.0),
        )

    def estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:  # type: ignore[override]
        return CostEstimate(input_tokens=0.0, output_tokens=0.0, cost=0.0)
