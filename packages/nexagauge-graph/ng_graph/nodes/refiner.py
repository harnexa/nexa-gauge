"""Refiner node — selects top-k items via a pluggable strategy."""

from ng_core.constants import DEFAULT_REFINER_STRATEGY, REFINER_TOP_K
from ng_core.dedup.mmr import deduplicate
from ng_core.types import CostEstimate, Item, RefinerArtifacts
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseNode

log = get_node_logger("refiner")


class RefinerNode(BaseNode):
    node_name = "refiner"

    def __init__(
        self,
        *,
        strategy: str = DEFAULT_REFINER_STRATEGY,
        top_k: int = REFINER_TOP_K,
    ) -> None:
        self.strategy = str(strategy).strip().lower()
        self.top_k = int(top_k)

    def run(self, items: list[Item]) -> RefinerArtifacts:  # type: ignore[override]
        if self.strategy == "mmr":
            selected_indices, dedup_map = deduplicate(items, top_k=self.top_k)
        else:
            raise ValueError(
                f"Unsupported refiner strategy '{self.strategy}'. Supported strategies: mmr."
            )

        unique_items: list[Item] = [items[i] for i in selected_indices]
        dropped = len(items) - len(unique_items)
        log.success(
            f"{len(unique_items)} refined item(s) kept ({dropped} filtered out, top_k={self.top_k})"
        )
        return RefinerArtifacts(
            items=unique_items,
            indices=selected_indices,
            dropped=dropped,
            dedup_map=dedup_map,
            cost=self.estimate(0.0, 0.0),
        )

    def estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:  # type: ignore[override]
        del input_tokens, output_tokens
        return CostEstimate(input_tokens=0.0, output_tokens=0.0, cost=0.0)
