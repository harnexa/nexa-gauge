"""Chunk Extractor Node."""

import hashlib

import semchunk

from lumiseval_core.constants import CHUNK_MIN_TOKENS_FOR_SPLIT, GENERATION_CHUNK_SIZE_TOKENS
from lumiseval_core.types import Chunk, ChunkArtifacts, CostEstimate, Item
from lumiseval_core.utils import _count_tokens
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseNode

log = get_node_logger("chunk")


class ChunkExtractorNode(BaseNode):
    node_name = "chunk"

    def __init__(self, chunk_size: int = GENERATION_CHUNK_SIZE_TOKENS) -> None:
        self.chunk_size = chunk_size

    def run(self, item: Item) -> ChunkArtifacts:  # type: ignore[override]
        text = item.text
        num_tokens = item.tokens if item.tokens > 0 else float(_count_tokens(text))
        cost = CostEstimate(
            input_tokens=0, output_tokens=0, cost=0,
        )

        if num_tokens < CHUNK_MIN_TOKENS_FOR_SPLIT:
            chunk = Chunk(
                index=0,
                item=Item(text=text, tokens=num_tokens),
                char_start=0,
                char_end=len(text),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
            return ChunkArtifacts(chunks=[chunk], cost=cost)

        chunker = semchunk.chunkerify(_count_tokens, self.chunk_size)
        raw_chunks: list[str] = list(chunker(text))

        chunks: list[Chunk] = []
        cursor = 0
        for i, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, cursor)
            if start < 0:
                start = cursor
            end = start + len(chunk_text)
            chunks.append(
                Chunk(
                    index=i,
                    item=Item(text=chunk_text, tokens=float(_count_tokens(chunk_text))),
                    char_start=start,
                    char_end=end,
                    sha256=hashlib.sha256(chunk_text.encode()).hexdigest(),
                )
            )
            cursor = end

       
        log.success(f"{len(chunks)} chunk(s) produced")
        return ChunkArtifacts(chunks=chunks, cost=cost)

    def estimate(self, item: Item) -> CostEstimate:
        chuck_artifact = self.run(item=item)
        return chuck_artifact.cost
