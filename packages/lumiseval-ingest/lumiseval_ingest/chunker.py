"""
Chunker — splits generation text into semantically coherent chunks using semchunk.

Uses 512-token target chunk size with 64-token overlap and respects semantic
boundaries (paragraph / sentence). Each chunk is SHA-256 hashed for dedup across
repeated runs on the same text.

TODO: Implement overlap logic (semchunk does not natively support overlap; consider
      sliding window post-processing).
"""

import hashlib

import semchunk
from lumiseval_core.constants import CHUNK_MIN_TOKENS_FOR_SPLIT
from lumiseval_core.types import Chunk
from lumiseval_core.utils import _count_tokens



_MIN_TOKENS_FOR_SPLIT = CHUNK_MIN_TOKENS_FOR_SPLIT


def chunk_text(text: str, chunk_size: int) -> list[Chunk]:
    """Split ``text`` into semantic chunks.

    If the text is under ``_MIN_TOKENS_FOR_SPLIT`` tokens it is returned as a
    single chunk without splitting.
    """
    if _count_tokens(text) < _MIN_TOKENS_FOR_SPLIT:
        return [
            Chunk(
                index=0,
                text=text,
                char_start=0,
                char_end=len(text),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        ]

    chunker = semchunk.chunkerify(_count_tokens, chunk_size)
    raw_chunks: list[str] = chunker(text)

    chunks: list[Chunk] = []
    cursor = 0
    for i, chunk_text_str in enumerate(raw_chunks):
        start = text.find(chunk_text_str, cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk_text_str)
        chunks.append(
            Chunk(
                index=i,
                text=chunk_text_str,
                char_start=start,
                char_end=end,
                sha256=hashlib.sha256(chunk_text_str.encode()).hexdigest(),
            )
        )
        cursor = end

    return chunks
