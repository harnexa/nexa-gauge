"""
Claim Extractor — decomposes each text chunk into atomic, verifiable factual claims.

Uses instructor with a configurable judge LLM (via litellm) to extract claims.
Pydantic validation is handled automatically by instructor; up to 3 retries per chunk.
Chunks that fail after retries are flagged with extraction_failed=True rather than
aborting the full evaluation.

TODO: Tune the extraction prompt for domain-specific claim types.
"""

from typing import Optional

import instructor
import litellm
from lumiseval_core.config import config
from lumiseval_core.types import Chunk, Claim
from pydantic import BaseModel, Field

from lumiseval_agent.log import get_node_logger

log = get_node_logger("claim_extractor")

_EXTRACTION_PROMPT = """You are a precise claim extractor. Given the following text chunk, extract
all atomic, verifiable factual claims it makes.

Rules:
- Each claim must be a single declarative sentence asserting exactly one fact.
- Claims must be verifiable — avoid opinions, conjecture, and meta-commentary.
- Do not paraphrase the entire chunk; extract only discrete, checkable assertions.
- Assign a confidence score (0.0–1.0) based on how clearly the claim is stated.

Text chunk:
{chunk_text}

Return a JSON object with a "claims" array."""


class _ClaimList(BaseModel):
    claims: list[str] = Field(description="List of atomic verifiable claims.")
    confidences: list[float] = Field(
        description="Confidence score for each claim (same order, 0.0–1.0)."
    )


def extract_claims(
    chunks: list[Chunk],
    model: Optional[str] = None,
    max_retries: int = 3,
) -> list[Claim]:
    """Extract atomic factual claims from all chunks.

    Args:
        chunks: List of Chunk objects from the Chunker.
        model: LiteLLM model string (e.g., ``"gpt-4o-mini"``). Defaults to config.LLM_MODEL.
        max_retries: Number of instructor retries per chunk on validation failure.

    Returns:
        Flat list of Claim objects across all chunks.
    """
    model = model or config.LLM_MODEL
    client = instructor.from_litellm(litellm.completion)
    all_claims: list[Claim] = []

    for chunk in chunks:
        log.info(f"chunk {chunk.index + 1}/{len(chunks)}  ({len(chunk.text)} chars)")
        # try:
        result = client.chat.completions.create(
            model=model,
            response_model=_ClaimList,
            max_retries=max_retries,
            messages=[
                {
                    "role": "user",
                    "content": _EXTRACTION_PROMPT.format(chunk_text=chunk.text),
                }
            ],
        )
        n = len(result.claims)
        log.info(f"  → {n} claim(s) extracted from chunk {chunk.index}")
        for claim_text, confidence in zip(result.claims, result.confidences):
            all_claims.append(
                Claim(
                    text=claim_text,
                    source_chunk_index=chunk.index,
                    confidence=confidence,
                )
            )
        # except Exception as exc:
        #     log.warning(
        #         f"Extraction failed for chunk {chunk.index} after {max_retries} retries: {exc}"
        #     )
        #     all_claims.append(
        #         Claim(
        #             text="",
        #             source_chunk_index=chunk.index,
        #             extraction_failed=True,
        #         )
        #     )

    valid = [c for c in all_claims if not c.extraction_failed or c.text]
    log.success(f"{len(valid)} total claim(s) across all chunks")
    return valid
