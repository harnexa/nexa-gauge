"""
Claim Extractor — decomposes each text chunk into atomic, verifiable factual claims.

Routes through the LLM Gateway so model selection, fallback strategy, and token
tracking are centralised. Per-node model override: LLM_CLAIMS_MODEL.

TODO: Tune the extraction prompt for domain-specific claim types.
"""

from typing import Optional

from lumiseval_core.config import config
from lumiseval_core.types import Chunk, Claim
from pydantic import BaseModel, Field

from lumiseval_agent.llm import get_llm
from lumiseval_agent.log import get_node_logger

log = get_node_logger("claims")

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
) -> list[Claim]:
    """Extract atomic factual claims from all chunks via the LLM Gateway.

    Args:
        chunks: List of Chunk objects from the Chunker.
        model: LiteLLM model string. Defaults to config.LLM_MODEL.
               Overridden at the node level by LLM_CLAIMS_MODEL env var.

    Returns:
        Flat list of Claim objects across all chunks.
    """
    default_model = model or config.LLM_MODEL
    # get_llm returns a cached instance — safe to call in the loop
    structured_llm = get_llm("claims", _ClaimList, default_model=default_model)
    all_claims: list[Claim] = []

    for chunk in chunks:
        log.info(f"chunk {chunk.index + 1}/{len(chunks)}  ({len(chunk.text)} chars)")

        response = structured_llm.invoke(
            [{"role": "user", "content": _EXTRACTION_PROMPT.format(chunk_text=chunk.text)}]
        )

        if response["parsing_error"]:
            raise response["parsing_error"]

        result: _ClaimList = response["parsed"]
        tokens = response["usage"]["total_tokens"]
        n = len(result.claims)
        log.info(
            f"  → {n} claim(s) extracted from chunk {chunk.index}"
            f"  (model={response['model']}  tokens={tokens})"
        )

        for claim_text, confidence in zip(result.claims, result.confidences):
            all_claims.append(
                Claim(
                    text=claim_text,
                    source_chunk_index=chunk.index,
                    confidence=confidence,
                )
            )

    valid = [c for c in all_claims if not c.extraction_failed or c.text]
    log.success(f"{len(valid)} total claim(s) across all chunks")
    return valid
