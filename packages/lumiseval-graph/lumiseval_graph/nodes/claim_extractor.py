# Run smoke test:
#   python -m lumiseval_graph.nodes.claim_extractor
"""
Claim Extractor — decomposes each text chunk into atomic, verifiable factual claims.

Routes through the LLM Gateway so model selection, fallback strategy, and token
tracking are centralised.

TODO: Tune the extraction prompt for domain-specific claim types.
"""

from lumiseval_core.constants import DEFAULT_JUDGE_MODEL
from lumiseval_core.types import Chunk, Claim, ClaimCostMeta, NodeCostBreakdown
from pydantic import BaseModel, Field

from lumiseval_graph.llm.gateway import get_llm
from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.metrics.token_utils import count_tokens, template_static_tokens

log = get_node_logger("claims")


class _ClaimList(BaseModel):
    claims: list[str] = Field(description="List of atomic verifiable claims.")
    confidences: list[float] = Field(
        description="Confidence score for each claim (same order, 0.0–1.0)."
    )


class ClaimExtractorNode:
    """Extracts atomic factual claims from text chunks via the LLM Gateway."""

    node_name = "claims"
    SYSTEM_PROMPT = "You are a precise claim extractor."
    USER_PROMPT = (
        "Given the following text chunk, extract the single most important atomic, verifiable factual claim it makes.\n\n"
        "Rules:\n"
        "- The claim must be a single declarative sentence asserting exactly one fact.\n"
        "- The claim must be verifiable — avoid opinions, conjecture, and meta-commentary.\n"
        "- Choose the claim that best represents the core assertion of the chunk.\n"
        "- Assign a confidence score (0.0–1.0) based on how clearly the claim is stated.\n\n"
        "Text chunk:\n{chunk_text}\n\n"
        "Return a JSON object with a 'claims' array containing exactly one claim "
        "and a 'confidences' array containing exactly one score."
    )

    # Static (non-placeholder) token overhead shared by every call — computed
    # once at class definition time from the prompts above.
    static_prompt_tokens: int = count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL) -> None:
        self.model = model

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model!r}>"

    def run(self, chunks: list[Chunk]) -> list[Claim]:
        """Extract atomic factual claims from all chunks via the LLM Gateway.

        Args:
            chunks: List of Chunk objects from the Chunker.

        Returns:
            Flat list of Claim objects across all chunks.
        """
        llm = get_llm("claims", _ClaimList, self.model)
        all_claims: list[Claim] = []

        for chunk in chunks:
            log.info(f"chunk {chunk.index + 1}/{len(chunks)}  ({len(chunk.text)} chars)")

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.USER_PROMPT.format(chunk_text=chunk.text),
                },
            ]
            response = llm.invoke(messages)

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

    def cost_estimate(
        self,
        *,
        cost_meta: ClaimCostMeta,
        **_ignored,
    ) -> NodeCostBreakdown:
        """Estimate LLM cost for claim extraction without running any calls.

        Args:
            eligible_records:      Records this node will process.
            avg_chunks_per_record: Average number of chunks per record.
            avg_chunk_tokens:      Average token count per chunk (input content).

        Returns:
            NodeCostBreakdown with judge_calls and cost_usd.
        """
        if cost_meta.eligible_records == 0:
            return NodeCostBreakdown(model_calls=0, cost_usd=0.0)

        pricing = get_model_pricing(self.model)
        chunks_per_record = max(1.0, cost_meta.avg_generation_chunks)
        total_calls = round(cost_meta.eligible_records * chunks_per_record)

        input_tokens = self.static_prompt_tokens + cost_meta.avg_generation_tokens
        # One claim per chunk: claim text tokens + confidence float (JSON verdict overhead)
        output_tokens = cost_meta.avg_claim_tokens + cost_meta.avg_output_token

        cost_per_call = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return NodeCostBreakdown(
            model_calls=total_calls,
            cost_usd=round(total_calls * cost_per_call, 6),
        )

    @classmethod
    def cost_formula(cls, cost_meta: ClaimCostMeta) -> str:
        chunks = max(1.0, cost_meta.avg_generation_chunks)
        total_calls = round(cost_meta.eligible_records * chunks)
        prompt_t = cls.static_prompt_tokens
        gen_t = round(cost_meta.avg_generation_tokens)
        input_t = prompt_t + gen_t
        out_claim = round(cost_meta.avg_claim_tokens)
        out_verdict = round(cost_meta.avg_output_token)
        output_t = out_claim + out_verdict
        return (
            f"{cost_meta.eligible_records} recs × {chunks:.1f} chunks/rec = {total_calls} calls\n"
            f"  input_tokens  = {prompt_t} (prompt) + {gen_t} (generation_chunk) = {input_t} tok/call\n"
            f"  output_tokens = {out_claim} (claim_text) + {out_verdict} (json_verdict) = {output_t} tok/call"
        )


# ── Manual smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Real claim extraction test.

    Input: two chunks about transformer architecture.
    Expected: several atomic claims extracted from each chunk with confidence scores.
    """
    from pprint import pprint

    chunks = [
        Chunk(
            index=0,
            text=(
                "The Transformer model introduced in 'Attention Is All You Need' (Vaswani et al., 2017) "
                "relies entirely on a self-attention mechanism to draw global dependencies between input "
                "and output, discarding recurrence and convolutions entirely."
            ),
        ),
        Chunk(
            index=1,
            text=(
                "Because the model contains no recurrence and no convolution, positional encodings are "
                "added to the input embeddings to inject information about the relative or absolute "
                "position of tokens in the sequence."
            ),
        ),
    ]

    node = ClaimExtractorNode(model="gpt-4o-mini")
    print(repr(node))
    claims = node.run(chunks)
    print(f"\n{len(claims)} claims extracted:")
    for c in claims:
        pprint(c)

    print("\nCost estimate:")
    pprint(node.cost_estimate(eligible_records=100, avg_chunks_per_record=2.0))
