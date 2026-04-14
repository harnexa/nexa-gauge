"""Claim Extractor Node."""

from typing import Any, Mapping, Optional

from lumoseval_core.constants import AVG_CLAIM_INPUT_TOKENS, DEFAULT_JUDGE_MODEL
from lumoseval_core.types import Chunk, Claim, ClaimArtifacts, CostEstimate, Item
from lumoseval_core.utils import _count_tokens, template_static_tokens
from lumoseval_graph.llm.gateway import get_llm
from lumoseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumoseval_graph.log import get_node_logger
from lumoseval_graph.nodes.base import BaseNode
from pydantic import BaseModel, Field

log = get_node_logger("claims")


class _ClaimList(BaseModel):
    claims: list[str] = Field(description="List of atomic verifiable claims.")
    confidences: list[float] = Field(description="Confidence score for each claim.")


class ClaimExtractorNode(BaseNode):
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
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        llm_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.llm_overrides = llm_overrides

    def run(self, chunks: list[Chunk]) -> ClaimArtifacts:
        self._reset_model_usage()
        llm = get_llm("claims", _ClaimList, self.model, llm_overrides=self.llm_overrides)
        pricing = get_model_pricing(self.model)

        all_claims: list[Claim] = []
        costs: list[CostEstimate] = []

        for chunk in chunks:
            response = llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self.USER_PROMPT.format(chunk_text=chunk.item.text),
                    },
                ]
            )
            self._record_model_response(response, primary_model=self.model)
            if response["parsing_error"]:
                raise response["parsing_error"]

            result: _ClaimList = response["parsed"]
            if result is None:
                continue

            prompt_tokens = float(response["usage"]["prompt_tokens"])
            completion_tokens = float(response["usage"]["completion_tokens"])
            costs.append(
                CostEstimate(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cost=(
                        cost_usd(prompt_tokens, pricing, "input")
                        + cost_usd(completion_tokens, pricing, "output")
                    ),
                )
            )

            for claim_text, confidence in zip(result.claims, result.confidences):
                all_claims.append(
                    Claim(
                        item=Item(text=claim_text, tokens=float(_count_tokens(claim_text))),
                        source_chunk_index=chunk.index,
                        confidence=confidence,
                    )
                )
        cumulative_cost = CostEstimate(
            input_tokens=sum(c.input_tokens or 0.0 for c in costs),
            output_tokens=sum(c.output_tokens or 0.0 for c in costs),
            cost=sum(c.cost for c in costs),
        )
        valid_claims = [c for c in all_claims if (not c.extraction_failed and c.item.text.strip())]
        log.success(f"{len(valid_claims)} total claim(s) across all chunks")
        return ClaimArtifacts(claims=valid_claims, cost=cumulative_cost)

    def estimate(self, chunks: list[Chunk]) -> CostEstimate:
        self._reset_model_usage()

        tokens = sum(c.item.tokens for c in chunks if c.item and c.item.text.strip())
        output_tokens = len(chunks) * AVG_CLAIM_INPUT_TOKENS

        input_tokens = self.static_prompt_tokens + tokens
        pricing = get_model_pricing(self.model)

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost_usd(input_tokens, pricing, "input")
            + cost_usd(output_tokens, pricing, "output"),
        )
