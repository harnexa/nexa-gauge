"""
Evidence Router — finds the best evidence for a claim using a prioritized source cascade.

Priority order:
  1. Local LanceDB (documents indexed from user-provided files)
  2. MCP LanceDB (user-provided external database, if configured)
  3. Tavily web search (last resort, opt-in)

Retrieved web documents are immediately indexed into local LanceDB so subsequent claims
can reuse them without a second API call.

TODO: Implement MCP LanceDB retrieval (requires MCP connection protocol).
"""

import logging
from typing import Any

import lancedb
from lumiseval_core.config import config
from lumiseval_core.constants import (
    EVIDENCE_RETRIEVAL_TOP_K,
    EVIDENCE_TAVILY_MAX_RESULTS,
    EVIDENCE_VERDICT_SUPPORTED_THRESHOLD,
    EVIDENCE_VERDICT_UNVERIFIABLE_THRESHOLD,
)
from lumiseval_core.types import (
    Claim,
    ClaimVerdict,
    EvidencePassage,
    EvidenceResult,
    EvidenceSource,
)
from sentence_transformers import SentenceTransformer

from .indexer import index_texts

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _query_lancedb(
    query_text: str,
    db_path: str,
    table_name: str = "documents",
    top_k: int = EVIDENCE_RETRIEVAL_TOP_K,
) -> list[dict[str, Any]]:
    # try:
    db = lancedb.connect(db_path)
    if table_name not in db.table_names():
        return []
    table = db.open_table(table_name)
    model = _get_model()
    embedding = model.encode([query_text], show_progress_bar=False)[0].tolist()
    results = table.search(embedding).limit(top_k).to_list()
    return results
    # except Exception as exc:
    #     logger.warning("LanceDB query failed: %s", exc)
    #     return []


def _tavily_search(query: str) -> list[dict[str, Any]]:
    """Execute a Tavily search and return raw result dicts."""
    # try:
    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query, max_results=EVIDENCE_TAVILY_MAX_RESULTS)
    return response.get("results", [])
    # except Exception as exc:
    #     logger.error("Tavily search failed for query '%s': %s", query, exc)
    #     return []


def _results_to_passages(
    results: list[dict[str, Any]], source: EvidenceSource
) -> list[EvidencePassage]:
    passages = []
    for r in results:
        score = r.get("_distance", None)
        if score is None:
            score = r.get("score", 0.0)
        else:
            # LanceDB returns L2 distance; convert to similarity heuristic
            score = max(0.0, 1.0 - score)
        passages.append(
            EvidencePassage(
                text=r.get("text", r.get("content", "")),
                source_doc_id=r.get("id", r.get("url", "unknown")),
                retrieval_score=score,
                source=source,
            )
        )
    return passages


def route(
    claim: Claim,
    db_path: str | None = None,
    mcp_uri: str | None = None,
    web_search: bool | None = None,
    threshold: float | None = None,
) -> EvidenceResult:
    """Find the best evidence for ``claim`` using the cheapest source first.

    Args:
        claim: The claim to verify.
        db_path: Local LanceDB path. Defaults to config.LANCEDB_PATH.
        mcp_uri: Optional MCP LanceDB URI. Defaults to config.LANCEDB_MCP_URI.
        web_search: Whether to allow Tavily fallback. Defaults to config.WEB_SEARCH_ENABLED.
        threshold: Minimum retrieval score to accept. Defaults to config.EVIDENCE_THRESHOLD.
    """
    db_path = db_path or config.LANCEDB_PATH
    mcp_uri = mcp_uri or config.LANCEDB_MCP_URI
    web_search = web_search if web_search is not None else config.WEB_SEARCH_ENABLED
    threshold = threshold if threshold is not None else config.EVIDENCE_THRESHOLD

    # Step 1: Local LanceDB
    local_results = _query_lancedb(claim.text, db_path)
    local_passages = _results_to_passages(local_results, EvidenceSource.LOCAL)
    best_local = max((p.retrieval_score for p in local_passages), default=0.0)

    if best_local >= threshold:
        verdict = _score_to_verdict(best_local)
        return EvidenceResult(
            claim_text=claim.text,
            source=EvidenceSource.LOCAL,
            passages=local_passages,
            verdict=verdict,
        )

    # Step 2: MCP LanceDB (stub — not yet implemented)
    if mcp_uri:
        logger.info("MCP LanceDB retrieval not yet implemented; skipping MCP source.")

    # Step 3: Tavily web search
    if web_search and config.TAVILY_API_KEY:
        web_results = _tavily_search(claim.text)
        if web_results:
            texts = [r.get("content", "") for r in web_results]
            ids = [r.get("url", f"web:{i}") for i, r in enumerate(web_results)]
            index_texts(texts, ids, db_path=db_path)

            # Re-query local after indexing
            local_results2 = _query_lancedb(claim.text, db_path)
            web_passages = _results_to_passages(local_results2, EvidenceSource.WEB)
            best_web = max((p.retrieval_score for p in web_passages), default=0.0)
            verdict = _score_to_verdict(best_web)
            return EvidenceResult(
                claim_text=claim.text,
                source=EvidenceSource.WEB,
                passages=web_passages,
                verdict=verdict,
                no_evidence_found=(best_web < threshold),
            )

    return EvidenceResult(
        claim_text=claim.text,
        source=EvidenceSource.NONE,
        passages=[],
        verdict=ClaimVerdict.UNVERIFIABLE,
        no_evidence_found=True,
    )


def _score_to_verdict(score: float) -> ClaimVerdict:
    if score >= EVIDENCE_VERDICT_SUPPORTED_THRESHOLD:
        return ClaimVerdict.SUPPORTED
    if score >= EVIDENCE_VERDICT_UNVERIFIABLE_THRESHOLD:
        return ClaimVerdict.UNVERIFIABLE
    return ClaimVerdict.CONTRADICTED
