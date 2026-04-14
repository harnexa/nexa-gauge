"""
MMR deduplication for extracted claims.

This module is intentionally placed in ``lumos-core`` so the underlying
algorithm can be reused by multiple orchestration layers (graph runner, future
API workers, benchmarking tools) without depending on ``lumos-evidence``
package boundaries.
"""

import numpy as np
from lumoseval_core.config import config
from lumoseval_core.constants import MMR_LAMBDA, MMR_SIMILARITY_THRESHOLD
from lumoseval_core.types import Item
from sentence_transformers import SentenceTransformer

_SIMILARITY_THRESHOLD = MMR_SIMILARITY_THRESHOLD
_LAMBDA = MMR_LAMBDA

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily initialize and reuse the local embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def deduplicate(
    items: list[Item],
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    lmb: float = _LAMBDA,
) -> tuple[list[Item], dict[int, int]]:
    """Deduplicate claims using Maximal Marginal Relevance.

    Args:
        claims: Ordered claims to deduplicate.
        similarity_threshold: Candidate-to-selected cosine threshold above which
            a candidate is treated as duplicate.
        lmb: MMR lambda balancing relevance (claim confidence) and diversity.

    Returns:
        unique_claims: Claims that survive deduplication.
        dedup_map: Mapping from discarded claim index -> retained representative index.
    """
    if len(items) < 2:
        return items, {}

    model = _get_model()
    texts = [c.text for c in items]
    embeddings = model.encode(texts, show_progress_bar=False)  # shape: (N, D)

    selected_indices: list[int] = []
    candidate_indices = list(range(len(items)))
    dedup_map: dict[int, int] = {}

    # Start with the highest-confidence claim.
    best_start = max(candidate_indices, key=lambda i: items[i].confidence)
    selected_indices.append(best_start)
    candidate_indices.remove(best_start)

    while candidate_indices:
        scores: list[tuple[int, float]] = []
        for ci in candidate_indices:
            relevance = items[ci].confidence
            max_sim = max(
                _cosine_similarity(embeddings[ci], embeddings[si]) for si in selected_indices
            )
            if max_sim >= similarity_threshold:
                nearest = max(
                    selected_indices,
                    key=lambda si: _cosine_similarity(embeddings[ci], embeddings[si]),
                )
                dedup_map[ci] = nearest
                continue
            mmr_score = lmb * relevance - (1 - lmb) * max_sim
            scores.append((ci, mmr_score))

        if not scores:
            break

        best_ci = max(scores, key=lambda x: x[1])[0]
        selected_indices.append(best_ci)
        candidate_indices = [c for c in candidate_indices if c not in dedup_map and c != best_ci]

    unique_items: list[Item] = [items[i] for i in selected_indices]
    return unique_items, dedup_map
