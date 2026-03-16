"""
MMR Deduplicator — removes semantically redundant claims using Maximal Marginal Relevance.

Uses local sentence-transformers embeddings (all-MiniLM-L6-v2 by default) — no API call.
λ=0.5 weights relevance and diversity equally. Claims with cosine similarity > 0.9 to an
already-selected claim are discarded.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from lumiseval_core.config import config
from lumiseval_core.types import Claim

_SIMILARITY_THRESHOLD = 0.9
_LAMBDA = 0.5

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def deduplicate(
    claims: list[Claim],
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    lmb: float = _LAMBDA,
) -> tuple[list[Claim], dict[int, int]]:
    """Deduplicate claims using MMR.

    Returns:
        unique_claims: Claims that survived deduplication.
        dedup_map: Mapping from discarded claim index → retained representative index.
    """
    if len(claims) < 2:
        return claims, {}

    model = _get_model()
    texts = [c.text for c in claims]
    embeddings = model.encode(texts, show_progress_bar=False)  # shape: (N, D)

    selected_indices: list[int] = []
    candidate_indices = list(range(len(claims)))
    dedup_map: dict[int, int] = {}

    # Start by selecting the first claim (highest confidence)
    best_start = max(candidate_indices, key=lambda i: claims[i].confidence)
    selected_indices.append(best_start)
    candidate_indices.remove(best_start)

    while candidate_indices:
        scores: list[tuple[int, float]] = []
        for ci in candidate_indices:
            # Relevance: original claim confidence
            relevance = claims[ci].confidence
            # Redundancy: max similarity to any already-selected claim
            max_sim = max(
                _cosine_similarity(embeddings[ci], embeddings[si]) for si in selected_indices
            )
            if max_sim >= similarity_threshold:
                # Mark as duplicate of the most similar selected claim
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

    unique_claims = [claims[i] for i in selected_indices]
    return unique_claims, dedup_map
