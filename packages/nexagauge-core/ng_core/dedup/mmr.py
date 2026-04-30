from threading import Lock

import numpy as np
from ng_core.config import config
from ng_core.constants import MMR_LAMBDA, MMR_SIMILARITY_THRESHOLD, REFINER_TOP_K
from ng_core.types import Item
from sentence_transformers import SentenceTransformer

_SIMILARITY_THRESHOLD = MMR_SIMILARITY_THRESHOLD
_MMR_TOP_K = REFINER_TOP_K
_LAMBDA = MMR_LAMBDA

_model: SentenceTransformer | None = None
_model_init_lock = Lock()


def _get_model() -> SentenceTransformer:
    """Lazily initialize and reuse the local embedding model."""
    global _model
    if _model is None:
        # Double-checked locking to avoid rare concurrent first-load races.
        # When running with async workers, many may call the `_get_model` at the same
        # time and hence the _model may be invoked multiple times
        # We put a lock here, the locak may introduce negligible overhead
        # and may not affect in real world. When the first hit locks the model
        # the other workers wait and can't pass `if _model is None:` as the model will be available
        with _model_init_lock:
            if _model is None:
                _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def deduplicate(
    items: list[Item],
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    top_k: int | None = _MMR_TOP_K,
    lmb: float = _LAMBDA,
) -> tuple[list[int], dict[int, int]]:
    """Deduplicate claims using Maximal Marginal Relevance.

    Args:
        items: Ordered items to deduplicate.
        similarity_threshold: Candidate-to-selected cosine threshold above which
            a candidate is treated as duplicate.
        lmb: MMR lambda balancing relevance (claim confidence) and diversity.
        top_k: Optional maximum number of unique items to retain. Items excluded
            only because this limit is reached are not added to ``dedup_map``.

    Returns:
        selected_indices: Indices of items selected by MMR.
        dedup_map: Mapping from duplicate item index -> retained representative index.
    """
    if top_k is not None and top_k <= 0:
        return [], {}

    if len(items) < 2:
        if top_k is None:
            return list(range(len(items))), {}
        return list(range(min(len(items), top_k))), {}

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

    while candidate_indices and (top_k is None or len(selected_indices) < top_k):
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

    return selected_indices, dedup_map
