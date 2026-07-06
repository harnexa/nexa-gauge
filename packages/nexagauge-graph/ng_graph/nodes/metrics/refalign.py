"""Semantic reference-alignment metric node.

Computes embedding-based similarity between model output and reference answer.

Outputs metric rows using the shared MetricResult contract:
- ``refalign_precision``
- ``refalign_recall``
- ``refalign_f1``
- ``refalign_global_similarity``
- ``refalign_score``

There are two coverage modes:

- ``atomic_chunks=False``: compare upstream chunks or sentence-fallback units.
  Precision/recall/F1 are coarse chunk-coverage scores. Matching is many-to-one
  so repeated or overlapping chunks can share a reference chunk, preserving the
  original non-atomic behavior.
- ``atomic_chunks=True``: first split both sides into smaller factual units with
  an LLM, then compare those units. Precision/recall/F1 are atomic claim-coverage
  scores. Matching is one-to-one so each claim can support at most one claim on
  the other side.

``refalign_global_similarity`` is always the simple full-text embedding cosine
similarity between the raw output text and raw reference text.
"""

from __future__ import annotations

import re
from threading import Lock
from typing import Any, Mapping, Optional

import numpy as np
from ng_core.config import config
from ng_core.constants import (
    REFALIGN_F1_METRIC_PASS_THRESHOLD,
    REFALIGN_GLOBAL_SIMILARITY_METRIC_PASS_THRESHOLD,
    REFALIGN_PRECISION_METRIC_PASS_THRESHOLD,
    REFALIGN_RECALL_METRIC_PASS_THRESHOLD,
    REFALIGN_SIMILARITY_THRESHOLD,
)
from ng_core.types import (
    Chunk,
    CostEstimate,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    Refalign,
    RefalignMetrics,
)
from ng_core.utils import _count_tokens, template_static_tokens
from ng_graph.llm.gateway import get_llm
from ng_graph.llm.pricing import cost_usd, get_node_pricing
from ng_graph.log import get_node_logger
from ng_graph.nodes.base import BaseMetricNode
from ng_graph.nodes.metrics.parallel import run_parallel
from ng_graph.nodes.metrics.scoring import verdict_from_score
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

REFERENCE_ALIGN_MAX_WORKERS = int(config.REFERENCE_ALIGN_MAX_WORKERS)

log = get_node_logger("refalign")

_ATOMIC_SYSTEM_PROMPT = (
    "You decompose text into atomic factual/requirement units. "
    "Split into smallest independent statements while preserving meaning. "
    "Do not oversplit on every work or phrase. Do not invent new facts."
)

_ATOMIC_USER_TEMPLATE = (
    "Side: {side}\n"
    "Input units (one per line):\n"
    "{units}\n\n"
    "Return only atomic units as short plain statements."
)


class _AtomicUnitsResponse(BaseModel):
    units: list[str] = Field(default_factory=list)


_model: SentenceTransformer | None = None
_model_lock = Lock()


def _get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    out: list[str] = []
    for part in raw:
        unit = part.strip()
        if unit:
            out.append(unit)
    return out


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _cosine_matrix(output_embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
    if output_embeddings.size == 0 or reference_embeddings.size == 0:
        return np.zeros((len(output_embeddings), len(reference_embeddings)), dtype=float)

    out_norm = output_embeddings / (np.linalg.norm(output_embeddings, axis=1, keepdims=True) + 1e-9)
    ref_norm = reference_embeddings / (
        np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-9
    )
    return out_norm @ ref_norm.T


def _many_to_one_alignment(
    matrix: np.ndarray,
    output_units: list[str],
    reference_units: list[str],
    similarity_threshold: float,
) -> dict[str, Any]:
    """Align each output unit to its best reference unit, allowing reuse.

    This is the original RefAlign coverage behavior. Every output unit is judged
    independently: if its best reference similarity clears the threshold, that
    output is counted as supported. Separately, each reference unit is counted as
    covered if any output unit clears the threshold against it.

    Example:
    - references: ``["A", "B"]``
    - outputs: ``["A1", "A2", "B1"]``
    - all three outputs clear the threshold against one of the two references

    Result: ``supported_output=3`` and ``covered_reference=2``. This is useful
    for loose support checks, but it allows several output units to reuse the
    same reference unit.
    """
    matched_pairs: list[dict[str, Any]] = []
    extra_output_chunks: list[dict[str, Any]] = []
    for out_idx, out_unit in enumerate(output_units):
        row = matrix[out_idx] if matrix.shape[1] else np.asarray([])
        ref_idx = int(np.argmax(row)) if row.size else -1
        best = float(row[ref_idx]) if row.size else 0.0
        if best >= similarity_threshold:
            matched_pairs.append(
                _matched_pair(out_idx, ref_idx, best, output_units, reference_units)
            )
        else:
            extra_output_chunks.append(_extra_output(out_idx, out_unit, best))

    missed_reference_chunks = [
        _missed_reference(
            ref_idx, ref_unit, float(np.max(matrix[:, ref_idx])) if matrix.shape[0] else 0.0
        )
        for ref_idx, ref_unit in enumerate(reference_units)
        if (float(np.max(matrix[:, ref_idx])) if matrix.shape[0] else 0.0) < similarity_threshold
    ]
    return _alignment_payload(
        supported_output=len(matched_pairs),
        covered_reference=len(reference_units) - len(missed_reference_chunks),
        matched_pairs=matched_pairs,
        extra_output_chunks=extra_output_chunks,
        missed_reference_chunks=missed_reference_chunks,
    )


def _one_to_one_alignment(
    matrix: np.ndarray,
    output_units: list[str],
    reference_units: list[str],
    similarity_threshold: float,
) -> dict[str, Any]:
    """Align output and reference units with one use per side.

    This is stricter than many-to-one alignment. It keeps only threshold-clearing
    pairs and finds a maximum-cardinality bipartite match, so each output unit
    and each reference unit appears at most once. Candidate references are tried
    from highest to lowest similarity for stable, readable pair choices.

    Example:
    - references: ``["A"]``
    - outputs: ``["A restated", "A repeated"]``
    - both outputs clear the threshold against the same reference

    Result: one matched pair, ``supported_output=1`` and
    ``covered_reference=1``. Precision becomes ``1 / 2`` because the repeated
    output claim cannot reuse the only reference claim.
    """
    ranked_refs = [
        [
            int(ref_idx)
            for ref_idx in np.argsort(matrix[out_idx])[::-1]
            if float(matrix[out_idx, ref_idx]) >= similarity_threshold
        ]
        for out_idx in range(matrix.shape[0])
    ]
    output_for_ref: dict[int, int] = {}

    def assign(out_idx: int, seen_refs: set[int]) -> bool:
        for ref_idx in ranked_refs[out_idx]:
            if ref_idx in seen_refs:
                continue
            seen_refs.add(ref_idx)
            if ref_idx not in output_for_ref or assign(output_for_ref[ref_idx], seen_refs):
                output_for_ref[ref_idx] = out_idx
                return True
        return False

    for out_idx in range(len(ranked_refs)):
        assign(out_idx, set())

    used_outputs = set(output_for_ref.values())
    used_references = set(output_for_ref)
    matched_pairs = [
        _matched_pair(
            out_idx, ref_idx, float(matrix[out_idx, ref_idx]), output_units, reference_units
        )
        for ref_idx, out_idx in sorted(output_for_ref.items(), key=lambda item: item[1])
    ]
    extra_output_chunks = [
        _extra_output(out_idx, out_unit, float(np.max(matrix[out_idx])) if matrix.shape[1] else 0.0)
        for out_idx, out_unit in enumerate(output_units)
        if out_idx not in used_outputs
    ]
    missed_reference_chunks = [
        _missed_reference(
            ref_idx, ref_unit, float(np.max(matrix[:, ref_idx])) if matrix.shape[0] else 0.0
        )
        for ref_idx, ref_unit in enumerate(reference_units)
        if ref_idx not in used_references
    ]
    return _alignment_payload(
        supported_output=len(matched_pairs),
        covered_reference=len(matched_pairs),
        matched_pairs=matched_pairs,
        extra_output_chunks=extra_output_chunks,
        missed_reference_chunks=missed_reference_chunks,
    )


def _matched_pair(
    out_idx: int,
    ref_idx: int,
    similarity: float,
    output_units: list[str],
    reference_units: list[str],
) -> dict[str, Any]:
    return {
        "output_index": out_idx,
        "reference_index": ref_idx,
        "similarity": round(similarity, 4),
        "output_text": output_units[out_idx],
        "reference_text": reference_units[ref_idx],
    }


def _extra_output(out_idx: int, output_text: str, best_similarity: float) -> dict[str, Any]:
    return {
        "output_index": out_idx,
        "output_text": output_text,
        "best_similarity": round(best_similarity, 4),
    }


def _missed_reference(ref_idx: int, reference_text: str, best_similarity: float) -> dict[str, Any]:
    return {
        "reference_index": ref_idx,
        "reference_text": reference_text,
        "best_similarity": round(best_similarity, 4),
    }


def _alignment_payload(
    *,
    supported_output: int,
    covered_reference: int,
    matched_pairs: list[dict[str, Any]],
    extra_output_chunks: list[dict[str, Any]],
    missed_reference_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "supported_output": supported_output,
        "covered_reference": covered_reference,
        "matched_pairs": matched_pairs,
        "extra_output_chunks": extra_output_chunks,
        "missed_reference_chunks": missed_reference_chunks,
    }


class RefalignNode(BaseMetricNode):
    node_name = "refalign"

    @staticmethod
    def _metric_result(
        name: str, score: float, threshold: float, payload: dict[str, Any]
    ) -> MetricResult:
        rounded_score = round(_clamp_score(score), 4)
        return MetricResult(
            name=name,
            category=MetricCategory.ANSWER,
            score=rounded_score,
            verdict=verdict_from_score(rounded_score, threshold),
            result=[payload],
        )

    def _response_cost(self, usage: Mapping[str, Any], model: str) -> CostEstimate:
        prompt_tokens = float(usage.get("prompt_tokens", 0.0) or 0.0)
        completion_tokens = float(usage.get("completion_tokens", 0.0) or 0.0)
        pricing = get_node_pricing(
            node_name=self.node_name,
            model=model,
            llm_overrides=self.llm_overrides,
        )
        cost = cost_usd(prompt_tokens, pricing, "input") + cost_usd(
            completion_tokens, pricing, "output"
        )
        return CostEstimate(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=cost,
        )

    def _atomic_units(
        self,
        *,
        side: str,
        units: list[str],
        record_usage: bool = True,
    ) -> tuple[list[str], CostEstimate, Mapping[str, Any] | None]:
        if not units:
            return [], CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0), None

        numbered = "\n".join(f"{idx + 1}. {unit}" for idx, unit in enumerate(units))
        user_prompt = _ATOMIC_USER_TEMPLATE.format(side=side, units=numbered)
        llm = get_llm(
            "refalign",
            _AtomicUnitsResponse,
            self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        response = llm.invoke(
            [
                {"role": "system", "content": _ATOMIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        if record_usage:
            self._record_model_response(response, primary_model=self.judge_model)

        parsed: BaseModel | None = response.get("parsed")
        if parsed is None:
            log.warning(f"Atomic extraction parse failed for {side}; using original units")
            return (
                units,
                self._response_cost(
                    response.get("usage", {}), response.get("model", "") or self.judge_model
                ),
                response,
            )

        extracted = [
            u.strip() for u in getattr(parsed, "units", []) if isinstance(u, str) and u.strip()
        ]
        if not extracted:
            extracted = units

        # Stable de-dup preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for unit in extracted:
            if unit in seen:
                continue
            seen.add(unit)
            deduped.append(unit)

        return (
            deduped,
            self._response_cost(
                response.get("usage", {}), response.get("model", "") or self.judge_model
            ),
            response,
        )

    def _apply_atomic_chunks(
        self,
        *,
        output_units: list[str],
        reference_units: list[str],
    ) -> tuple[list[str], list[str], CostEstimate]:
        atomic_jobs = [("output", output_units), ("reference", reference_units)]

        def _atomic_worker(
            job: tuple[str, list[str]],
        ) -> tuple[str, list[str], CostEstimate, Mapping[str, Any] | None]:
            side, side_units = job
            extracted_units, side_cost, response = self._atomic_units(
                side=side,
                units=side_units,
                record_usage=False,
            )
            return side, extracted_units, side_cost, response

        atomic_results = run_parallel(
            atomic_jobs, _atomic_worker, max_workers=REFERENCE_ALIGN_MAX_WORKERS
        )

        output_cost = CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0)
        reference_cost = CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0)
        for side, extracted_units, side_cost, response in atomic_results:
            if response is not None:
                self._record_model_response(response, primary_model=self.judge_model)
            if side == "output":
                output_units = extracted_units
                output_cost = side_cost
            else:
                reference_units = extracted_units
                reference_cost = side_cost

        total_cost = CostEstimate(
            input_tokens=float(output_cost.input_tokens or 0.0)
            + float(reference_cost.input_tokens or 0.0),
            output_tokens=float(output_cost.output_tokens or 0.0)
            + float(reference_cost.output_tokens or 0.0),
            cost=float(output_cost.cost or 0.0) + float(reference_cost.cost or 0.0),
        )
        return output_units, reference_units, total_cost

    def _compute_embedding_metrics(
        self,
        *,
        output_text: str,
        reference_text: str,
        output_units: list[str],
        reference_units: list[str],
        similarity_threshold: float,
        atomic_chunks: bool,
    ) -> list[MetricResult]:
        model = _get_embedding_model()

        global_similarity = 0.0
        if output_text.strip() and reference_text.strip():
            global_embeddings = model.encode([output_text, reference_text], show_progress_bar=False)
            global_similarity = _clamp_score(
                _cosine_similarity(
                    np.asarray(global_embeddings[0]), np.asarray(global_embeddings[1])
                )
            )

        output_embeddings = (
            np.asarray(model.encode(output_units, show_progress_bar=False))
            if output_units
            else np.asarray([])
        )
        reference_embeddings = (
            np.asarray(model.encode(reference_units, show_progress_bar=False))
            if reference_units
            else np.asarray([])
        )

        matrix = _cosine_matrix(output_embeddings, reference_embeddings)
        max_similarity = round(float(np.max(matrix)) if matrix.size else 0.0, 5)

        alignment_strategy = "one_to_one" if atomic_chunks else "many_to_one"
        alignment = (
            _one_to_one_alignment(matrix, output_units, reference_units, similarity_threshold)
            if atomic_chunks
            else _many_to_one_alignment(matrix, output_units, reference_units, similarity_threshold)
        )
        supported_output = int(alignment["supported_output"])
        covered_reference = int(alignment["covered_reference"])

        precision = round((supported_output / len(output_units)) if output_units else 0.0, 5)
        recall = round((covered_reference / len(reference_units)) if reference_units else 0.0, 5)
        f1 = round(
            (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0, 5
        )

        shared_payload: dict[str, Any] = {
            "atomic_chunks": atomic_chunks,
            "alignment_strategy": alignment_strategy,
            "similarity_threshold": round(similarity_threshold, 4),
            "counts": {
                "supported_output_chunks": supported_output,
                "output_chunks": len(output_units),
                "covered_reference_chunks": covered_reference,
                "reference_chunks": len(reference_units),
            },
            "matched_pairs": alignment["matched_pairs"],
            "missed_reference_chunks": alignment["missed_reference_chunks"],
            "extra_output_chunks": alignment["extra_output_chunks"],
        }

        return [
            self._metric_result(
                "refalign_precision",
                precision,
                REFALIGN_PRECISION_METRIC_PASS_THRESHOLD,
                {"metric": "precision"},
            ),
            self._metric_result(
                "refalign_recall",
                recall,
                REFALIGN_RECALL_METRIC_PASS_THRESHOLD,
                {"metric": "recall"},
            ),
            self._metric_result(
                "refalign_f1",
                f1,
                REFALIGN_F1_METRIC_PASS_THRESHOLD,
                {"metric": "f1"},
            ),
            self._metric_result(
                "refalign_global_similarity",
                global_similarity,
                REFALIGN_GLOBAL_SIMILARITY_METRIC_PASS_THRESHOLD,
                {"metric": "global_similarity"},
            ),
            self._metric_result(
                "refalign_score",
                max_similarity,
                REFALIGN_SIMILARITY_THRESHOLD,
                {**shared_payload, "metric": "max_score"},
            ),
        ]

    def run(  # type: ignore[override]
        self,
        output: Item | str,
        reference: Item | str | None,
        output_chunks: list[Chunk] | None = None,
        reference_chunks: list[Chunk] | None = None,
        refalign: Optional[Refalign] = None,
        enable_output_metrics: bool = True,
    ) -> RefalignMetrics:
        """Compute semantic reference-alignment scores for one case.

        Inputs:
        - ``output``: model answer text (``Item`` or ``str``).
        - ``reference``: expected answer text (``Item`` or ``str``). If empty,
          the node returns an empty metric list.
        - ``output_chunks``: optional refined output chunks from upstream nodes.
          When provided, these are the primary output units for coverage scoring.
        - ``reference_chunks``: optional refined reference chunks from upstream
          nodes. When provided, these are the primary reference units.
        - ``refalign``: optional per-case config:
          - ``atomic_chunks``: when ``True``, run LLM-assisted atomic unit
            extraction on both output and reference unit lists.
          - ``similarity_threshold``: pass threshold for coverage checks.
        - ``enable_output_metrics``: global node gate; when ``False``, returns
          empty metrics.

        Processing flow:
        1. Normalize ``output``/``reference`` to plain text and apply skip gates.
        2. Build output/reference unit lists:
           - output units from ``output_chunks`` when available,
           - reference units from ``reference_chunks`` when available,
           - otherwise sentence splitting fallback,
           - final fallback to one full-text unit.
        3. If ``atomic_chunks=True``, replace both unit lists using LLM atomic
           extraction (with de-dup and fallback to original units on parse/empty).
        4. Embed full texts for ``global_similarity``.
        5. Embed unit lists and build cosine similarity matrix.
        6. Derive coverage stats:
           - ``atomic_chunks=False``: chunk/sentence-level coverage with
             many-to-one matching. A coarse output chunk is supported when its
             best reference chunk clears the threshold. A reference chunk is
             covered when any output chunk clears the threshold against it.
           - ``atomic_chunks=True``: atomic claim-level coverage with one-to-one
             matching. Each extracted output claim and reference claim can be
             used in at most one matched pair.
           - precision (supported output units / output units),
           - recall (covered reference units / reference units),
           - f1 (harmonic mean),
           - max similarity across the matrix,
           plus explainability artifacts (matched pairs, missed reference units,
           extra output units).
        7. Emit metric rows with verdicts from threshold constants.

        Output contract:
        - Returns ``RefalignMetrics(metrics, cost)``.
        - ``metrics`` contains:
          - ``refalign_precision``
          - ``refalign_recall``
          - ``refalign_f1``
          - ``refalign_global_similarity``
          - ``refalign_score`` (max pairwise similarity)
        - ``cost`` is non-zero only when atomic extraction triggers LLM calls;
          pure embedding path is zero-cost.
        """
        self._reset_model_usage()
        zero_cost = CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
        if not enable_output_metrics:
            return RefalignMetrics(metrics=[], cost=zero_cost)

        output_text = output.text if isinstance(output, Item) else (output or "")
        if isinstance(reference, Item):
            reference_text = reference.text
        else:
            reference_text = reference or ""

        if not reference_text.strip():
            log.info("No reference provided - skipping refalign metrics")
            return RefalignMetrics(metrics=[], cost=zero_cost)

        cfg = refalign or Refalign()
        similarity_threshold = _clamp_score(float(cfg.similarity_threshold))

        # Fetch all output units
        output_units: list[str] = []
        if output_chunks:
            output_units = [
                chunk.item.text.strip()
                for chunk in output_chunks
                if chunk.item and chunk.item.text.strip()
            ]
        if not output_units:
            output_units = _split_sentences(output_text)
        if not output_units and output_text.strip():
            output_units = [output_text.strip()]

        # Fetch all reference units
        reference_units: list[str] = []
        if reference_chunks:
            reference_units = [
                chunk.item.text.strip()
                for chunk in reference_chunks
                if chunk.item and chunk.item.text.strip()
            ]
        if not reference_units:
            reference_units = _split_sentences(reference_text)
        if not reference_units and reference_text.strip():
            reference_units = [reference_text.strip()]

        total_cost = CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0)

        if cfg.atomic_chunks:
            output_units, reference_units, total_cost = self._apply_atomic_chunks(
                output_units=output_units,
                reference_units=reference_units,
            )

        metrics = self._compute_embedding_metrics(
            output_text=output_text,
            reference_text=reference_text,
            output_units=output_units,
            reference_units=reference_units,
            similarity_threshold=similarity_threshold,
            atomic_chunks=bool(cfg.atomic_chunks),
        )

        # Keep the node-level cost contract consistent with other metric nodes.
        final_cost = CostEstimate(
            cost=round(float(total_cost.cost or 0.0), 8),
            input_tokens=(
                float(total_cost.input_tokens)
                if total_cost.input_tokens is not None and total_cost.input_tokens > 0
                else None
            ),
            output_tokens=(
                float(total_cost.output_tokens)
                if total_cost.output_tokens is not None and total_cost.output_tokens > 0
                else None
            ),
        )

        return RefalignMetrics(metrics=metrics, cost=final_cost)

    def estimate(self, inputs: Inputs) -> CostEstimate:  # type: ignore[override]
        self._reset_model_usage()
        if not inputs.has_reference or not inputs.has_output:
            return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)

        cfg = inputs.refalign or Refalign()
        if not cfg.atomic_chunks:
            return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)

        output_item = inputs.output
        reference_item = inputs.reference
        if reference_item is None:
            return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)

        static_tokens = _count_tokens(_ATOMIC_SYSTEM_PROMPT) + template_static_tokens(
            _ATOMIC_USER_TEMPLATE.format(side="{side}", units="{units}")
        )
        input_tokens = (
            (2 * static_tokens) + float(output_item.tokens) + float(reference_item.tokens)
        )
        output_tokens = 120.0

        pricing = get_node_pricing(
            node_name=self.node_name,
            model=self.judge_model,
            llm_overrides=self.llm_overrides,
        )
        estimated_cost = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens,
            pricing,
            "output",
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimated_cost,
        )
