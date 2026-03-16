"""
RAGAS Node — computes RAG triad metrics (faithfulness, answer relevancy,
context precision, context recall) using the ragas library.

Only activated when retrieved context passages are available.
Skipped gracefully if no context is found (scores would be meaningless).

TODO: Wire LiteLLM as the judge LLM inside RAGAS so billing is unified.
"""

import logging
from typing import Optional

from lumiseval_core.types import EvidenceResult, RAGASMetricResult

logger = logging.getLogger(__name__)


def run(
    generation: str,
    evidence_results: list[EvidenceResult],
    question: Optional[str] = None,
    ground_truth: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
) -> RAGASMetricResult:
    """Compute RAGAS metrics on the generation + retrieved evidence.

    Args:
        generation: The LLM-generated text to evaluate.
        evidence_results: Evidence passages retrieved by the Evidence Router.
        question: The original query/question (improves answer relevancy score).
        ground_truth: Optional reference answer (enables context_recall).
        judge_model: LiteLLM model string used as the RAGAS judge LLM.

    Returns:
        RAGASMetricResult with available metric scores. Missing inputs → None scores.
    """
    passages = [p.text for er in evidence_results for p in er.passages]
    if not passages:
        logger.info("RAGAS skipped: no retrieved context passages available.")
        return RAGASMetricResult()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy

        metrics = [Faithfulness(), AnswerRelevancy()]

        data = {
            "question": [question or ""],
            "answer": [generation],
            "contexts": [passages],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)
        scores = result.to_pandas().iloc[0].to_dict()

        return RAGASMetricResult(
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
        )
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return RAGASMetricResult(error=str(exc))
