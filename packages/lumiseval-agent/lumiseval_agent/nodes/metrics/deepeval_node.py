"""
DeepEval Node — computes hallucination, G-Eval, Privacy, and Bias metrics.

Runs metrics programmatically (not via pytest). G-Eval is only invoked when a rubric
is provided. Privacy and Bias metrics run only when adversarial=True.

TODO: Batch metric execution to reduce redundant LLM calls.
"""

from typing import Optional

from lumiseval_core.types import DeepEvalMetricResult, EvidenceResult, RubricRule
from lumiseval_agent.log import get_node_logger
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import HallucinationMetric

log = get_node_logger("deepeval")


def run(
    generation: str,
    evidence_results: list[EvidenceResult],
    rubric_rules: Optional[list[RubricRule]] = None,
    adversarial: bool = False,
    judge_model: str = "gpt-4o-mini",
) -> DeepEvalMetricResult:
    """Compute DeepEval metrics.

    Args:
        generation: The LLM-generated text to evaluate.
        evidence_results: Evidence passages retrieved by the Evidence Router.
        rubric_rules: Optional rubric rules for G-Eval (rubric evaluation mode).
        adversarial: Whether to run Privacy and Bias safety metrics.
        judge_model: LiteLLM model string for the judge LLM.

    Returns:
        DeepEvalMetricResult with available scores.
    """
    context = [p.text for er in evidence_results for p in er.passages]
    log.info(f"Context passages available: {len(context)}")
    result = DeepEvalMetricResult()

    log.info("Running HallucinationMetric")
    # try:


    test_case = LLMTestCase(
        input="",
        actual_output=generation,
        context=context or None,
    )

    hallucination = HallucinationMetric(model=judge_model)
    hallucination.measure(test_case)
    result.hallucination_score = hallucination.score
    log.info(f"  hallucination_score={hallucination.score}")
    # except Exception as exc:
    #     log.error(f"HallucinationMetric failed: {exc}")
    #     result.error = str(exc)

    if rubric_rules:
        log.info(f"Running GEval against {len(rubric_rules)} rubric rule(s)")
        # try:
        criteria = "\n".join(f"- {r.statement}" for r in rubric_rules)
        test_case = LLMTestCase(input="", actual_output=generation)
        g_eval = GEval(
            name="rubric_adherence",
            criteria=criteria,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge_model,
        )
        g_eval.measure(test_case)
        result.g_eval_score = g_eval.score
        log.info(f"  g_eval_score={g_eval.score}")
        # except Exception as exc:
        #     log.error(f"GEval failed: {exc}")

    if adversarial:
        log.info("Running Privacy + Bias metrics  (adversarial=True)")
        # try:
        from deepeval.metrics import BiasMetric, PrivacyMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(input="", actual_output=generation)
        privacy = PrivacyMetric(model=judge_model)
        bias = BiasMetric(model=judge_model)
        privacy.measure(test_case)
        bias.measure(test_case)
        result.privacy_score = privacy.score
        result.bias_score = bias.score
        log.info(f"  privacy_score={privacy.score}  bias_score={bias.score}")
        # except Exception as exc:
        #     log.error(f"Safety metrics failed: {exc}")

    return result
