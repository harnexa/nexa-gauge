"""
Rubric Evaluator Node — scores the generation against each RubricRule using DeepEval GEval.

All rules are evaluated in parallel via asyncio.gather to minimize total wall-clock time.
Returns one MetricResult per rule (score = GEval confidence, passed = score >= 0.5).

TODO: Implement Rubric Extractor agent that auto-derives rules from reference documents.
"""

import asyncio

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from lumiseval_core.constants import METRIC_PASS_THRESHOLD
from lumiseval_core.types import MetricCategory, MetricResult, RubricRule

from lumiseval_agent.log import get_node_logger

log = get_node_logger("rubric_eval")


async def _evaluate_rule(
    rule: RubricRule,
    generation: str,
    judge_model: str,
) -> MetricResult:
    criteria = f"{rule.statement}\n\nPass condition: {rule.pass_condition}"
    test_case = LLMTestCase(input="", actual_output=generation)
    g_eval = GEval(
        name=rule.id,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge_model,
    )
    # GEval.measure is sync; run in thread to not block event loop
    await asyncio.get_event_loop().run_in_executor(None, g_eval.measure, test_case)

    score = g_eval.score or 0.0
    passed = score >= METRIC_PASS_THRESHOLD
    log.info(f"  [{'PASS' if passed else 'FAIL'}] {rule.id}  score={score:.3f}")

    return MetricResult(
        name=rule.id,
        category=MetricCategory.ANSWER,
        score=score,
        passed=passed,
        reasoning=g_eval.reason or "",
    )


def run(
    generation: str,
    rubric_rules: list[RubricRule],
    judge_model: str = "gpt-4o-mini",
) -> list[MetricResult]:
    """Evaluate the generation against all rubric rules in parallel.

    Args:
        generation: The LLM-generated text to evaluate.
        rubric_rules: List of RubricRule objects to check.
        judge_model: LiteLLM model string for the judge LLM.

    Returns:
        list[MetricResult] — one per rubric rule, scored 0.0–1.0.
    """
    if not rubric_rules:
        return []

    async def _run_all() -> list[MetricResult]:
        tasks = [_evaluate_rule(rule, generation, judge_model) for rule in rubric_rules]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run_all())
