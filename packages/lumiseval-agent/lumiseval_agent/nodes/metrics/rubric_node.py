"""
Rubric Evaluator Node — scores the generation against each RubricRule using DeepEval GEval.

All rules are evaluated in parallel via asyncio.gather to minimize total wall-clock time.
If a judge call fails for a rule, that rule is marked UNCERTAIN rather than aborting.

TODO: Implement Rubric Extractor agent that auto-derives rules from reference documents.
"""

import asyncio

from lumiseval_core.types import RubricEvalResult, RubricRule, RubricRuleResult, RuleCompliance

from lumiseval_agent.log import get_node_logger

log = get_node_logger("rubric_eval")


async def _evaluate_rule(
    rule: RubricRule,
    generation: str,
    judge_model: str,
) -> RubricRuleResult:
    # try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

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
    compliance = RuleCompliance.PASS if score >= 0.5 else RuleCompliance.FAIL
    log.info(f"  [{compliance.value}] {rule.id}  score={score:.3f}")
    return RubricRuleResult(
        rule_id=rule.id,
        compliance=compliance,
        reasoning=g_eval.reason or "",
        confidence=score,
    )
    # except Exception as exc:
    #     log.error(f"Rule '{rule.id}' evaluation failed: {exc}")
    #     return RubricRuleResult(
    #         rule_id=rule.id,
    #         compliance=RuleCompliance.UNCERTAIN,
    #         reasoning="",
    #         confidence=0.0,
    #         error=str(exc),
    #     )


def run(
    generation: str,
    rubric_rules: list[RubricRule],
    judge_model: str = "gpt-4o-mini",
) -> RubricEvalResult:
    """Evaluate the generation against all rubric rules in parallel.

    Args:
        generation: The LLM-generated text to evaluate.
        rubric_rules: List of RubricRule objects to check.
        judge_model: LiteLLM model string for the judge LLM.

    Returns:
        RubricEvalResult with per-rule compliance and aggregate scores.
    """
    if not rubric_rules:
        return RubricEvalResult()

    async def _run_all() -> list[RubricRuleResult]:
        tasks = [_evaluate_rule(rule, generation, judge_model) for rule in rubric_rules]
        return await asyncio.gather(*tasks)

    rule_results = asyncio.run(_run_all())

    passed = sum(1 for r in rule_results if r.compliance == RuleCompliance.PASS)
    compliance_rate = passed / len(rule_results) if rule_results else 0.0
    composite = sum(r.confidence for r in rule_results) / len(rule_results) if rule_results else 0.0

    return RubricEvalResult(
        rule_results=rule_results,
        compliance_rate=compliance_rate,
        composite_adherence_score=composite,
    )
