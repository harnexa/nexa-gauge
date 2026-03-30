# Run smoke test:
#   python -m lumiseval_graph.nodes.metrics.rubric
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
from lumiseval_core.types import MetricCategory, MetricResult, NodeCostBreakdown, RubricRule

from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.metrics.base import BaseMetricNode

log = get_node_logger("rubric_eval")


class RubricNode(BaseMetricNode):
    node_name = "rubric"
    # Delegates to DeepEval GEval — no custom prompt

    async def _evaluate_rule(self, rule: RubricRule, generation: str) -> MetricResult:
        criteria = f"{rule.statement}\n\nPass condition: {rule.pass_condition}"
        test_case = LLMTestCase(input="", actual_output=generation)
        g_eval = GEval(
            name=rule.id,
            criteria=criteria,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.judge_model,
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

    def run(  # type: ignore[override]
        self,
        *,
        generation: str,
        rubric_rules: list[RubricRule],
    ) -> list[MetricResult]:
        """Evaluate the generation against all rubric rules in parallel.

        Args:
            generation:   The LLM-generated text to evaluate.
            rubric_rules: List of RubricRule objects to check.

        Returns:
            list[MetricResult] — one per rubric rule, scored 0.0–1.0.
        """
        if not rubric_rules:
            return []

        async def _run_all() -> list[MetricResult]:
            tasks = [self._evaluate_rule(rule, generation) for rule in rubric_rules]
            return await asyncio.gather(*tasks)

        return asyncio.run(_run_all())

    def cost_estimate(self, *, rubric_rule_count: int = 0, **kwargs) -> NodeCostBreakdown:
        # TODO: compute from GEval prompt tokens * rubric_rule_count
        return NodeCostBreakdown(judge_calls=0, cost_usd=0.0)
