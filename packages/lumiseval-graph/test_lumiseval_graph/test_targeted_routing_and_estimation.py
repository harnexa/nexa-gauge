from lumiseval_core.pipeline import NODE_PREREQUISITES
from lumiseval_core.types import (
    CostMetadata,
    EvalJobConfig,
    GorundingCostMeta,
    InputMetadata,
    RedTeamCostMeta,
    RelevanceCostMeta,
    RubricCostMeta,
)
from lumiseval_graph.node_runner import NodeRunner
from lumiseval_graph.nodes.cost_estimator import CostEstimator


def _cost_meta(
    *,
    grounding_eligible: int = 10,
    relevance_eligible: int = 10,
    rubric_eligible: int = 10,
    redteam_eligible: int = 10,
) -> CostMetadata:
    return CostMetadata(
        grounding=GorundingCostMeta(
            eligible_records=grounding_eligible,
            avg_claims_per_record=2.0,
            avg_context_tokens=200,
        ),
        relevance=RelevanceCostMeta(
            eligible_records=relevance_eligible,
            avg_claims_per_record=2.0,
            avg_question_tokens=20,
        ),
        rubric=RubricCostMeta(
            eligible_records=rubric_eligible,
            rule_count=0,
            unique_rule_count=0,
            rule_tokens=0.0,
            unique_rule_tokens=0.0,
        ),
        readteam=RedTeamCostMeta(eligible_records=redteam_eligible),
    )


def _meta(generation_chunk_count: int = 20) -> InputMetadata:
    return InputMetadata(
        record_count=10,
        total_tokens=1000,
        generation_chunk_count=generation_chunk_count,
        cost_meta=_cost_meta(),
    )


def test_redteam_and_rubric_do_not_depend_on_claims_branch() -> None:
    redteam_plan = NODE_PREREQUISITES["redteam"]
    rubric_plan = NODE_PREREQUISITES["rubric"]

    for plan in (redteam_plan, rubric_plan):
        assert "chunk" not in plan
        assert "claims" not in plan
        assert "dedupe" not in plan


def test_estimate_for_redteam_is_not_claim_based() -> None:
    cfg = EvalJobConfig(
        job_id="j1",
        enable_redteam=True,
        enable_grounding=False,
        enable_relevance=False,
    )
    report = CostEstimator(cfg).estimate(_meta())
    row = report.row("redteam")

    assert row.model_calls > 0
    assert row.source.startswith("redteam[")
    assert report.row("claims").model_calls == 0  # claims disabled (grounding+relevance off)


def test_estimate_for_relevance_includes_claim_path() -> None:
    cfg = EvalJobConfig(
        job_id="j2",
        enable_grounding=False,
        enable_relevance=True,
    )
    meta = _meta()
    report = CostEstimator(cfg).estimate(meta)

    relevance_row = report.row("relevance")
    claims_row = report.row("claims")

    # relevance row = claims + relevance combined; formula annotates the relevance portion
    assert relevance_row.source.startswith("claims + relevance[")

    estimator = CostEstimator(cfg)
    individual_claims = estimator._estimate_node("claims", meta).model_calls
    individual_relevance = estimator._estimate_node("relevance", meta).model_calls
    assert relevance_row.model_calls == individual_claims + individual_relevance

    # claims row shows claims with its formula
    assert claims_row.source.startswith("claims[")


def test_estimate_for_rubric_uses_rule_count() -> None:
    meta = InputMetadata(
        record_count=10,
        total_tokens=1000,
        generation_chunk_count=20,
        cost_meta=_cost_meta(rubric_eligible=10).model_copy(
            update={
                "rubric": RubricCostMeta(
                    eligible_records=10,
                    rule_count=17,
                    unique_rule_count=17,
                    rule_tokens=200.0,
                    unique_rule_tokens=200.0,
                )
            }
        ),
    )
    cfg = EvalJobConfig(job_id="j3", enable_rubric=True)
    report = CostEstimator(cfg).estimate(meta)
    row = report.row("rubric")

    assert row.model_calls > 0
    assert row.source.startswith("rubric[")


def test_case_eligibility_requires_context_for_claim_path_and_rubric_for_rubric() -> None:
    from lumiseval_core.types import EvalCase, Rubric

    no_context = EvalCase(case_id="c1", generation="answer")
    with_context_and_rubric = EvalCase(
        case_id="c2",
        generation="answer",
        context=["context passage"],
        rubric=[Rubric(id="R-1", statement="s", pass_condition="p")],
    )

    assert not NodeRunner.is_case_eligible_for_node(no_context, "claims")
    assert not NodeRunner.is_case_eligible_for_node(no_context, "relevance")
    assert not NodeRunner.is_case_eligible_for_node(no_context, "grounding")
    assert not NodeRunner.is_case_eligible_for_node(no_context, "rubric")
    assert NodeRunner.is_case_eligible_for_node(no_context, "redteam")

    assert NodeRunner.is_case_eligible_for_node(with_context_and_rubric, "claims")
    assert NodeRunner.is_case_eligible_for_node(with_context_and_rubric, "relevance")
    assert NodeRunner.is_case_eligible_for_node(with_context_and_rubric, "grounding")
    assert NodeRunner.is_case_eligible_for_node(with_context_and_rubric, "rubric")


def test_disabled_nodes_have_zero_cost() -> None:
    cfg = EvalJobConfig(
        job_id="j5",
        enable_grounding=False,
        enable_relevance=False,
        enable_redteam=False,
        enable_rubric=False,
    )
    report = CostEstimator(cfg).estimate(_meta())

    for node in ("claims", "grounding", "relevance", "redteam", "rubric"):
        assert report.row(node).model_calls == 0, f"{node} should be zero cost when disabled"


def test_eval_row_is_sum_of_all_metric_nodes() -> None:
    cfg = EvalJobConfig(
        job_id="j6",
        enable_grounding=True,
        enable_redteam=True,
        enable_rubric=True,
    )
    meta = InputMetadata(
        record_count=10,
        total_tokens=1000,
        generation_chunk_count=20,
        cost_meta=_cost_meta(rubric_eligible=10).model_copy(
            update={
                "rubric": RubricCostMeta(
                    eligible_records=10,
                    rule_count=5,
                    unique_rule_count=5,
                    rule_tokens=100.0,
                    unique_rule_tokens=100.0,
                )
            }
        ),
    )
    report = CostEstimator(cfg).estimate(meta)
    eval_row = report.row("eval")

    # eval cumulative = all individual node costs summed directly
    from lumiseval_graph.nodes.cost_estimator import CostEstimator as CE

    estimator = CE(cfg)
    individual_claims = estimator._estimate_node("claims", meta).model_calls
    individual_grounding = estimator._estimate_node("grounding", meta).model_calls
    individual_relevance = estimator._estimate_node("relevance", meta).model_calls
    individual_redteam = estimator._estimate_node("redteam", meta).model_calls
    individual_rubric = estimator._estimate_node("rubric", meta).model_calls

    assert eval_row.model_calls == (
        individual_claims
        + individual_grounding
        + individual_relevance
        + individual_redteam
        + individual_rubric
    )
    assert "claims" in eval_row.source
    assert "grounding" in eval_row.source
    assert "relevance" in eval_row.source
    assert "redteam" in eval_row.source
    assert "rubric" in eval_row.source
