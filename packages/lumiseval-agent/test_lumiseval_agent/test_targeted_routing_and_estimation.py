from lumiseval_agent.node_runner import NodeRunner
from lumiseval_agent.nodes.cost_estimator import estimate
from lumiseval_core.constants import COST_WEB_SEARCH_CLAIM_FRACTION
from lumiseval_core.types import EvalJobConfig, InputMetadata


def _meta() -> InputMetadata:
    return InputMetadata(
        record_count=10,
        total_tokens=1000,
        total_chars=6000,
        estimated_chunk_count=20,
        estimated_claim_count=40,
    )


def test_redteam_and_rubric_do_not_depend_on_claims_branch() -> None:
    redteam_plan = NodeRunner._prerequisites["redteam"]
    rubric_plan = NodeRunner._prerequisites["rubric"]

    for plan in (redteam_plan, rubric_plan):
        assert "chunk" not in plan
        assert "claims" not in plan
        assert "dedupe" not in plan
        assert "retrieve" not in plan


def test_estimate_for_redteam_is_not_claim_based() -> None:
    cfg = EvalJobConfig(job_id="j1", enable_adversarial=True, web_search=True)
    cost = estimate(_meta(), cfg, target_node="redteam")

    assert cost.estimated_judge_calls == 10
    assert cost.estimated_embedding_calls == 0
    assert cost.estimated_tavily_calls == 0


def test_estimate_for_relevance_includes_claim_path() -> None:
    cfg = EvalJobConfig(job_id="j2", web_search=True)
    cost = estimate(_meta(), cfg, target_node="relevance")

    # claims extraction (per chunk) + relevance metric call (per record)
    assert cost.estimated_judge_calls == 30
    assert cost.estimated_embedding_calls == 20
    assert cost.estimated_tavily_calls == 16


def test_estimate_for_rubric_uses_rule_count() -> None:
    cfg = EvalJobConfig(job_id="j3", enable_rubric=True)
    cost = estimate(_meta(), cfg, target_node="rubric", rubric_rule_count=17)

    assert cost.estimated_judge_calls == 17
    assert cost.estimated_embedding_calls == 0
    assert cost.estimated_tavily_calls == 0


def test_case_eligibility_requires_context_for_claim_path_and_rubric_for_rubric() -> None:
    from lumiseval_core.types import EvalCase, RubricRule

    no_context = EvalCase(case_id="c1", generation="answer")
    with_context_and_rubric = EvalCase(
        case_id="c2",
        generation="answer",
        context=["context passage"],
        rubric_rules=[RubricRule(id="R-1", statement="s", pass_condition="p")],
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


def test_estimate_uses_eligible_counts_for_context_bound_nodes() -> None:
    meta = InputMetadata(
        record_count=10,
        total_tokens=1000,
        total_chars=6000,
        estimated_chunk_count=20,
        estimated_claim_count=40,
        eligible_record_count={"relevance": 4, "claims": 4, "retrieve": 4, "redteam": 10},
        eligible_chunk_count={"claims": 8},
        eligible_claim_count={"retrieve": 16},
    )
    cfg = EvalJobConfig(job_id="j4", web_search=True)
    cost = estimate(meta, cfg, target_node="relevance")

    assert cost.estimated_judge_calls == 12  # 8 claim calls + 4 relevance calls
    assert cost.estimated_embedding_calls == 8
    assert cost.estimated_tavily_calls == int(16 * COST_WEB_SEARCH_CLAIM_FRACTION)
