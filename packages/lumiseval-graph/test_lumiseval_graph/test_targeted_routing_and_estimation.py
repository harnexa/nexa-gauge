from lumiseval_core.cache import CacheStore
from lumiseval_core.geval_cache import GevalArtifactCache, compute_geval_signature
from lumiseval_core.pipeline import NODE_ORDER, NODES_BY_NAME
from lumiseval_core.types import (
    ClaimCostMeta,
    CostMetadata,
    EvalCase,
    EvalJobConfig,
    GevalConfig,
    GevalCostMeta,
    GevalMetricSpec,
    GevalStepsCostMeta,
    GorundingCostMeta,
    InputMetadata,
    MetricCategory,
    MetricResult,
    NodeCostBreakdown,
    RedTeamCostMeta,
    ReferenceCostMeta,
    RelevanceCostMeta,
)
from lumiseval_graph.node_runner import CachedNodeRunner, NodeRunner
from lumiseval_graph.nodes import eval as eval_node
from lumiseval_graph.nodes.cost_estimator import CostEstimator
from lumiseval_ingest.scanner import scan_cases


def _cost_meta(
    *,
    claim_eligible: int = 10,
    grounding_eligible: int = 10,
    relevance_eligible: int = 10,
    geval_steps_eligible: int = 10,
    geval_steps_count: int = 10,
    unique_geval_steps_count: int = 10,
    geval_steps_tokens: float = 200.0,
    unique_geval_steps_tokens: float = 200.0,
    geval_eligible: int = 10,
    geval_rule_count: int = 10,
    geval_unique_rule_count: int = 10,
    geval_rule_tokens: float = 200.0,
    geval_unique_rule_tokens: float = 200.0,
    redteam_eligible: int = 10,
) -> CostMetadata:
    return CostMetadata(
        claim=ClaimCostMeta(
            eligible_records=claim_eligible,
            avg_generation_chunks=2.0,
            avg_generation_tokens=50.0,
        ),
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
        geval_steps=GevalStepsCostMeta(
            eligible_records=geval_steps_eligible,
            criteria_count=geval_steps_count,
            unique_criteria_count=unique_geval_steps_count,
            criteria_tokens=geval_steps_tokens,
            unique_criteria_tokens=unique_geval_steps_tokens,
        ),
        geval=GevalCostMeta(
            eligible_records=geval_eligible,
            rule_count=geval_rule_count,
            unique_rule_count=geval_unique_rule_count,
            rule_tokens=geval_rule_tokens,
            unique_rule_tokens=geval_unique_rule_tokens,
        ),
        readteam=RedTeamCostMeta(eligible_records=redteam_eligible),
        reference=ReferenceCostMeta(eligible_records=0),
    )


def _meta(generation_chunk_count: int = 20) -> InputMetadata:
    return InputMetadata(
        record_count=10,
        total_tokens=1000,
        generation_chunk_count=generation_chunk_count,
        cost_meta=_cost_meta(),
    )


def test_redteam_and_geval_do_not_depend_on_claims_branch() -> None:
    redteam_plan = list(NODES_BY_NAME["redteam"].prerequisites)
    geval_plan = list(NODES_BY_NAME["geval"].prerequisites)

    for plan in (redteam_plan, geval_plan):
        assert "chunk" not in plan
        assert "claims" not in plan
        assert "dedupe" not in plan
    assert geval_plan == ["scan", "geval_steps"]


def test_topology_no_longer_includes_estimate_or_approve_nodes() -> None:
    assert "estimate" not in NODE_ORDER
    assert "approve" not in NODE_ORDER
    assert list(NODES_BY_NAME["chunk"].prerequisites) == ["scan"]
    assert list(NODES_BY_NAME["claims"].prerequisites) == ["scan", "chunk"]
    assert list(NODES_BY_NAME["dedupe"].prerequisites) == ["scan", "chunk", "claims"]
    assert list(NODES_BY_NAME["geval_steps"].prerequisites) == ["scan"]
    assert list(NODES_BY_NAME["geval"].prerequisites) == ["scan", "geval_steps"]
    assert list(NODES_BY_NAME["redteam"].prerequisites) == ["scan"]
    assert list(NODES_BY_NAME["reference"].prerequisites) == ["scan"]
    assert "estimate" not in NODES_BY_NAME["eval"].prerequisites
    assert "approve" not in NODES_BY_NAME["eval"].prerequisites
    assert "geval_steps" in NODES_BY_NAME["eval"].prerequisites
    assert "geval" in NODES_BY_NAME["eval"].prerequisites


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
    assert row.source.startswith("redteam(")
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
    assert relevance_row.source.startswith("claims + relevance(")

    estimator = CostEstimator(cfg)
    individual_claims = estimator._estimate_node("claims", meta).model_calls
    individual_relevance = estimator._estimate_node("relevance", meta).model_calls
    assert relevance_row.model_calls == individual_claims + individual_relevance

    # claims row shows claims with its formula
    assert claims_row.source.startswith("claims(")


def test_claim_cost_uses_claim_eligibility_when_records_present() -> None:
    """Regression: relevance-only scans without context must still price claims."""
    cfg = EvalJobConfig(
        job_id="j2b",
        enable_grounding=False,
        enable_relevance=True,
    )
    cases = [
        EvalCase(
            case_id="c-relevance-only",
            generation="The Eiffel Tower is in Paris.",
            question="Where is the Eiffel Tower?",
            context=[],
        )
    ]
    meta = scan_cases(cases, show_progress=False)
    estimator = CostEstimator(cfg)

    assert estimator._eligible_records("grounding", meta) == 0
    assert estimator._eligible_records("claims", meta) == 1
    assert estimator._estimate_node("claims", meta).model_calls > 0


def test_estimate_for_geval_uses_rule_count() -> None:
    meta = InputMetadata(
        record_count=10,
        total_tokens=1000,
        generation_chunk_count=20,
        cost_meta=_cost_meta(geval_eligible=10).model_copy(
            update={
                "geval": GevalCostMeta(
                    eligible_records=10,
                    rule_count=17,
                    unique_rule_count=17,
                    rule_tokens=200.0,
                    unique_rule_tokens=200.0,
                ),
            }
        ),
    )
    cfg = EvalJobConfig(job_id="j3", enable_geval=True)
    report = CostEstimator(cfg).estimate(meta)
    row = report.row("geval")

    assert row.model_calls > 0
    assert row.source.startswith("geval_steps + geval(")


def test_case_eligibility_generation_for_claims_question_for_relevance_context_for_grounding() -> None:
    from lumiseval_core.types import EvalCase

    no_context = EvalCase(case_id="c1", generation="answer")
    with_context = EvalCase(
        case_id="c2",
        generation="answer",
        context=["context passage"],
    )
    with_question = EvalCase(case_id="c3", generation="answer", question="What is it?")
    with_geval = EvalCase(
        case_id="c4",
        generation="answer",
        geval=GevalConfig(
            metrics=[
                GevalMetricSpec(
                    name="factuality",
                    criteria="Must be factual.",
                    record_fields=["generation"],
                )
            ]
        ),
    )

    # claims only requires generation
    assert NodeRunner.is_case_eligible_for_node(no_context, "claims")
    assert NodeRunner.is_case_eligible_for_node(with_context, "claims")

    # relevance requires question
    assert not NodeRunner.is_case_eligible_for_node(no_context, "relevance")
    assert not NodeRunner.is_case_eligible_for_node(with_context, "relevance")
    assert NodeRunner.is_case_eligible_for_node(with_question, "relevance")

    # grounding requires context
    assert not NodeRunner.is_case_eligible_for_node(no_context, "grounding")
    assert NodeRunner.is_case_eligible_for_node(with_context, "grounding")
    assert not NodeRunner.is_case_eligible_for_node(no_context, "geval_steps")
    assert not NodeRunner.is_case_eligible_for_node(no_context, "geval")
    assert NodeRunner.is_case_eligible_for_node(with_geval, "geval_steps")
    assert NodeRunner.is_case_eligible_for_node(with_geval, "geval")

    # reference requires reference field; redteam only requires generation
    assert not NodeRunner.is_case_eligible_for_node(no_context, "reference")
    assert NodeRunner.is_case_eligible_for_node(no_context, "redteam")


def test_disabled_nodes_have_zero_cost() -> None:
    cfg = EvalJobConfig(
        job_id="j5",
        enable_grounding=False,
        enable_relevance=False,
        enable_redteam=False,
        enable_geval=False,
    )
    report = CostEstimator(cfg).estimate(_meta())

    for node in ("claims", "grounding", "relevance", "redteam", "geval_steps", "geval"):
        assert report.row(node).model_calls == 0, f"{node} should be zero cost when disabled"


def test_dedupe_has_zero_individual_llm_cost() -> None:
    cfg = EvalJobConfig(job_id="j5b", enable_grounding=True, enable_relevance=False)
    report = CostEstimator(cfg).estimate(_meta())
    row = report.row("dedupe")

    assert row.individual_cost_usd == 0.0
    # Cumulative dedupe cost still includes upstream claim extraction when enabled.
    assert row.cost_usd == report.row("claims").cost_usd


def test_eval_row_is_sum_of_all_metric_nodes() -> None:
    cfg = EvalJobConfig(
        job_id="j6",
        enable_grounding=True,
        enable_redteam=True,
        enable_geval=True,
    )
    meta = InputMetadata(
        record_count=10,
        total_tokens=1000,
        generation_chunk_count=20,
        cost_meta=_cost_meta(geval_eligible=10).model_copy(
            update={
                "geval": GevalCostMeta(
                    eligible_records=10,
                    rule_count=5,
                    unique_rule_count=5,
                    rule_tokens=100.0,
                    unique_rule_tokens=100.0,
                ),
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
    individual_geval_steps = estimator._estimate_node("geval_steps", meta).model_calls
    individual_geval = estimator._estimate_node("geval", meta).model_calls

    assert eval_row.model_calls == (
        individual_claims
        + individual_grounding
        + individual_relevance
        + individual_redteam
        + individual_geval_steps
        + individual_geval
    )
    assert "claims" in eval_row.source
    assert "grounding" in eval_row.source
    assert "relevance" in eval_row.source
    assert "redteam" in eval_row.source
    assert "geval_steps" in eval_row.source
    assert "geval" in eval_row.source


def test_geval_steps_estimate_uses_artifact_cache_hits(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LUMISEVAL_CACHE_DIR", str(tmp_path))
    cfg = EvalJobConfig(
        job_id="j-geval-cache",
        enable_grounding=False,
        enable_relevance=False,
        enable_redteam=False,
        enable_geval=True,
        enable_reference=False,
    )
    geval_cost_meta = GevalCostMeta(
        eligible_records=1,
        rule_count=1,
        unique_rule_count=1,
        rule_tokens=20.0,
        unique_rule_tokens=20.0,
    )
    meta = InputMetadata(
        record_count=1,
        total_tokens=100,
        generation_chunk_count=1,
        cost_meta=_cost_meta(geval_eligible=1).model_copy(update={"geval": geval_cost_meta}),
    )
    metric = GevalMetricSpec(
        name="mention_paris",
        record_fields=["generation"],
        criteria="The answer must mention Paris.",
    )
    case = EvalCase(
        case_id="c-geval",
        generation="The Eiffel Tower is in Paris.",
        geval=GevalConfig(metrics=[metric]),
    )

    estimator = CostEstimator(cfg)
    miss_breakdown = estimator._estimate_node("geval_steps", meta, cases=[case])
    assert miss_breakdown.model_calls == 1

    signature = compute_geval_signature(criteria=metric.criteria or "", model=cfg.judge_model)
    GevalArtifactCache().put_steps(
        signature=signature,
        model=cfg.judge_model,
        criteria=metric.criteria or "",
        evaluation_steps=["Check if Paris is mentioned."],
    )
    hit_breakdown = estimator._estimate_node("geval_steps", meta, cases=[case])
    assert hit_breakdown.model_calls == 0


def test_eval_scoring_uses_geval_metrics() -> None:
    cfg = EvalJobConfig(job_id="j-geval-scoring", enable_geval=True)
    report = eval_node.aggregate(
        job_id=cfg.job_id,
        grounding_metrics=[],
        relevance_metrics=[],
        redteam_metrics=[],
        geval_metrics=[
            MetricResult(
                name="geval_rule",
                category=MetricCategory.ANSWER,
                score=1.0,
            )
        ],
        reference_metrics=[],
        cost_estimate=None,
        cost_actual_usd=0.0,
        job_config=cfg,
    )

    assert report.answer_score is not None
    assert report.answer_score.score == 1.0
    assert [m.name for m in report.answer_score.metrics] == ["geval_rule"]


def test_estimate_can_use_node_overrides_for_delta_costing() -> None:
    cfg = EvalJobConfig(job_id="j7")
    meta = _meta()
    overrides = {node: NodeCostBreakdown(model_calls=0, cost_usd=0.0) for node in NODE_ORDER}
    overrides["redteam"] = NodeCostBreakdown(model_calls=4, cost_usd=0.2)
    eligible = {node: 0 for node in NODE_ORDER}
    eligible["redteam"] = 2

    report = CostEstimator(cfg).estimate(
        meta,
        individual_overrides=overrides,
        eligible_overrides=eligible,
    )

    assert report.row("redteam").model_calls == 4
    assert report.row("redteam").individual_cost_usd == 0.2
    assert report.row("redteam").eligible_records == 2
    assert report.row("eval").cost_usd == 0.2


def test_eval_run_injects_cost_estimate_without_graph_estimate_node(tmp_path) -> None:
    runner = CachedNodeRunner(cache_store=CacheStore(tmp_path))
    cfg = EvalJobConfig(
        job_id="j8",
        enable_grounding=False,
        enable_relevance=False,
        enable_redteam=False,
        enable_geval=False,
        enable_reference=False,
    )
    case = EvalCase(case_id="c-eval", generation="A concise answer.")

    result = runner.run_case(case=case, node_name="eval", job_config=cfg)

    assert result.final_state.get("cost_estimate") is not None
    report = result.final_state.get("report")
    assert report is not None
    assert report.cost_estimate is not None
