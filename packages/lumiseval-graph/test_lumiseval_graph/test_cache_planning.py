from lumiseval_core.cache import CacheStore, compute_case_hash, compute_config_hash
from lumiseval_core.pipeline import NODES_BY_NAME
from lumiseval_core.types import EvalCase, EvalJobConfig
from lumiseval_graph.node_runner import CachedNodeRunner


def _seed_cache_for_target(
    store: CacheStore,
    *,
    case: EvalCase,
    cfg: EvalJobConfig,
    target_node: str,
) -> None:
    case_hash = compute_case_hash(
        generation=case.generation,
        question=case.question,
        reference=case.reference,
        geval=case.geval,
        context=case.context or [],
        reference_files=case.reference_files or [],
    )
    config_hash = compute_config_hash(cfg)
    for step in list(NODES_BY_NAME[target_node].prerequisites) + [target_node]:
        store.put(case_hash, config_hash, step, {"_cached": True})


def test_grounding_plan_reuses_cached_claim_path(tmp_path) -> None:
    store = CacheStore(tmp_path)
    runner = CachedNodeRunner(cache_store=store)
    cfg = EvalJobConfig(job_id="job-1", enable_grounding=True, enable_relevance=True)
    case = EvalCase(
        case_id="case-1",
        generation="The Eiffel Tower is in Paris.",
        question="Where is Eiffel Tower?",
        context=["The Eiffel Tower is in Paris, France."],
    )

    _seed_cache_for_target(store, case=case, cfg=cfg, target_node="relevance")
    plan = runner.plan_dataset(cases=[case], node_name="grounding", job_config=cfg)

    for shared_step in ("scan", "chunk", "claims", "dedupe"):
        assert plan.cached_count(shared_step) == 1
        assert plan.to_run_count(shared_step) == 0
    assert plan.to_run_count("grounding") == 1


def test_relevance_plan_only_runs_new_cases(tmp_path) -> None:
    store = CacheStore(tmp_path)
    runner = CachedNodeRunner(cache_store=store)
    cfg = EvalJobConfig(job_id="job-2", enable_grounding=False, enable_relevance=True)
    cached_cases = [
        EvalCase(
            case_id="case-1",
            generation="A",
            question="Q1",
            context=["ctx1"],
        ),
        EvalCase(
            case_id="case-2",
            generation="B",
            question="Q2",
            context=["ctx2"],
        ),
    ]
    new_case = EvalCase(
        case_id="case-3",
        generation="C",
        question="Q3",
        context=["ctx3"],
    )
    all_cases = [*cached_cases, new_case]

    for case in cached_cases:
        _seed_cache_for_target(store, case=case, cfg=cfg, target_node="relevance")

    plan = runner.plan_dataset(cases=all_cases, node_name="relevance", job_config=cfg)
    relevance_path = list(NODES_BY_NAME["relevance"].prerequisites) + ["relevance"]
    for step in relevance_path:
        assert plan.cached_count(step) == 2
        assert plan.to_run_count(step) == 1
