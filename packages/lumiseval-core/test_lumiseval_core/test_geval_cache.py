from lumiseval_core.geval_cache import (
    GevalArtifactCache,
    collect_geval_signatures,
    compute_geval_signature,
)
from lumiseval_core.types import EvalCase, GevalConfig, GevalMetricSpec


def test_compute_geval_signature_changes_with_model_and_criteria() -> None:
    sig_a_default = compute_geval_signature(criteria="Mention Paris.", model="gpt-4o-mini")
    sig_a_other_model = compute_geval_signature(criteria="Mention Paris.", model="gpt-4o")
    sig_b_default = compute_geval_signature(criteria="Mention Lyon.", model="gpt-4o-mini")

    assert sig_a_default != sig_a_other_model
    assert sig_a_default != sig_b_default


def test_geval_artifact_cache_reads_canonical_artifacts(tmp_path) -> None:
    signature = compute_geval_signature(
        criteria="Mention Paris.",
        model="gpt-4o-mini",
    )
    cache = GevalArtifactCache(tmp_path)
    cache.put_steps(
        signature=signature,
        model="gpt-4o-mini",
        criteria="Mention Paris.",
        evaluation_steps=["Check whether Paris appears in the answer."],
    )

    assert cache.get_steps(signature) == ["Check whether Paris appears in the answer."]
    assert cache.count_missing({signature}) == 0


def test_collect_geval_signatures_deduplicates_across_cases() -> None:
    metric = GevalMetricSpec(
        name="factuality",
        record_fields=["generation"],
        criteria="Must be factual.",
    )
    cases = [
        EvalCase(case_id="c1", generation="a", geval=GevalConfig(metrics=[metric])),
        EvalCase(case_id="c2", generation="b", geval=GevalConfig(metrics=[metric])),
    ]
    signatures = collect_geval_signatures(cases=cases, model="gpt-4o-mini")
    assert len(signatures) == 1
