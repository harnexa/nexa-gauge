from __future__ import annotations

from types import SimpleNamespace

import ng_cli.main as main_module
import pytest
from ng_cli.main import (
    DEFAULT_FALLBACK_LLM,
    DEFAULT_PRIMARY_LLM,
    _collect_estimate_rows,
    _is_case_eligible_for_target_path,
    _parse_model_overrides,
    _resolve_runtime_llm_overrides,
)
from ng_cli.util import _resolve_target_node
from ng_core.types import CostEstimate


def test_parse_model_overrides_accepts_global_and_node_values() -> None:
    global_model, per_node, warnings = _parse_model_overrides(
        ["openai/gpt-4o", "grounding=openai/gpt-4o-mini"],
        option_name="--llm-model",
    )
    assert global_model == "openai/gpt-4o"
    assert per_node == {"grounding": "openai/gpt-4o-mini"}
    assert warnings == []


def test_parse_model_overrides_duplicate_node_last_wins_with_warning() -> None:
    _, per_node, warnings = _parse_model_overrides(
        ["grounding=openai/gpt-4o", "grounding=openai/gpt-4o-mini"],
        option_name="--llm-model",
    )
    assert per_node == {"grounding": "openai/gpt-4o-mini"}
    assert len(warnings) == 1
    assert "last value 'openai/gpt-4o-mini' wins" in warnings[0]


def test_parse_model_overrides_rejects_conflicting_global_values() -> None:
    with pytest.raises(ValueError, match="Conflicting global values"):
        _parse_model_overrides(
            ["openai/gpt-4o", "openai/gpt-4o-mini"],
            option_name="--llm-model",
        )


def test_parse_model_overrides_rejects_unknown_node() -> None:
    with pytest.raises(ValueError, match="Unknown node"):
        _parse_model_overrides(
            ["not_a_node=openai/gpt-4o-mini"],
            option_name="--llm-model",
        )


def test_resolve_runtime_llm_overrides_uses_defaults_for_branch() -> None:
    _, overrides, warnings = _resolve_runtime_llm_overrides(
        target_node="grounding",
        legacy_model=DEFAULT_PRIMARY_LLM,
        llm_model_values=(),
        llm_fallback_values=(),
    )
    assert warnings == []

    expected_nodes = {"scan", "chunk", "claims", "dedup", "grounding"}
    assert set(overrides["models"]) == expected_nodes
    assert set(overrides["fallback_models"]) == expected_nodes
    assert all(model == DEFAULT_PRIMARY_LLM for model in overrides["models"].values())
    assert all(model == DEFAULT_FALLBACK_LLM for model in overrides["fallback_models"].values())


def test_resolve_runtime_llm_overrides_ignores_non_branch_node_with_warning() -> None:
    _, overrides, warnings = _resolve_runtime_llm_overrides(
        target_node="grounding",
        legacy_model=DEFAULT_PRIMARY_LLM,
        llm_model_values=("relevance=openai/gpt-4o",),
        llm_fallback_values=(),
    )
    assert "relevance" not in overrides["models"]
    assert len(warnings) == 1
    assert "not in target branch 'grounding'" in warnings[0]


def test_resolve_runtime_llm_overrides_rejects_legacy_global_conflict() -> None:
    with pytest.raises(
        ValueError, match="Conflicting global model values from --model and --llm-model"
    ):
        _resolve_runtime_llm_overrides(
            target_node="grounding",
            legacy_model="openai/gpt-4o",
            llm_model_values=("openai/gpt-4o-mini",),
            llm_fallback_values=(),
        )


def test_resolve_runtime_llm_overrides_applies_global_then_node_override() -> None:
    _, overrides, warnings = _resolve_runtime_llm_overrides(
        target_node="grounding",
        legacy_model=DEFAULT_PRIMARY_LLM,
        llm_model_values=("openai/gpt-4o", "grounding=openai/gpt-4o-mini"),
        llm_fallback_values=(),
    )
    assert warnings == []

    assert overrides["models"]["scan"] == "openai/gpt-4o"
    assert overrides["models"]["grounding"] == "openai/gpt-4o-mini"
    assert overrides["fallback_models"]["grounding"] == DEFAULT_FALLBACK_LLM


def test_collect_estimate_rows_includes_all_branch_nodes_with_status() -> None:
    rows = _collect_estimate_rows(
        target_node="grounding",
        cost_by_node={
            "chunk": CostEstimate(cost=0.15, input_tokens=10, output_tokens=2),
            "claims": CostEstimate(cost=0.0, input_tokens=0, output_tokens=0),
            "dedup": CostEstimate(cost=0.0, input_tokens=0, output_tokens=0),
            "grounding": CostEstimate(cost=0.0, input_tokens=0, output_tokens=0),
        },
        node_stats={
            "scan": {"executed": 1, "cached": 0, "estimated": 0},
            "chunk": {"executed": 1, "cached": 0, "estimated": 1},
            "claims": {"executed": 1, "cached": 0, "estimated": 1},
            "dedup": {"executed": 1, "cached": 0, "estimated": 1},
            "grounding": {"executed": 0, "cached": 1, "estimated": 0},
        },
        total_selected_cases=2,
        successful_cases=1,
        effective_judge_model=DEFAULT_PRIMARY_LLM,
        llm_overrides={
            "models": {
                "scan": DEFAULT_PRIMARY_LLM,
                "chunk": "openai/gpt-4o",
                "claims": DEFAULT_PRIMARY_LLM,
                "dedup": DEFAULT_PRIMARY_LLM,
                "grounding": DEFAULT_PRIMARY_LLM,
            },
            "fallback_models": {},
        },
    )

    assert rows == [
        ("scan", DEFAULT_PRIMARY_LLM, "skipped/ineligible", "1 / 2", "0 / 2", "0 / 2", "0.0%", 0.0),
        ("chunk", "openai/gpt-4o", "billable", "1 / 2", "0 / 2", "0 / 2", "0.0%", 0.15),
        ("claims", DEFAULT_PRIMARY_LLM, "zero_cost", "1 / 2", "0 / 2", "0 / 2", "0.0%", 0.0),
        ("dedup", DEFAULT_PRIMARY_LLM, "zero_cost", "1 / 2", "0 / 2", "0 / 2", "0.0%", 0.0),
        ("grounding", DEFAULT_PRIMARY_LLM, "cached_only", "0 / 2", "1 / 2", "0 / 2", "0.0%", 0.0),
    ]


def test_is_case_eligible_for_target_path_requires_full_grounding_branch() -> None:
    assert (
        _is_case_eligible_for_target_path(
            target_node="grounding",
            case={"case_id": "c1", "generation": "hello", "context": "ctx"},
        )
        is True
    )
    assert (
        _is_case_eligible_for_target_path(
            target_node="grounding",
            case={"case_id": "c2", "generation": "hello"},
        )
        is False
    )


def test_is_case_eligible_for_eval_keeps_non_intersection_records() -> None:
    assert (
        _is_case_eligible_for_target_path(
            target_node="eval",
            case={"case_id": "c1", "generation": "hello"},
        )
        is True
    )


def test_resolve_target_node_allows_eval() -> None:
    assert _resolve_target_node("eval") == "eval"


def test_resolve_target_node_rejects_report() -> None:
    with pytest.raises(ValueError, match="not directly invocable"):
        _resolve_target_node("report")


def test_estimate_command_uses_estimate_execution_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Adapter:
        def iter_cases(self, split: str = "train", limit: int | None = None):
            del split, limit
            yield {"case_id": "c1", "generation": "hello", "context": "ctx"}

    class _Runner:
        def __init__(self, cache_store):
            del cache_store

        def run_cases_iter(self, **kwargs):
            captured["execution_mode"] = kwargs.get("execution_mode")
            captured["node_name"] = kwargs.get("node_name")
            captured["debug"] = kwargs.get("debug")
            yield SimpleNamespace(
                case_id="c1",
                error=None,
                result=SimpleNamespace(
                    final_state={
                        "estimated_costs": {
                            "chunk": CostEstimate(cost=0.2, input_tokens=5, output_tokens=1)
                        }
                    }
                ),
            )

    monkeypatch.setattr(main_module, "create_dataset_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(main_module, "CachedNodeRunner", _Runner)

    main_module.estimate(
        node_name="grounding",
        input="dummy.json",
        split="train",
        start=0,
        end=None,
        limit=1,
        adapter="auto",
        hf_config=None,
        hf_revision=None,
        judge_model=DEFAULT_PRIMARY_LLM,
        llm_model=[],
        llm_fallback=[],
        continue_on_error=True,
        max_workers=1,
        max_in_flight=None,
        force=False,
        no_cache=True,
        cache_dir=None,
        debug=True,
    )

    assert captured["execution_mode"] == "estimate"
    assert captured["node_name"] == "grounding"
    assert captured["debug"] is True


def test_estimate_command_filters_ineligible_cases_for_grounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_case_ids: list[str] = []

    class _Adapter:
        def iter_cases(self, split: str = "train", limit: int | None = None):
            del split, limit
            yield {"case_id": "no-context", "generation": "hello"}
            yield {"case_id": "with-context", "generation": "hello", "context": "ctx"}

    class _Runner:
        def __init__(self, cache_store):
            del cache_store

        def run_cases_iter(self, **kwargs):
            for case in kwargs["cases"]:
                captured_case_ids.append(case["case_id"])
                yield SimpleNamespace(
                    case_id=case["case_id"],
                    error=None,
                    result=SimpleNamespace(final_state={"estimated_costs": {}}),
                )

    monkeypatch.setattr(main_module, "create_dataset_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(main_module, "CachedNodeRunner", _Runner)

    main_module.estimate(
        node_name="grounding",
        input="dummy.json",
        split="train",
        start=0,
        end=None,
        limit=10,
        adapter="auto",
        hf_config=None,
        hf_revision=None,
        judge_model=DEFAULT_PRIMARY_LLM,
        llm_model=[],
        llm_fallback=[],
        continue_on_error=True,
        max_workers=1,
        max_in_flight=None,
        force=False,
        no_cache=True,
        cache_dir=None,
    )

    assert captured_case_ids == ["with-context"]


def test_estimate_command_uses_all_selected_records_as_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_total_selected_cases: list[int] = []

    class _Adapter:
        def iter_cases(self, split: str = "train", limit: int | None = None):
            del split, limit
            yield {"case_id": "no-context", "generation": "hello"}
            yield {"case_id": "with-context", "generation": "hello", "context": "ctx"}

    class _Runner:
        def __init__(self, cache_store):
            del cache_store

        def run_cases_iter(self, **kwargs):
            for case in kwargs["cases"]:
                yield SimpleNamespace(
                    case_id=case["case_id"],
                    error=None,
                    result=SimpleNamespace(
                        final_state={
                            "estimated_costs": {},
                            "inputs": SimpleNamespace(has_generation=True, has_context=True),
                        },
                        executed_nodes=["scan"],
                        cached_nodes=[],
                    ),
                )

    def _fake_collect_estimate_rows(**kwargs):
        captured_total_selected_cases.append(kwargs["total_selected_cases"])
        return []

    monkeypatch.setattr(main_module, "create_dataset_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(main_module, "CachedNodeRunner", _Runner)
    monkeypatch.setattr("ng_cli.estimate._collect_estimate_rows", _fake_collect_estimate_rows)

    main_module.estimate(
        node_name="grounding",
        input="dummy.json",
        split="train",
        start=0,
        end=None,
        limit=10,
        adapter="auto",
        hf_config=None,
        hf_revision=None,
        judge_model=DEFAULT_PRIMARY_LLM,
        llm_model=[],
        llm_fallback=[],
        continue_on_error=True,
        max_workers=1,
        max_in_flight=None,
        force=False,
        no_cache=True,
        cache_dir=None,
    )

    assert captured_total_selected_cases == [2]


def test_run_command_filters_ineligible_cases_for_grounding(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_case_ids: list[str] = []

    class _Adapter:
        def iter_cases(self, split: str = "train", limit: int | None = None):
            del split, limit
            yield {"case_id": "no-context", "generation": "hello"}
            yield {"case_id": "with-context", "generation": "hello", "context": "ctx"}

    class _Runner:
        def __init__(self, cache_store):
            del cache_store

        def run_cases_iter(self, **kwargs):
            for case in kwargs["cases"]:
                captured_case_ids.append(case["case_id"])
                yield SimpleNamespace(
                    case_id=case["case_id"],
                    error=None,
                    result=SimpleNamespace(
                        case_id=case["case_id"],
                        executed_nodes=["scan"],
                        cached_nodes=[],
                        final_state={},
                    ),
                )

    monkeypatch.setattr(main_module, "create_dataset_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(main_module, "CachedNodeRunner", _Runner)

    main_module.run(
        node_name="grounding",
        input="dummy.json",
        split="train",
        start=0,
        end=None,
        limit=10,
        adapter="auto",
        hf_config=None,
        hf_revision=None,
        judge_model=DEFAULT_PRIMARY_LLM,
        llm_model=[],
        llm_fallback=[],
        web_search=False,
        evidence_threshold=0.75,
        yes=False,
        continue_on_error=True,
        max_workers=1,
        max_in_flight=None,
        force=False,
        no_cache=True,
        cache_dir=None,
        output_dir=None,
    )

    assert captured_case_ids == ["with-context"]


def test_run_command_does_not_prefetch_all_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_case_ids: list[str] = []

    class _Adapter:
        def iter_cases(self, split: str = "train", limit: int | None = None):
            del split, limit
            yield {"case_id": "c1", "generation": "hello", "context": "ctx"}
            raise AssertionError("run() should not prefetch additional records")

    class _Runner:
        def __init__(self, cache_store):
            del cache_store

        def run_cases_iter(self, **kwargs):
            first = next(iter(kwargs["cases"]))
            captured_case_ids.append(first["case_id"])
            yield SimpleNamespace(
                case_id=first["case_id"],
                error=None,
                result=SimpleNamespace(
                    case_id=first["case_id"],
                    executed_nodes=["scan"],
                    cached_nodes=[],
                    final_state={},
                ),
            )

    monkeypatch.setattr(main_module, "create_dataset_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(main_module, "CachedNodeRunner", _Runner)

    main_module.run(
        node_name="grounding",
        input="dummy.json",
        split="train",
        start=0,
        end=None,
        limit=None,
        adapter="auto",
        hf_config=None,
        hf_revision=None,
        judge_model=DEFAULT_PRIMARY_LLM,
        llm_model=[],
        llm_fallback=[],
        web_search=False,
        evidence_threshold=0.75,
        yes=False,
        continue_on_error=True,
        max_workers=1,
        max_in_flight=None,
        force=False,
        no_cache=True,
        cache_dir=None,
        output_dir=None,
    )

    assert captured_case_ids == ["c1"]
