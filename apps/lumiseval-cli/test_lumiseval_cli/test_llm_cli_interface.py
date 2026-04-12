from __future__ import annotations

import pytest

from lumiseval_cli.main import (
    DEFAULT_FALLBACK_LLM,
    DEFAULT_PRIMARY_LLM,
    _parse_model_overrides,
    _resolve_runtime_llm_overrides,
)


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
    with pytest.raises(ValueError, match="Conflicting global model values from --model and --llm-model"):
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
