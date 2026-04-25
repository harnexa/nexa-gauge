from __future__ import annotations

import builtins
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ng_core.types import MetricCategory, MetricResult


@pytest.fixture
def make_metric() -> Callable[[str, float, MetricCategory], MetricResult]:
    def _make_metric(name: str, score: float, category: MetricCategory) -> MetricResult:
        return MetricResult(name=name, score=score, category=category)

    return _make_metric


@pytest.fixture
def graph_module():
    # graph.py currently evaluates `Path` in annotations at import time.
    builtins.Path = Path
    try:
        module = importlib.import_module("ng_graph.graph")
        return module
    finally:
        if hasattr(builtins, "Path"):
            delattr(builtins, "Path")


def make_fake_get_judge_model(
    captured: dict[str, Any], model_suffix: str = "model"
) -> Callable[[str, str, Any], str]:
    """Factory for creating _fake_get_judge_model that captures single call to dict."""

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides: Any = None) -> str:
        captured["resolved_node_name"] = node_name
        captured["resolved_overrides"] = llm_overrides
        return f"resolved-{node_name}-{model_suffix}"

    return _fake_get_judge_model


def make_fake_get_judge_model_multi(
    captured: list[tuple[str, Any]]
) -> Callable[[str, str, Any], str]:
    """Factory for creating _fake_get_judge_model that captures multiple calls to list."""

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides: Any = None) -> str:
        captured.append((node_name, llm_overrides))
        return f"resolved-{node_name}"

    return _fake_get_judge_model
