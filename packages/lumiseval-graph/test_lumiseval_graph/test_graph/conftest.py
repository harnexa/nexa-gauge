from __future__ import annotations

from collections.abc import Callable
import builtins
import importlib
from pathlib import Path

import pytest

from lumiseval_core.types import MetricCategory, MetricResult


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
        module = importlib.import_module("lumiseval_graph.graph")
        return module
    finally:
        if hasattr(builtins, "Path"):
            delattr(builtins, "Path")
