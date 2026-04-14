"""Scenario tests for the declarative report aggregate function.

Each test constructs a mock EvalCase state with real Pydantic objects for nodes
that "ran" and None for nodes that didn't, then verifies the report output shape.

Scenarios:
  1. Only grounding runs
  2. Only relevance runs
  3. Only geval runs
  4. Only redteam runs
  5. Grounding + relevance run
  6. Full eval (all nodes)
"""

from __future__ import annotations

from typing import Any

import pytest
from lumoseval_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    GevalMetrics,
    GroundingMetrics,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    RedteamMetrics,
    ReferenceMetrics,
    RelevanceMetrics,
)
from lumoseval_graph.nodes import report

# ---------------------------------------------------------------------------
# Fixture builders — each returns a realistic Pydantic object
# ---------------------------------------------------------------------------


def _item(text: str, tokens: float = 3.0) -> Item:
    return Item(text=text, tokens=tokens)


def _cost(cost: float = 0.01, inp: float = 100.0, out: float = 20.0) -> CostEstimate:
    return CostEstimate(cost=cost, input_tokens=inp, output_tokens=out)


def _metric(name: str, score: float = 0.85) -> MetricResult:
    return MetricResult(
        name=name,
        category=MetricCategory.ANSWER,
        score=score,
        result=[{"detail": "ok"}],
    )


def _inputs() -> Inputs:
    return Inputs(
        case_id="case-1",
        generation=_item("Generated answer.", tokens=5.0),
        question=_item("What is X?", tokens=3.0),
        context=_item("X is a thing.", tokens=4.0),
        reference=_item("X", tokens=1.0),
    )


def _chunks() -> ChunkArtifacts:
    return ChunkArtifacts(
        chunks=[
            Chunk(
                index=0,
                item=_item("Generated answer.", tokens=5.0),
                char_start=0,
                char_end=18,
                sha256="abc123",
            )
        ],
        cost=_cost(0.0, 0.0, 0.0),
    )


def _claims() -> ClaimArtifacts:
    return ClaimArtifacts(
        claims=[
            Claim(item=_item("X is true.", tokens=3.0), source_chunk_index=0, confidence=0.9)
        ],
        cost=_cost(0.001, 10.0, 3.0),
    )


def _grounding() -> GroundingMetrics:
    return GroundingMetrics(
        metrics=[_metric("grounding", 1.0)],
        cost=_cost(0.002, 20.0, 2.0),
    )


def _relevance() -> RelevanceMetrics:
    return RelevanceMetrics(
        metrics=[_metric("answer_relevancy", 0.9)],
        cost=_cost(0.003, 30.0, 5.0),
    )


def _redteam() -> RedteamMetrics:
    return RedteamMetrics(
        metrics=[_metric("vulnerability_prompt_injection", 0.2)],
        cost=_cost(0.004, 40.0, 8.0),
    )


def _geval() -> GevalMetrics:
    return GevalMetrics(
        metrics=[_metric("geval_coherence", 0.7)],
        cost=_cost(0.005, 50.0, 10.0),
    )


def _reference() -> ReferenceMetrics:
    return ReferenceMetrics(
        metrics=[_metric("rouge_l", 0.6)],
        cost=_cost(0.006, 60.0, 12.0),
    )


def _make_state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal EvalCase-like state dict.

    All artifact keys default to None (node didn't run).
    Pass keyword args to override specific keys.
    """
    base: dict[str, Any] = {
        "target_node": "eval",
        "inputs": _inputs(),
        "generation_chunk": None,
        "generation_claims": None,
        "generation_dedup_claims": None,
        "grounding_metrics": None,
        "relevance_metrics": None,
        "redteam_metrics": None,
        "geval_metrics": None,
        "reference_metrics": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helpers for assertions
# ---------------------------------------------------------------------------

ALWAYS_PRESENT = {"target_node", "input"}


def _assert_sections(result: dict, expected_extra: set[str]) -> None:
    """Assert that the report has exactly the always-present keys plus expected_extra."""
    assert set(result.keys()) == ALWAYS_PRESENT | expected_extra


def _assert_input(result: dict) -> None:
    """Assert the input section is correctly populated."""
    inp = result["input"]
    assert inp["case_id"] == "case-1"
    assert inp["generation"] == "Generated answer."
    assert inp["question"] == "What is X?"
    assert inp["context"] == "X is a thing."
    assert inp["reference"] == "X"


def _assert_cost(cost_dict: dict, expected_cost: float, expected_inp: float, expected_out: float) -> None:
    assert cost_dict["cost"] == pytest.approx(expected_cost)
    assert cost_dict["input_tokens"] == pytest.approx(expected_inp)
    assert cost_dict["output_tokens"] == pytest.approx(expected_out)


# ---------------------------------------------------------------------------
# Scenario 1: Only grounding runs
# ---------------------------------------------------------------------------


def test_only_grounding() -> None:
    """Verify report aggregate includes only grounding-related sections when grounding branch ran.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_aggregate.py::test_only_grounding
    """
    state = _make_state(
        generation_chunk=_chunks(),
        generation_claims=_claims(),
        generation_dedup_claims=_claims(),
        grounding_metrics=_grounding(),
    )
    result = report.aggregate(state=state)

    _assert_sections(result, {"chunks", "claims", "claims_unique", "grounding"})
    _assert_input(result)

    # Chunks
    assert result["chunks"]["text"] == ["Generated answer."]
    _assert_cost(result["chunks"]["cost"], 0.0, 0.0, 0.0)

    # Claims
    assert result["claims"]["text"] == ["X is true."]
    _assert_cost(result["claims"]["cost"], 0.001, 10.0, 3.0)

    # Claims unique
    assert result["claims_unique"]["text"] == ["X is true."]

    # Grounding — metrics is now a list of dicts
    assert result["grounding"]["metrics"] == [
        {"name": "grounding", "score": 1.0, "result": [{"detail": "ok"}]},
    ]
    _assert_cost(result["grounding"]["cost"], 0.002, 20.0, 2.0)

    # Absent sections
    assert "relevance" not in result
    assert "redteam" not in result
    assert "geval" not in result
    assert "reference" not in result


# ---------------------------------------------------------------------------
# Scenario 2: Only relevance runs
# ---------------------------------------------------------------------------


def test_only_relevance() -> None:
    """Verify report aggregate includes only relevance-related sections when relevance branch ran.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_aggregate.py::test_only_relevance
    """
    state = _make_state(
        generation_chunk=_chunks(),
        generation_claims=_claims(),
        generation_dedup_claims=_claims(),
        relevance_metrics=_relevance(),
    )
    result = report.aggregate(state=state)

    _assert_sections(result, {"chunks", "claims", "claims_unique", "relevance"})
    _assert_input(result)

    assert result["relevance"]["metrics"] == [
        {"name": "answer_relevancy", "score": 0.9, "result": [{"detail": "ok"}]},
    ]
    _assert_cost(result["relevance"]["cost"], 0.003, 30.0, 5.0)

    assert "grounding" not in result
    assert "redteam" not in result
    assert "geval" not in result
    assert "reference" not in result


# ---------------------------------------------------------------------------
# Scenario 3: Only geval runs
# ---------------------------------------------------------------------------


def test_only_geval() -> None:
    """Verify report aggregate includes only GEval section when only GEval metrics are present.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_aggregate.py::test_only_geval
    """
    state = _make_state(
        geval_metrics=_geval(),
    )
    result = report.aggregate(state=state)

    _assert_sections(result, {"geval"})
    _assert_input(result)

    assert result["geval"]["metrics"] == [
        {"name": "geval_coherence", "score": 0.7, "result": [{"detail": "ok"}]},
    ]
    _assert_cost(result["geval"]["cost"], 0.005, 50.0, 10.0)

    assert "chunks" not in result
    assert "claims" not in result
    assert "grounding" not in result
    assert "relevance" not in result
    assert "redteam" not in result
    assert "reference" not in result


# ---------------------------------------------------------------------------
# Scenario 4: Only redteam runs
# ---------------------------------------------------------------------------


def test_only_redteam() -> None:
    """Verify report aggregate includes only redteam section when only redteam metrics are present.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_aggregate.py::test_only_redteam
    """
    state = _make_state(
        redteam_metrics=_redteam(),
    )
    result = report.aggregate(state=state)

    _assert_sections(result, {"redteam"})
    _assert_input(result)

    assert result["redteam"]["metrics"] == [
        {"name": "vulnerability_prompt_injection", "score": 0.2, "result": [{"detail": "ok"}]},
    ]
    _assert_cost(result["redteam"]["cost"], 0.004, 40.0, 8.0)

    assert "chunks" not in result
    assert "claims" not in result
    assert "grounding" not in result
    assert "relevance" not in result
    assert "geval" not in result
    assert "reference" not in result


# ---------------------------------------------------------------------------
# Scenario 5: Grounding + relevance run together
# ---------------------------------------------------------------------------


def test_grounding_and_relevance() -> None:
    """Verify report aggregate includes both grounding and relevance sections for combined branch outputs.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_aggregate.py::test_grounding_and_relevance
    """
    state = _make_state(
        generation_chunk=_chunks(),
        generation_claims=_claims(),
        generation_dedup_claims=_claims(),
        grounding_metrics=_grounding(),
        relevance_metrics=_relevance(),
    )
    result = report.aggregate(state=state)
    from lumoseval_core.utils import pprint_model
    pprint_model(result)

    _assert_sections(result, {"chunks", "claims", "claims_unique", "grounding", "relevance"})
    _assert_input(result)

    assert result["grounding"]["metrics"] == [
        {"name": "grounding", "score": 1.0, "result": [{"detail": "ok"}]},
    ]
    assert result["relevance"]["metrics"] == [
        {"name": "answer_relevancy", "score": 0.9, "result": [{"detail": "ok"}]},
    ]
    _assert_cost(result["grounding"]["cost"], 0.002, 20.0, 2.0)
    _assert_cost(result["relevance"]["cost"], 0.003, 30.0, 5.0)

    assert "redteam" not in result
    assert "geval" not in result
    assert "reference" not in result


# ---------------------------------------------------------------------------
# Scenario 6: Full eval — all nodes present
# ---------------------------------------------------------------------------


def test_full_eval() -> None:
    """Verify report aggregate exposes all expected sections when full eval artifacts are present.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_aggregate.py::test_full_eval
    """
    state = _make_state(
        generation_chunk=_chunks(),
        generation_claims=_claims(),
        generation_dedup_claims=_claims(),
        grounding_metrics=_grounding(),
        relevance_metrics=_relevance(),
        redteam_metrics=_redteam(),
        geval_metrics=_geval(),
        reference_metrics=_reference(),
    )
    result = report.aggregate(state=state)

    _assert_sections(
        result,
        {"chunks", "claims", "claims_unique", "grounding", "relevance", "redteam", "geval", "reference"},
    )
    _assert_input(result)

    # Verify every section has the expected structure
    assert result["target_node"] == "eval"
    assert isinstance(result["chunks"]["text"], list)
    assert isinstance(result["claims"]["text"], list)
    assert isinstance(result["claims_unique"]["text"], list)

    for section in ("grounding", "relevance", "redteam", "geval", "reference"):
        assert "metrics" in result[section]
        assert "cost" in result[section]
        cost = result[section]["cost"]
        assert "cost" in cost
        assert "input_tokens" in cost
        assert "output_tokens" in cost
