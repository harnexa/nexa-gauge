from __future__ import annotations

from lumoseval_core.types import (
    Chunk,
    ChunkArtifacts,
    CostEstimate,
    Inputs,
    Item,
)
from lumoseval_graph.nodes import report

# ---------------------------------------------------------------------------
# _extract_path tests (unchanged function, kept as-is)
# ---------------------------------------------------------------------------


def test_extract_path_scalar() -> None:
    """Verify _extract_path resolves a simple scalar path.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_extract_path_scalar
    """
    data = {"inputs": {"case_id": "case-1"}}
    assert report._extract_path(data, "inputs.case_id") == "case-1"


def test_extract_path_list_wildcard() -> None:
    """Verify _extract_path supports list wildcard traversal for nested fields.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_extract_path_list_wildcard
    """
    data = {
        "claims": [
            {"item": {"text": "A", "tokens": 1.0}},
            {"item": {"text": "B", "tokens": 2.0}},
        ]
    }
    assert report._extract_path(data, "claims[*].item.text") == ["A", "B"]
    assert report._extract_path(data, "claims[*].item.tokens") == [1.0, 2.0]


def test_extract_path_missing_returns_nullish_defaults() -> None:
    """Verify _extract_path returns []/None defaults for missing wildcard and scalar paths.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_extract_path_missing_returns_nullish_defaults
    """
    data = {"claims": None}
    assert report._extract_path(data, "claims[*].item.text") == []
    assert report._extract_path(data, "claims.item.text") is None


# ---------------------------------------------------------------------------
# resolve_path tests
# ---------------------------------------------------------------------------


def test_resolve_path_scalar_from_state() -> None:
    """Verify resolve_path reads a top-level scalar from state.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_resolve_path_scalar_from_state
    """
    state = {"target_node": "eval"}
    assert report.resolve_path(state, "target_node") == "eval"


def test_resolve_path_nested_pydantic() -> None:
    """Verify resolve_path traverses nested attributes on Pydantic models.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_resolve_path_nested_pydantic
    """
    state = {
        "inputs": Inputs(
            case_id="case-1",
            generation=Item(text="gen text", tokens=5.0),
        ),
    }
    assert report.resolve_path(state, "inputs.case_id") == "case-1"
    assert report.resolve_path(state, "inputs.generation.text") == "gen text"


def test_resolve_path_none_intermediate() -> None:
    """Verify resolve_path returns None when an intermediate segment is None.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_resolve_path_none_intermediate
    """
    state = {"inputs": None}
    assert report.resolve_path(state, "inputs.generation.text") is None


def test_resolve_path_wildcard_on_pydantic() -> None:
    """Verify resolve_path handles wildcard extraction from Pydantic list fields.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_resolve_path_wildcard_on_pydantic
    """
    state = {
        "generation_chunk": ChunkArtifacts(
            chunks=[
                Chunk(index=0, item=Item(text="A", tokens=1.0), char_start=0, char_end=1, sha256="a"),
                Chunk(index=1, item=Item(text="B", tokens=2.0), char_start=1, char_end=2, sha256="b"),
            ],
            cost=CostEstimate(cost=0.01, input_tokens=10.0, output_tokens=5.0),
        )
    }
    assert report.resolve_path(state, "generation_chunk.chunks[*].item.text") == ["A", "B"]
    assert report.resolve_path(state, "generation_chunk.cost.cost") == 0.01


# ---------------------------------------------------------------------------
# resolve_section tests
# ---------------------------------------------------------------------------


def test_resolve_section_string_spec() -> None:
    """Verify resolve_section returns scalar values for string visibility specs.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_resolve_section_string_spec
    """
    state = {"target_node": "grounding"}
    assert report.resolve_section(state, "target_node") == "grounding"


def test_resolve_section_nested_dict_spec() -> None:
    """Verify resolve_section builds nested dictionaries from declarative visibility specs.

    Run: uv run pytest -s packages/lumos-graph/test_lumos_graph/test_nodes/test_report_projection.py::test_resolve_section_nested_dict_spec
    """
    state = {
        "target_node": "eval",
        "inputs": Inputs(
            case_id="case-1",
            generation=Item(text="answer", tokens=3.0),
        ),
    }
    spec = {
        "node": "target_node",
        "details": {
            "id": "inputs.case_id",
            "gen": "inputs.generation.text",
        },
    }
    result = report.resolve_section(state, spec)
    assert result == {
        "node": "eval",
        "details": {
            "id": "case-1",
            "gen": "answer",
        },
    }
