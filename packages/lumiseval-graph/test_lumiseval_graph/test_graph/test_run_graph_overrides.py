# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_run_graph_overrides.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_run_graph_overrides.py::test_normalize_runtime_overrides_parses_and_normalizes
# uv run pytest -s -k "run_graph_overrides" packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_run_graph_overrides.py

from __future__ import annotations

from lumiseval_core.types import ClaimArtifacts, CostEstimate
from lumiseval_graph.llm import normalize_runtime_overrides


def test_normalize_runtime_overrides_parses_and_normalizes() -> None:
    """normalize_runtime_overrides canonicalises keys and values."""
    overrides = normalize_runtime_overrides({
        "models": {"relevance": "openai/gpt-4.1"},
        "fallback_models": {"relevance": "openai/gpt-4o-mini"},
        "temperatures": {"relevance": 0.2},
    })

    assert overrides["models"]["relevance"] == "openai/gpt-4.1"
    assert overrides["fallback_models"]["relevance"] == "openai/gpt-4o-mini"
    assert overrides["temperatures"]["relevance"] == 0.2


def test_graph_forwards_llm_overrides_to_nodes(graph_module, monkeypatch) -> None:
    """llm_overrides in the initial invoke payload reach every LLM-calling node."""
    captured: list[tuple[str, object]] = []

    # Replace LLM/metric-touching node functions with stubs that record the
    # llm_overrides they see in state, then return minimal valid output.
    def _fake_claims(state: dict) -> dict:
        captured.append(("claims", state.get("llm_overrides")))
        return {"generation_claims": ClaimArtifacts(claims=[], cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None))}

    def _fake_grounding(state: dict) -> dict:
        captured.append(("grounding", state.get("llm_overrides")))
        return {"grounding_metrics": None}

    def _fake_relevance(state: dict) -> dict:
        captured.append(("relevance", state.get("llm_overrides")))
        return {"relevance_metrics": None}

    def _fake_redteam(state: dict) -> dict:
        captured.append(("redteam", state.get("llm_overrides")))
        return {"redteam_metrics": None}

    def _fake_geval_steps(state: dict) -> dict:
        captured.append(("geval_steps", state.get("llm_overrides")))
        return {"geval_steps": None}

    def _fake_geval(state: dict) -> dict:
        captured.append(("geval", state.get("llm_overrides")))
        return {"geval_metrics": None}

    monkeypatch.setattr(graph_module, "node_generation_claims", _fake_claims)
    monkeypatch.setattr(graph_module, "node_grounding", _fake_grounding)
    monkeypatch.setattr(graph_module, "node_relevance", _fake_relevance)
    monkeypatch.setattr(graph_module, "node_redteam", _fake_redteam)
    monkeypatch.setattr(graph_module, "node_geval_steps", _fake_geval_steps)
    monkeypatch.setattr(graph_module, "node_geval", _fake_geval)

    llm_overrides = {"models": {"claims": "openai/gpt-4o-mini"}}
    app = graph_module.build_graph().compile()
    final_state = app.invoke({
        "record": {"case_id": "t1", "generation": "The sky is blue."},
        "llm_overrides": llm_overrides,
    })

    assert captured, "No LLM-calling nodes were reached"
    assert all(ov == llm_overrides for _, ov in captured), (
        f"Expected every node to receive {llm_overrides!r}; got {captured}"
    )
    # llm_overrides must also survive in the final pipeline state
    assert final_state.get("llm_overrides") == llm_overrides
