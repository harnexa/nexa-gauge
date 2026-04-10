# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_llm/test_config.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_llm/test_config.py::test_get_judge_model_prefers_runtime_over_env
# uv run pytest -s -k "llm and config" packages/lumiseval-graph/test_lumiseval_graph/test_llm/test_config.py

import pytest

from lumiseval_graph.llm.config import get_judge_model, normalize_node_name


def test_normalize_node_name_aliases() -> None:
    assert normalize_node_name("claims") == "claims"
    assert normalize_node_name("node_claims") == "claims"
    assert normalize_node_name("generation_claims") == "claims"
    assert normalize_node_name("node_grounding") == "grounding"



def test_get_judge_model_prefers_runtime_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CLAIMS_MODEL", "env-claims-model")

    llm_overrides = {
        "models": {"claims": "runtime-claims-model"},
    }

    model = get_judge_model("claims", "default-model", llm_overrides=llm_overrides)
    assert model == "runtime-claims-model"



def test_get_judge_model_uses_env_before_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_RELEVANCE_MODEL", "env-relevance-model")

    model = get_judge_model("relevance", "default-model")
    assert model == "env-relevance-model"



