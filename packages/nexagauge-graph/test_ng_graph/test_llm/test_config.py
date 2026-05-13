# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_llm/test_config.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_llm/test_config.py::test_get_judge_model_prefers_runtime_over_env
# uv run pytest -s -k "llm and config" packages/nexagauge-graph/test_ng_graph/test_llm/test_config.py

import pytest
from ng_graph.llm.config import get_judge_model, get_node_config, normalize_node_name


def test_normalize_node_name() -> None:
    assert normalize_node_name("claims") == "claims"
    assert normalize_node_name("refiner") == "refiner"
    assert normalize_node_name("Refiner") == "refiner"


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


def test_get_node_config_api_base_prefers_runtime_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://global.example/v1")
    monkeypatch.setenv("LLM_RELEVANCE_API_BASE", "http://node.example/v1")

    cfg = get_node_config(
        "relevance",
        llm_overrides={"api_bases": {"relevance": "http://runtime.example/v1/"}},
    )
    assert cfg.api_base == "http://runtime.example/v1"


def test_get_node_config_api_key_prefers_node_over_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "global-key")
    monkeypatch.setenv("LLM_GROUNDING_API_KEY", "node-key")

    cfg = get_node_config("grounding")
    assert cfg.api_key == "node-key"
