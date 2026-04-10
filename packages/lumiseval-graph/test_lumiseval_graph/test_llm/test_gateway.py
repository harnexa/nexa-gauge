# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_llm/test_gateway.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_llm/test_gateway.py::test_get_llm_cache_key_changes_with_runtime_override
# uv run pytest -s -k "llm and gateway" packages/lumiseval-graph/test_lumiseval_graph/test_llm/test_gateway.py

from pydantic import BaseModel

from lumiseval_graph.llm.gateway import _cache, get_llm


class _Schema(BaseModel):
    value: str


def test_get_llm_cache_key_changes_with_runtime_override() -> None:
    _cache.clear()

    llm_a = get_llm(
        "relevance",
        _Schema,
        default_model="default-model",
        llm_overrides={"models": {"relevance": "model-a"}},
    )
    llm_a_again = get_llm(
        "relevance",
        _Schema,
        default_model="default-model",
        llm_overrides={"models": {"relevance": "model-a"}},
    )
    llm_b = get_llm(
        "relevance",
        _Schema,
        default_model="default-model",
        llm_overrides={"models": {"relevance": "model-b"}},
    )

    assert llm_a is llm_a_again
    assert llm_a is not llm_b


def test_get_llm_cache_key_includes_temp_and_fallback_overrides() -> None:
    _cache.clear()

    baseline = get_llm("claims", _Schema, default_model="default-model")
    with_temp = get_llm(
        "claims",
        _Schema,
        default_model="default-model",
        llm_overrides={"temperatures": {"claims": 0.4}},
    )
    with_fallback = get_llm(
        "claims",
        _Schema,
        default_model="default-model",
        llm_overrides={"fallback_models": {"claims": "backup-model"}},
    )

    assert baseline is not with_temp
    assert baseline is not with_fallback
    assert with_temp is not with_fallback
