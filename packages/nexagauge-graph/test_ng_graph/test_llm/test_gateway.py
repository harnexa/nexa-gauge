# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_llm/test_gateway.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_llm/test_gateway.py::test_get_llm_cache_key_changes_with_runtime_override
# uv run pytest -s -k "llm and gateway" packages/nexagauge-graph/test_ng_graph/test_llm/test_gateway.py

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

import pytest
from ng_graph.llm.gateway import (
    StructuredLLM,
    _cache,
    get_llm,
    get_llm_concurrency,
    set_llm_concurrency,
)
from pydantic import BaseModel


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


def test_get_llm_cache_key_changes_with_api_base_override() -> None:
    _cache.clear()
    llm_a = get_llm(
        "claims",
        _Schema,
        default_model="default-model",
        llm_overrides={"api_bases": {"claims": "http://a.local/v1"}},
    )
    llm_b = get_llm(
        "claims",
        _Schema,
        default_model="default-model",
        llm_overrides={"api_bases": {"claims": "http://b.local/v1"}},
    )
    assert llm_a is not llm_b


def test_set_llm_concurrency_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        set_llm_concurrency(0)


def test_invoke_respects_global_llm_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    _cache.clear()
    previous_limit = get_llm_concurrency()
    set_llm_concurrency(2)

    in_flight = 0
    max_in_flight = 0
    lock = Lock()

    def _fake_call(self: StructuredLLM, messages, model):  # type: ignore[no-untyped-def]
        del messages, model
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"ok"}'))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(StructuredLLM, "_call", _fake_call)

    llm = get_llm("claims", _Schema, default_model="default-model")
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(
                pool.map(
                    lambda _: llm.invoke([{"role": "user", "content": "ping"}]),
                    range(6),
                )
            )
        assert all(response["parsed"] is not None for response in responses)
        assert max_in_flight <= 2
    finally:
        set_llm_concurrency(previous_limit)


def test_invoke_passes_api_base_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _cache.clear()
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"ok"}'))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("ng_graph.llm.gateway.litellm.completion", _fake_completion)

    llm = get_llm(
        "claims",
        _Schema,
        default_model="openai/local-model",
        llm_overrides={
            "api_bases": {"claims": "http://localhost:8080/v1"},
            "api_keys": {"claims": "secret"},
        },
    )
    llm.invoke([{"role": "user", "content": "hello"}])

    assert captured["api_base"] == "http://localhost:8080/v1"
    assert captured["api_key"] == "secret"


def test_invoke_uses_local_api_key_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _cache.clear()
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"ok"}'))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("ng_graph.llm.gateway.litellm.completion", _fake_completion)

    llm = get_llm(
        "claims",
        _Schema,
        default_model="openai/local-model",
        llm_overrides={"api_bases": {"claims": "http://127.0.0.1:8080/v1"}},
    )
    llm.invoke([{"role": "user", "content": "hello"}])

    assert captured["api_key"] == "local"


def test_invoke_retries_without_response_format_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cache.clear()
    call_count = {"n": 0}

    def _fake_completion(**kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("response_format json_schema not supported")
        assert "response_format" not in kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"ok"}'))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("ng_graph.llm.gateway.litellm.completion", _fake_completion)

    llm = get_llm("claims", _Schema, default_model="default-model")
    response = llm.invoke([{"role": "user", "content": "ping"}])

    assert response["parsed"] is not None
    assert call_count["n"] == 2
