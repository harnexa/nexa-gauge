from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ng_graph.llm import host_model


@pytest.fixture(autouse=True)
def _reset_cache():
    host_model.reset_cache()
    yield
    host_model.reset_cache()


def _fake_urlopen(payload: Any, status: int = 200):
    body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


def test_resolve_host_model_identity_joins_sorted_ids():
    payload = {"data": [{"id": "model-b"}, {"id": "model-a"}]}
    with patch(
        "ng_graph.llm.host_model.urllib.request.urlopen", return_value=_fake_urlopen(payload)
    ):
        identity = host_model.resolve_host_model_identity("http://localhost:8080/v1")
    assert identity == "model-a|model-b"


def test_resolve_host_model_identity_returns_none_on_network_error():
    with patch(
        "ng_graph.llm.host_model.urllib.request.urlopen", side_effect=OSError("connection refused")
    ):
        identity = host_model.resolve_host_model_identity("http://localhost:8080/v1")
    assert identity is None


def test_resolve_host_model_identity_returns_none_on_malformed_body():
    cm = MagicMock()
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b"not json"
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    with patch("ng_graph.llm.host_model.urllib.request.urlopen", return_value=cm):
        identity = host_model.resolve_host_model_identity("http://localhost:8080/v1")
    assert identity is None


def test_resolve_host_model_identity_returns_none_when_data_missing():
    with patch(
        "ng_graph.llm.host_model.urllib.request.urlopen",
        return_value=_fake_urlopen({"object": "list"}),
    ):
        assert host_model.resolve_host_model_identity("http://localhost:8080/v1") is None


def test_resolve_host_model_identity_caches_per_api_base():
    payload = {"data": [{"id": "model-a"}]}
    mock_urlopen = MagicMock(return_value=_fake_urlopen(payload))
    with patch("ng_graph.llm.host_model.urllib.request.urlopen", mock_urlopen):
        first = host_model.resolve_host_model_identity("http://localhost:8080/v1")
        second = host_model.resolve_host_model_identity("http://localhost:8080/v1/")
        third = host_model.resolve_host_model_identity("http://localhost:8081/v1")
    assert first == second == "model-a"
    assert third == "model-a"
    # Two distinct api_bases → exactly two probes; the trailing-slash variant must hit cache.
    assert mock_urlopen.call_count == 2


def test_resolve_host_model_identity_returns_none_for_missing_api_base():
    assert host_model.resolve_host_model_identity(None) is None
    assert host_model.resolve_host_model_identity("") is None
