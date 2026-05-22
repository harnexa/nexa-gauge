"""Best-effort identity probe for self-hosted OpenAI-compatible servers.

The cache fingerprint needs *something* that distinguishes one model loaded
behind ``http://localhost:8080/v1`` from another loaded at the same URL on a
subsequent run. The OpenAI-compatible ``GET /models`` endpoint returns the id
of whatever the server is currently serving — exactly that distinguishing bit.

Failures here are non-fatal: the caller falls back to the unmodified routing
hash, and the user can always pass ``--llm-model`` to supply an explicit tag.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Optional

from ng_graph.llm.config import normalize_api_base

_DEFAULT_TIMEOUT = 3.0

_lock = threading.Lock()
_cache: dict[str, Optional[str]] = {}


def resolve_host_model_identity(
    api_base: Optional[str], *, timeout: float = _DEFAULT_TIMEOUT
) -> Optional[str]:
    """Return a stable identifier for the model(s) served by ``api_base``.

    Joins sorted ``data[*].id`` values returned by ``GET {api_base}/models``.
    The result is memoised per normalised ``api_base`` for the life of the
    process so we make at most one HTTP call per endpoint. Returns ``None`` on
    any failure (no api_base, network error, non-200, malformed body) so the
    caller can decide on a fallback.
    """
    normalized = normalize_api_base(api_base)
    if not normalized:
        return None

    with _lock:
        if normalized in _cache:
            return _cache[normalized]

    identity = _probe(normalized, timeout=timeout)

    with _lock:
        _cache[normalized] = identity
    return identity


def _probe(api_base: str, *, timeout: float) -> Optional[str]:
    url = f"{api_base}/models"
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            body = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None

    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None

    ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id.strip():
            ids.append(item_id.strip())

    if not ids:
        return None
    return "|".join(sorted(set(ids)))


def reset_cache() -> None:
    """Clear the in-process probe cache. Intended for tests."""
    with _lock:
        _cache.clear()
