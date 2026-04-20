"""Signature helpers + cache key builder for GEval step artifacts.

GEval step artifacts persist via the universal ``NodeCacheBackend``
(``ng_core.cache.CacheStore``) under opaque keys produced by
:func:`build_geval_artifact_cache_key`. The ``signature`` drives the key and
is deliberately case-independent so N cases sharing a criterion hit one
cached entry.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

from ng_core.cache import CACHE_KEY_VERSION

GEVAL_STEPS_PROMPT_VERSION = "v2"
GEVAL_STEPS_PARSER_VERSION = "v2"
GEVAL_SCORE_PROMPT_VERSION = "v2"
GEVAL_SCORE_PARSER_VERSION = "v2"


def compute_geval_signature(
    *,
    criteria: str,
    item_fields: Sequence[str],
    model: str,
    prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
    parser_version: str = GEVAL_STEPS_PARSER_VERSION,
) -> str:
    """Build a stable signature used as the GEval artifact cache key.

    ``item_fields`` are sorted so the signature is order-independent — two
    metrics that expose the same fields in different orders share cached steps.
    """

    fields_key = ",".join(sorted(item_fields))
    payload = (
        f"{model}\x00{prompt_version}\x00{parser_version}\x00{fields_key}\x00{criteria.strip()}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def build_geval_artifact_cache_key(signature: str) -> str:
    """Opaque key for GEval step artifacts in the universal cache store.

    The ``v2:`` prefix matches ``CACHE_KEY_VERSION``; the ``geval_artifact``
    namespace keeps these keys disjoint from per-node output keys produced by
    ``build_node_cache_key`` (which embed ``case_fingerprint``).
    """

    return f"{CACHE_KEY_VERSION}:geval_artifact:{signature}"


def collect_geval_signatures(
    *,
    cases: Iterable[Any],
    model: str,
    prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
    parser_version: str = GEVAL_STEPS_PARSER_VERSION,
) -> set[str]:
    """Return unique GEval step signatures across an iterable of cases."""

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _criteria_text(metric: Any) -> str:
        criteria = _get(metric, "criteria")
        if criteria is None:
            return ""
        if isinstance(criteria, dict):
            return str(criteria.get("text", "")).strip()
        if hasattr(criteria, "text"):
            return str(getattr(criteria, "text", "")).strip()
        return str(criteria).strip()

    signatures: set[str] = set()
    for case in cases:
        inputs = _get(case, "inputs") or _get(case, "input_payload")
        geval = _get(inputs, "geval") if inputs is not None else _get(case, "geval")
        if geval is None:
            continue
        metrics = _get(geval, "metrics") or []
        for metric in metrics:
            metric_steps = _get(metric, "evaluation_steps") or []
            if metric_steps:
                continue
            criteria_text = _criteria_text(metric)
            if not criteria_text:
                continue
            item_fields = list(_get(metric, "item_fields") or [])
            signatures.add(
                compute_geval_signature(
                    criteria=criteria_text,
                    item_fields=item_fields,
                    model=model,
                    prompt_version=prompt_version,
                    parser_version=parser_version,
                )
            )
    return signatures
