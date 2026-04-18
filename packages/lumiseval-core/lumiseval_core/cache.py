"""Key-value node cache for LumisEval.

The cache stores per-node execution patches under an opaque `cache_key`.
Key construction is owned by the runner so it can encode case fingerprint,
execution mode, node route/model routing, and dependency chain versioning.

Storage layout:
    {cache_dir}/kv/{prefix}/{sha256(cache_key)}.json

Each entry stores:
  - `cache_key` and `node_name` for validation/debugging
  - serialized `node_output` patch
  - optional free-form `metadata`
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel

from lumiseval_core.constants import CACHE_DIR
from lumiseval_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    EvalReport,
    Geval,
    GevalConfig,
    GevalMetrics,
    GevalStepsArtifacts,
    GroundingMetrics,
    Inputs,
    MetricResult,
    Redteam,
    RedteamMetrics,
    ReferenceMetrics,
    RelevanceMetrics,
)

# ── Field → Pydantic type map ────────────────────────────────────────────────
# Used by _deserialize() to reconstruct typed objects from plain dicts.
# Only fields that contain Pydantic models need entries here;
# primitives (str, bool, float, list[str]) pass through unchanged.

_FIELD_TYPE_MAP: dict[str, Any] = {
    "inputs": Inputs,
    "generation_chunk": ChunkArtifacts,
    "generation_claims": ClaimArtifacts,
    "generation_dedup_claims": ClaimArtifacts,
    "chunks": (list, Chunk),
    "raw_claims": (list, Claim),
    "unique_claims": (list, Claim),
    "geval_steps": GevalStepsArtifacts,
    "grounding_metrics": GroundingMetrics,
    "relevance_metrics": RelevanceMetrics,
    "redteam_metrics": RedteamMetrics,
    "geval_metrics": GevalMetrics,
    "reference_metrics": ReferenceMetrics,
    "report": EvalReport,
    "geval": GevalConfig,
}

_METRIC_GROUP_FIELDS = {
    "grounding_metrics",
    "relevance_metrics",
    "redteam_metrics",
    "geval_metrics",
    "reference_metrics",
}


class KVCacheEntry(TypedDict):
    """Typed key-value cache envelope for opaque-key backends."""

    cache_key: str
    node_name: str
    created_at: str | None
    node_output: dict[str, Any]
    metadata: dict[str, Any]


@runtime_checkable
class NodeCacheBackend(Protocol):
    """Minimal cache backend contract used by CachedNodeRunner."""

    def has_key(self, cache_key: str) -> bool: ...

    def get_entry_by_key(self, cache_key: str) -> Optional[KVCacheEntry]: ...

    def put_by_key(
        self,
        cache_key: str,
        node_name: str,
        node_output: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None: ...


# ── Serialisation helpers ────────────────────────────────────────────────────


def _serialize(node_output: dict[str, Any]) -> dict[str, Any]:
    """Convert a node output dict to a JSON-serialisable form."""

    def _to_jsonable(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        return value

    result: dict[str, Any] = {}
    for key, value in node_output.items():
        result[key] = _to_jsonable(value)
    return result


def _deserialize(node_output_raw: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct typed objects from a deserialised JSON node output dict."""
    result: dict[str, Any] = {}
    for key, value in node_output_raw.items():
        if value is None:
            result[key] = None
            continue
        if key in _METRIC_GROUP_FIELDS and isinstance(value, list):
            result[key] = [MetricResult.model_validate(v) for v in value]
            continue
        if key == "estimated_costs" and isinstance(value, dict):
            result[key] = {k: CostEstimate.model_validate(v) for k, v in value.items()}
            continue
        if key == "report" and isinstance(value, (list, dict)):
            result[key] = value
            continue
        type_info = _FIELD_TYPE_MAP.get(key)
        if type_info is None:
            result[key] = value  # primitive — pass through
        elif isinstance(type_info, tuple):
            _, item_type = type_info
            result[key] = [item_type.model_validate(v) for v in value]
        else:
            result[key] = type_info.model_validate(value)
    return result


# ── Hash helpers ─────────────────────────────────────────────────────────────

CACHE_KEY_VERSION = "v2"

# Estimate mode cache policy:
# - Reads are always allowed so estimate can reuse run-mode cache.
# - Writes are restricted to this hardcoded allowlist (empty by default).
ESTIMATE_CACHE_WRITE_NODES: frozenset[str] = frozenset()
NON_CACHEABLE_NODES: frozenset[str] = frozenset({"eval", "report"})


def cache_read_allowed(*, execution_mode: str, node_name: str) -> bool:
    if node_name in NON_CACHEABLE_NODES:
        return False
    if execution_mode == "run":
        return True
    if execution_mode == "estimate":
        return True
    return True


def cache_write_allowed(*, execution_mode: str, node_name: str) -> bool:
    if node_name in NON_CACHEABLE_NODES:
        return False
    if execution_mode == "run":
        return True
    if execution_mode == "estimate":
        return node_name in ESTIMATE_CACHE_WRITE_NODES
    return True


def build_node_cache_key(
    *,
    case_fingerprint: str,
    node_name: str,
    execution_mode: str,
    node_route_fingerprint: str,
) -> str:
    """Build opaque cache key for a single node execution output."""
    return (
        f"{CACHE_KEY_VERSION}:"
        f"{execution_mode}:"
        f"{node_name}:"
        f"{case_fingerprint}:"
        f"{node_route_fingerprint}"
    )


def compute_case_hash(
    generation: str,
    question: Optional[str],
    reference: Optional[str],
    geval: Optional[GevalConfig | Geval] = None,
    redteam: Optional[Redteam] = None,
    context: Optional[list[str]] = None,
    reference_files: Optional[list[str]] = None,
) -> str:
    """Stable SHA-256 hash of the case's input content.

    Changing generation / question / reference / context / GEval metrics /
    reference_files produces a different hash, which causes
    a cache miss for the affected case.
    """

    def _value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return str(value.get("text", "")).strip()
        if hasattr(value, "text"):
            return str(getattr(value, "text", "")).strip()
        return str(value).strip()

    def _metric_steps_text(raw_steps: Any) -> str:
        if not isinstance(raw_steps, list):
            return ""
        parts = [_text(step) for step in raw_steps]
        return "|".join([p for p in parts if p])

    def _metric_fields_text(metric: Any) -> str:
        fields = _value(metric, "item_fields") or _value(metric, "item_fields") or []
        if not isinstance(fields, list):
            return ""
        return "|".join([_text(field) for field in fields if _text(field)])

    geval_text = ""
    if geval is not None:
        parts: list[str] = []
        metrics = _value(geval, "metrics", [])
        for metric in metrics if isinstance(metrics, list) else []:
            name = _text(_value(metric, "name"))
            criteria_text = _text(_value(metric, "criteria"))
            steps_text = _metric_steps_text(_value(metric, "evaluation_steps"))
            fields_text = _metric_fields_text(metric)
            parts.append(f"{name}\x1f{fields_text}\x1f{criteria_text}\x1f{steps_text}")
        geval_text = "|".join(sorted(parts))

    redteam_text = ""
    if redteam is not None:
        parts = []
        metrics = _value(redteam, "metrics", [])
        for metric in metrics if isinstance(metrics, list) else []:
            rubric = _value(metric, "rubric", {})
            rubric_goal = _text(_value(rubric, "goal"))
            rubric_violations = _metric_steps_text(_value(rubric, "violations"))
            rubric_non_violations = _metric_steps_text(_value(rubric, "non_violations"))
            fields_text = _metric_fields_text(metric)
            name = _text(_value(metric, "name"))
            parts.append(
                f"{name}\x1f{fields_text}\x1f{rubric_goal}\x1f"
                f"{rubric_violations}\x1f{rubric_non_violations}"
            )
        redteam_text = "|".join(sorted(parts))

    context_text = "|".join([str(c) for c in (context or []) if c is not None])
    reference_text = "|".join(sorted(reference_files or []))
    raw = (
        f"{generation}\x00{question or ''}\x00{reference or ''}\x00"
        f"{context_text}\x00{geval_text}\x00{redteam_text}\x00{reference_text}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── CacheStore ───────────────────────────────────────────────────────────────


class CacheStore:
    """Filesystem-backed key-value store for node execution patches."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        raw = cache_dir or os.getenv("LUMISEVAL_CACHE_DIR") or CACHE_DIR
        self._root = Path(raw)

    def _kv_path(self, cache_key: str) -> Path:
        digest = hashlib.sha256(cache_key.encode()).hexdigest()
        return self._root / "kv" / digest[:2] / f"{digest}.json"

    def has_key(self, cache_key: str) -> bool:
        """Return True if an opaque-key cache entry exists."""
        return self._kv_path(cache_key).exists()

    def get_entry_by_key(self, cache_key: str) -> Optional[KVCacheEntry]:
        """Load and deserialize a key-value cache envelope using an opaque key."""
        p = self._kv_path(cache_key)
        if not p.exists():
            return None
        try:
            envelope = json.loads(p.read_text())
            if envelope.get("cache_key") != cache_key:
                return None
            node_output_raw = envelope.get("node_output")
            if not isinstance(node_output_raw, dict):
                return None
            metadata_raw = envelope.get("metadata")
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            return {
                "cache_key": cache_key,
                "node_name": str(envelope.get("node_name", "")),
                "created_at": envelope.get("created_at"),
                "node_output": _deserialize(node_output_raw),
                "metadata": metadata,
            }
        except Exception:
            return None

    def get_by_key(self, cache_key: str) -> Optional[dict[str, Any]]:
        """Load and deserialize node output using an opaque key."""
        entry = self.get_entry_by_key(cache_key)
        if entry is None:
            return None
        return entry["node_output"]

    def put_by_key(
        self,
        cache_key: str,
        node_name: str,
        node_output: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Serialize and persist one opaque-key cache envelope.

        Uses atomic rename to avoid torn writes under concurrent execution.
        """
        p = self._kv_path(cache_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "cache_key": cache_key,
            "node_name": node_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_output": _serialize(node_output),
            "metadata": metadata or {},
        }
        tmp_path = p.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(envelope, indent=2))
        os.replace(tmp_path, p)


class NoOpCacheStore(CacheStore):
    """Drop-in replacement that never reads or writes — used with --no-cache."""

    def has_key(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def get_entry_by_key(self, *args: Any, **kwargs: Any) -> Optional[KVCacheEntry]:
        return None

    def get_by_key(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return None

    def put_by_key(self, *args: Any, **kwargs: Any) -> None:
        return
