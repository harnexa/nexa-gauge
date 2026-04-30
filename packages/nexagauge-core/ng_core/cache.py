"""Key-value node cache for NexaGauge.

==============================================================================
What this module is
==============================================================================

A thin opaque-key key-value store used by two kinds of caches in the pipeline:

1. **Per-node output cache** — the runner consults this before executing every
   node in a plan. A hit short-circuits the node; a miss runs the node and
   writes the result. Keyed by "which case × which node × which path got us
   here × which model is wired up".

2. **GEval step artifact cache** — node-internal, keyed by criteria signature
   so N cases sharing a criterion trigger ONE LLM call. It deliberately does
   NOT include the case content in its key. Same backend, different namespace.

Both share this module's ``CacheStore`` / ``NoOpCacheStore`` (the latter is
installed by ``--no-cache``).

==============================================================================
Storage layout
==============================================================================

    {cache_dir}/kv/{sha256(cache_key)[:2]}/{sha256(cache_key)}.json

Each file is a JSON envelope (:class:`KVCacheEntry`) carrying:
    cache_key    — the original opaque string (used to validate reads)
    node_name    — human-readable node tag for audits
    created_at   — ISO-8601 UTC timestamp
    node_output  — the JSON-serialized patch the node emitted
    metadata     — free-form; runner stamps execution_mode + case_fingerprint

Writes go via ``tmp + os.replace`` (atomic) so concurrent workers never see a
torn file.

==============================================================================
Cache key anatomy  (this is the part to read before adding keys)
==============================================================================

All keys start with ``CACHE_KEY_VERSION`` (currently ``"v2"``). Bump that
constant to invalidate every entry across the whole store at once.

──────────────────────────────────────────────────────────────────────────────
 (A) Per-node output key  — built by :func:`build_node_cache_key`
──────────────────────────────────────────────────────────────────────────────

    v2:{execution_mode}:{node_name}:{case_fingerprint}:{node_route_fingerprint}

Five axes, each orthogonal:

  1. ``execution_mode``  — ``"run"`` vs ``"estimate"``. Different modes get
     different entries because estimate mode produces cost-only stubs that
     ``run`` must not see.

  2. ``node_name``       — e.g. ``"grounding"``, ``"claims"``. Node-level
     isolation: changing ``claims`` must not evict ``redteam``.

  3. ``case_fingerprint`` — SHA-256 of case content, computed by
     :func:`compute_case_hash`. Controls "is this the same input?".
     **Inputs currently hashed:**
         generation (str)
         question (str | None)
         reference (str | None)
         context (list[str] | None)           ← joined with "|"
         geval metric spec, per metric:       ← name, item_fields,
             name                               criteria text, steps text
             item_fields                        (joined, sorted)
             criteria.text
             evaluation_steps[*].text
         redteam metric spec, per metric:     ← name, item_fields,
             name                               rubric goal,
             item_fields                        violations,
             rubric.goal                        non_violations
             rubric.violations
             rubric.non_violations
         reference_files (list[str])          ← sorted, joined

     **Want a new case-content dimension to affect cache validity?**
     Edit :func:`compute_case_hash` here. Anything NOT included is ignored
     by the cache — two cases differing only in that field share an entry.
     (If the schema change is also a breaking data shape change, also bump
     ``CACHE_KEY_VERSION`` to force-miss historical entries.)

  4. ``node_route_fingerprint`` — NOT computed in this file. The runner
     (``ng_graph/runner/fingerprints.py``) builds it by chaining, for this node
     AND every prerequisite node in its plan path:
         model             (resolved via llm_overrides → node config → cfg)
         fallback_model
         temperature
         execution_mode
         node_name
     So a cache hit requires: every upstream node on the path used the same
     model routing AND every upstream node itself was cacheable. This is the
     "did anything in my dependency chain change?" check.

     **Want a new routing dimension to affect cache validity?**
     Edit ``_node_route_fingerprint`` in ``runner/fingerprints.py`` — this file doesn't
     see routing; it just accepts whatever opaque string the runner hands in.

  5. ``CACHE_KEY_VERSION`` — global prefix. Bump to mass-invalidate.

──────────────────────────────────────────────────────────────────────────────
 (B) GEval artifact key  — built by ``build_geval_artifact_cache_key``
     (lives in ``ng_graph/nodes/metrics/geval/cache.py``)
──────────────────────────────────────────────────────────────────────────────

    v2:geval_artifact:{signature}

where ``signature = sha256(model | prompt_version | parser_version |
                           sorted(item_fields) | criteria.strip())[:24]``.

Intentionally case-independent: two different cases that share a criterion
produce the same signature → one cached LLM response, reused N times. This
is the core cost optimisation from the G-Eval paper applied to our runner.

**Want to add a dimension that invalidates step generation?**
Edit ``compute_geval_signature`` in the graph package. Bumping
``GEVAL_STEPS_PROMPT_VERSION`` or ``GEVAL_STEPS_PARSER_VERSION`` there is the
surgical way to bust only GEval-step artifacts without touching per-node
output keys.

==============================================================================
Policy gates  (who may read / write the cache and when)
==============================================================================

- ``NON_CACHEABLE_NODES``  — ``{"eval", "report"}``. These are pure
  aggregation / presentation; their outputs are fast to recompute and depend
  on upstream artifacts that ARE cached. Reads and writes are always denied.
  Add here to exclude a new node.

- ``ESTIMATE_CACHE_WRITE_NODES`` — allowlist for ``execution_mode == "estimate"``
  writes. Empty by default: estimate mode READS from run-mode entries (via
  fallback in the runner) but does not write, so speculative cost runs can't
  poison the real cache. If a new node has a stable estimate with value
  worth persisting, add its name here.

- ``cache_read_allowed`` / ``cache_write_allowed`` — the policy functions
  the runner consults. Change these (not individual call sites) to adjust
  semantics globally.

==============================================================================
Deserialisation  —  ``_FIELD_TYPE_MAP``
==============================================================================

Envelopes on disk are plain JSON. On read, :func:`_deserialize` walks the
``node_output`` dict and reconstructs typed Pydantic objects by key name.

**Want to cache a new Pydantic-typed field in a node output?**
Add an entry to ``_FIELD_TYPE_MAP``:
    "my_new_field": MyNewModel              # single model
    "my_list_field": (list, MyNewModel)     # list of models
Primitives (str/int/float/bool/list-of-primitives) don't need an entry; they
pass through unchanged.

Two special-case branches live inside ``_deserialize``:
- ``_METRIC_GROUP_FIELDS`` — any of these as a list → ``list[MetricResult]``.
  Add the field name here when registering a new metric-group node.
- ``estimated_costs`` dict → ``dict[str, CostEstimate]``.
- ``report`` dict/list → passed through raw (aggregation is structural).

==============================================================================
Backends
==============================================================================

:class:`CacheStore` — disk-backed; default path resolves to
``$NEXAGAUGE_CACHE_DIR`` when set, otherwise a per-user XDG-style directory
from :func:`ng_core.constants.default_cache_dir` (``~/.cache/nexagauge`` on
macOS/Linux, ``%LOCALAPPDATA%\\nexagauge`` on Windows).

:class:`NoOpCacheStore` — inherits from CacheStore; every method is a no-op.
Installed by the runner when ``--no-cache`` is passed. Because the node code
only talks to the :class:`NodeCacheBackend` Protocol (``has_key`` /
``get_entry_by_key`` / ``put_by_key``), nothing downstream needs to care
which backend is active.

==============================================================================
TL;DR  —  "I want to add a new cache dimension"
==============================================================================

- Input content of the case (e.g. a new ``metadata`` block in records)
      → add it to :func:`compute_case_hash`.

- Model routing (e.g. a new per-node ``top_p`` setting)
      → add it to ``_node_route_fingerprint`` in runner.py.

- A new node with typed output that should survive cache round-trips
      → add an entry to ``_FIELD_TYPE_MAP`` (and ``_METRIC_GROUP_FIELDS``
        if it's a metric node).

- A new node that should never be cached (pure aggregation / side-effect)
      → add its name to ``NON_CACHEABLE_NODES``.

- Mass-invalidate everything after a breaking change
      → bump ``CACHE_KEY_VERSION``.

- Invalidate only GEval step artifacts after a prompt change
      → bump ``GEVAL_STEPS_PROMPT_VERSION`` / ``GEVAL_STEPS_PARSER_VERSION``
        in ``ng_graph/nodes/metrics/geval/cache.py``.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel

from ng_core.constants import default_cache_dir
from ng_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    EvalReport,
    Geval,
    GevalCacheArtifact,
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
    "generation_refined_chunks": ChunkArtifacts,
    "generation_claims": ClaimArtifacts,
    "chunks": (list, Chunk),
    "raw_claims": (list, Claim),
    "unique_claims": (list, Claim),
    "geval_steps": GevalStepsArtifacts,
    "geval_artifact": GevalCacheArtifact,
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
        raw = cache_dir or os.getenv("NEXAGAUGE_CACHE_DIR") or default_cache_dir()
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
            # Corrupt or partially-written cache file — treat as miss rather than crash.
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
