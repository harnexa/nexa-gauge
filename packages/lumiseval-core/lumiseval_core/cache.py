"""
Node-level execution cache for LumisEval.

Cache keys are derived from the input content (case_hash) and the evaluation
configuration (config_hash) so that:
  - Changing the generation/question/rubric invalidates the cache for that case.
  - Changing the judge model or enable_* flags invalidates only config-sensitive nodes.
  - Adding new cases to a dataset runs only the new cases.

Storage layout:
    {cache_dir}/{case_hash}/{config_hash}/{node_name}.json

Each file stores the node's output dict (the dict returned by the node function)
together with metadata so entries can be inspected without running the pipeline.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from lumiseval_core.constants import CACHE_DIR
from lumiseval_core.types import (
    Chunk,
    Claim,
    CostEstimate,
    EvalJobConfig,
    EvalReport,
    InputMetadata,
    MetricResult,
    Rubric,
)

# ── Field → Pydantic type map ────────────────────────────────────────────────
# Used by _deserialize() to reconstruct typed objects from plain dicts.
# Only fields that contain Pydantic models need entries here;
# primitives (str, bool, float, list[str]) pass through unchanged.

_FIELD_TYPE_MAP: dict[str, Any] = {
    "metadata": InputMetadata,
    "cost_estimate": CostEstimate,
    "chunks": (list, Chunk),
    "raw_claims": (list, Claim),
    "unique_claims": (list, Claim),
    "grounding_metrics": (list, MetricResult),
    "relevance_metrics": (list, MetricResult),
    "redteam_metrics": (list, MetricResult),
    "rubric_metrics": (list, MetricResult),
    "report": EvalReport,
    "job_config": EvalJobConfig,
    "rubric": (list, Rubric),
}


# ── Serialisation helpers ────────────────────────────────────────────────────


def _serialize(node_output: dict[str, Any]) -> dict[str, Any]:
    """Convert a node output dict to a JSON-serialisable form."""
    result: dict[str, Any] = {}
    for key, value in node_output.items():
        if value is None:
            result[key] = None
        elif isinstance(value, BaseModel):
            result[key] = value.model_dump()
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            result[key] = [v.model_dump() for v in value]
        else:
            result[key] = value
    return result


def _deserialize(node_output_raw: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct typed objects from a deserialised JSON node output dict."""
    result: dict[str, Any] = {}
    for key, value in node_output_raw.items():
        if value is None:
            result[key] = None
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


def compute_case_hash(
    generation: str,
    question: Optional[str],
    ground_truth: Optional[str],
    rubric: list[Rubric],
    context: Optional[list[str]] = None,
    reference_files: Optional[list[str]] = None,
) -> str:
    """Stable SHA-256 hash of the case's input content.

    Changing generation / question / ground_truth / context / rubric rules / reference_files
    produces a different hash, which causes a cache miss for the affected case.
    """
    rubric_text = "|".join(sorted(f"{r.id}\x1f{r.statement}\x1f{r.pass_condition}" for r in rubric))
    context_text = "|".join(context or [])
    reference_text = "|".join(sorted(reference_files or []))
    raw = (
        f"{generation}\x00{question or ''}\x00{ground_truth or ''}\x00"
        f"{context_text}\x00{rubric_text}\x00{reference_text}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_config_hash(job_config: EvalJobConfig) -> str:
    """Stable SHA-256 hash of the evaluation configuration fields that affect node outputs.

    Excludes job_id and budget_cap_usd (they do not influence node computation).
    """
    raw = (
        f"{job_config.judge_model}"
        f"|{job_config.enable_grounding}"
        f"|{job_config.enable_relevance}"
        f"|{job_config.enable_redteam}"
        f"|{job_config.enable_rubric}"
        f"|{job_config.web_search}"
        f"|{job_config.evidence_threshold}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── CacheStore ───────────────────────────────────────────────────────────────


class CacheStore:
    """Filesystem-backed store for per-node execution outputs.

    Each entry is a small JSON file:
        {cache_dir}/{case_hash}/{config_hash}/{node_name}.json

    The store is safe to use from a single process (no file locking).
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        raw = cache_dir or os.getenv("LUMISEVAL_CACHE_DIR") or CACHE_DIR
        self._root = Path(raw)

    def _path(self, case_hash: str, config_hash: str, node_name: str) -> Path:
        return self._root / case_hash / config_hash / f"{node_name}.json"

    def has(self, case_hash: str, config_hash: str, node_name: str) -> bool:
        """Return True if a valid cache entry exists for this node."""
        return self._path(case_hash, config_hash, node_name).exists()

    def get(self, case_hash: str, config_hash: str, node_name: str) -> Optional[dict[str, Any]]:
        """Load and deserialise a cached node output.

        Returns None on a cache miss or if the file cannot be parsed.
        """
        p = self._path(case_hash, config_hash, node_name)
        if not p.exists():
            return None
        try:
            envelope = json.loads(p.read_text())
            return _deserialize(envelope["node_output"])
        except Exception:
            return None

    def put(
        self,
        case_hash: str,
        config_hash: str,
        node_name: str,
        node_output: dict[str, Any],
    ) -> None:
        """Serialise and persist a node output dict."""
        p = self._path(case_hash, config_hash, node_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "node_name": node_name,
            "case_hash": case_hash,
            "config_hash": config_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_output": _serialize(node_output),
        }
        p.write_text(json.dumps(envelope, indent=2))

    def cached_nodes(self, case_hash: str, config_hash: str) -> list[str]:
        """Return the names of all nodes cached for this (case, config) pair."""
        d = self._root / case_hash / config_hash
        if not d.exists():
            return []
        return [p.stem for p in d.glob("*.json")]


class NoOpCacheStore(CacheStore):
    """Drop-in replacement that never reads or writes — used with --no-cache."""

    def has(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def get(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return None

    def put(self, *args: Any, **kwargs: Any) -> None:
        return

    def cached_nodes(self, *args: Any, **kwargs: Any) -> list[str]:
        return []
