"""GEval signature/artifact cache utilities shared by core tests and runners."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lumoseval_core.constants import CACHE_DIR

GEVAL_STEPS_PROMPT_VERSION = "v1"
GEVAL_STEPS_PARSER_VERSION = "v1"
_GEVAL_ARTIFACT_DIR = "geval_artifacts"


def compute_geval_signature(
    *,
    criteria: str,
    model: str,
    prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
    parser_version: str = GEVAL_STEPS_PARSER_VERSION,
) -> str:
    """Build a stable signature for GEval-step artifacts."""

    payload = f"{model}\x00{prompt_version}\x00{parser_version}\x00{criteria.strip()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _criteria_text(metric: Any) -> str:
    criteria = _get_value(metric, "criteria")
    if criteria is None:
        return ""
    if isinstance(criteria, dict):
        return str(criteria.get("text", "")).strip()
    if hasattr(criteria, "text"):
        return str(getattr(criteria, "text", "")).strip()
    return str(criteria).strip()


def collect_geval_signatures(
    *,
    cases: Iterable[Any],
    model: str,
    prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
    parser_version: str = GEVAL_STEPS_PARSER_VERSION,
) -> set[str]:
    """Collect deduplicated GEval signatures for metrics that still need steps."""

    signatures: set[str] = set()
    for case in cases:
        geval = _get_value(case, "geval")
        if geval is None:
            inputs = _get_value(case, "inputs")
            geval = _get_value(inputs, "geval") if inputs is not None else None
        if geval is None:
            continue

        metrics = _get_value(geval, "metrics") or []
        for metric in metrics:
            steps = _get_value(metric, "evaluation_steps") or []
            if steps:
                continue
            criteria = _criteria_text(metric)
            if not criteria:
                continue
            signatures.add(
                compute_geval_signature(
                    criteria=criteria,
                    model=model,
                    prompt_version=prompt_version,
                    parser_version=parser_version,
                )
            )
    return signatures


class GevalArtifactCache:
    """Filesystem cache storing generated GEval evaluation steps by signature."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        raw = cache_dir or os.getenv("LUMISEVAL_CACHE_DIR") or CACHE_DIR
        self._root = Path(raw) / _GEVAL_ARTIFACT_DIR

    def _path(self, signature: str) -> Path:
        return self._root / f"{signature}.json"

    def get_steps(self, signature: str) -> list[str] | None:
        path = self._path(signature)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        raw_steps = payload.get("evaluation_steps")
        if not isinstance(raw_steps, list):
            return None
        steps = [str(step).strip() for step in raw_steps if str(step).strip()]
        return steps or None

    def has(self, signature: str) -> bool:
        return self.get_steps(signature) is not None

    def put_steps(
        self,
        *,
        signature: str,
        model: str,
        criteria: str,
        evaluation_steps: list[str],
        prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
        parser_version: str = GEVAL_STEPS_PARSER_VERSION,
    ) -> None:
        path = self._path(signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": signature,
            "model": model,
            "criteria": criteria,
            "prompt_version": prompt_version,
            "parser_version": parser_version,
            "evaluation_steps": [
                str(step).strip() for step in evaluation_steps if str(step).strip()
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def count_missing(self, signatures: set[str]) -> int:
        return sum(1 for signature in signatures if not self.has(signature))
