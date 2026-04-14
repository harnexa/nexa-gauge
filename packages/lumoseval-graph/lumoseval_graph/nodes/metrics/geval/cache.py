"""Signature-keyed artifact cache for GEval step generation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from lumoseval_core.constants import CACHE_DIR
from lumoseval_core.types import Item
from pydantic import BaseModel

GEVAL_STEPS_PROMPT_VERSION = "v1"
GEVAL_STEPS_PARSER_VERSION = "v1"
_GEVAL_ARTIFACT_DIR = "geval_artifacts"


class GevalStepsArtifact(BaseModel):
    """Persisted artifact for one GEval-step signature."""

    signature: str
    model: str
    prompt_version: str
    parser_version: str
    criteria: Item
    evaluation_steps: list[Item]
    created_at: str


def compute_geval_signature(
    *,
    criteria: str,
    model: str,
    prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
    parser_version: str = GEVAL_STEPS_PARSER_VERSION,
) -> str:
    """Build a stable signature used as the GEval artifact cache key."""

    payload = f"{model}\x00{prompt_version}\x00{parser_version}\x00{criteria.strip()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


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
            signatures.add(
                compute_geval_signature(
                    criteria=criteria_text,
                    model=model,
                    prompt_version=prompt_version,
                    parser_version=parser_version,
                )
            )
    return signatures


class GevalArtifactCache:
    """Filesystem cache for GEval step-generation artifacts."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        raw = cache_dir or os.getenv("LUMISEVAL_CACHE_DIR") or CACHE_DIR
        root = Path(raw)
        self._root = root / _GEVAL_ARTIFACT_DIR

    def _path(self, signature: str) -> Path:
        return self._root / f"{signature}.json"

    @staticmethod
    def _coerce_item(raw: Any, *, cached_default: bool) -> Optional[Item]:
        if isinstance(raw, Item):
            return Item(**raw.model_dump())

        if isinstance(raw, dict):
            text = str(raw.get("text", "")).strip()
            if not text:
                return None
            return Item(
                id=str(raw.get("id", "")),
                text=text,
                tokens=float(raw.get("tokens", 0.0) or 0.0),
                confidence=float(raw.get("confidence", 1.0) or 1.0),
                cached=bool(raw.get("cached", cached_default)),
            )

        text = str(raw or "").strip()
        if not text:
            return None
        return Item(text=text, tokens=0.0, cached=cached_default)

    @classmethod
    def _coerce_artifact(cls, raw: dict, signature: str) -> GevalStepsArtifact:
        """Parse a persisted GEval artifact payload into GevalStepsArtifact."""
        criteria = cls._coerce_item(raw.get("criteria", {}), cached_default=True)
        if criteria is None:
            criteria = Item(text="", tokens=0.0, cached=True)

        raw_steps = raw.get("evaluation_steps", [])
        steps: list[Item] = []
        for item in raw_steps if isinstance(raw_steps, list) else []:
            parsed = cls._coerce_item(item, cached_default=True)
            if parsed is not None:
                steps.append(parsed)

        return GevalStepsArtifact(
            signature=str(raw.get("signature", signature)),
            model=str(raw.get("model", "")),
            prompt_version=str(raw.get("prompt_version", GEVAL_STEPS_PROMPT_VERSION)),
            parser_version=str(raw.get("parser_version", GEVAL_STEPS_PARSER_VERSION)),
            criteria=criteria,
            evaluation_steps=steps,
            created_at=str(raw.get("created_at", "")),
        )

    def _read(self, path: Path, signature: str) -> Optional[GevalStepsArtifact]:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            artifact = self._coerce_artifact(raw, signature)
            if not artifact.evaluation_steps:
                return None
            return artifact
        except Exception:
            return None

    def get(self, signature: str) -> Optional[GevalStepsArtifact]:
        """Read and validate one artifact by signature."""

        return self._read(self._path(signature), signature)

    def get_steps(self, signature: str) -> Optional[list[Item]]:
        """Return cached evaluation steps for a signature, when present."""

        artifact = self.get(signature)
        if artifact is None:
            return None
        return [Item(**step.model_dump()) for step in artifact.evaluation_steps]

    def has(self, signature: str) -> bool:
        """True when an artifact exists and can be parsed."""

        return self.get(signature) is not None

    def put(self, artifact: GevalStepsArtifact) -> None:
        """Persist one GEval artifact to canonical storage."""

        path = self._path(artifact.signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact.model_dump(), indent=2))

    def put_steps(
        self,
        *,
        signature: str,
        model: str,
        criteria: Item | str,
        evaluation_steps: list[Item] | list[str],
        prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
        parser_version: str = GEVAL_STEPS_PARSER_VERSION,
    ) -> GevalStepsArtifact:
        """Create and persist a GEval artifact from generated evaluation steps."""

        parsed_criteria = self._coerce_item(criteria, cached_default=True)
        if parsed_criteria is None:
            parsed_criteria = Item(text="", tokens=0.0, cached=True)

        parsed_steps: list[Item] = []
        for step in evaluation_steps:
            parsed = self._coerce_item(step, cached_default=True)
            if parsed is not None:
                parsed_steps.append(parsed)

        artifact = GevalStepsArtifact(
            signature=signature,
            model=model,
            prompt_version=prompt_version,
            parser_version=parser_version,
            criteria=parsed_criteria,
            evaluation_steps=parsed_steps,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.put(artifact)
        return artifact

    def count_missing(self, signatures: set[str]) -> int:
        """Count how many signatures are absent from the cache."""

        return sum(1 for signature in signatures if self.get(signature) is None)
