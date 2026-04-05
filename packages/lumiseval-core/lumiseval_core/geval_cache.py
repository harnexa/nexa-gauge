"""Signature-keyed artifact cache for GEval step generation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel

from lumiseval_core.constants import CACHE_DIR
from lumiseval_core.types import EvalCase

GEVAL_STEPS_PROMPT_VERSION = "v1"
GEVAL_STEPS_PARSER_VERSION = "v1"
_GEVAL_ARTIFACT_DIR = "geval_artifacts"


class GevalStepsArtifact(BaseModel):
    """Persisted artifact for one GEval-step signature."""

    signature: str
    model: str
    prompt_version: str
    parser_version: str
    criteria: str
    evaluation_steps: list[str]
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
    cases: Iterable[EvalCase],
    model: str,
    prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
    parser_version: str = GEVAL_STEPS_PARSER_VERSION,
) -> set[str]:
    """Return unique GEval step signatures across an iterable of cases."""

    signatures: set[str] = set()
    for case in cases:
        if case.geval is None:
            continue
        for metric in case.geval.metrics:
            if metric.evaluation_steps:
                continue
            if not metric.criteria:
                continue
            signatures.add(
                compute_geval_signature(
                    criteria=metric.criteria,
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
    def _coerce_artifact(raw: dict, signature: str) -> GevalStepsArtifact:
        """Parse a persisted GEval artifact payload into GevalStepsArtifact."""

        return GevalStepsArtifact(
            signature=str(raw.get("signature", signature)),
            model=str(raw.get("model", "")),
            prompt_version=str(raw.get("prompt_version", GEVAL_STEPS_PROMPT_VERSION)),
            parser_version=str(raw.get("parser_version", GEVAL_STEPS_PARSER_VERSION)),
            criteria=str(raw.get("criteria", "")).strip(),
            evaluation_steps=list(raw.get("evaluation_steps", [])),
            created_at=str(raw.get("created_at", "")),
        )

    def _read(self, path: Path, signature: str) -> Optional[GevalStepsArtifact]:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            artifact = self._coerce_artifact(raw, signature)
            if not artifact.evaluation_steps:
                return None
            return artifact
        except Exception:
            return None

    def get(self, signature: str) -> Optional[GevalStepsArtifact]:
        """Read and validate one artifact by signature."""

        return self._read(self._path(signature), signature)

    def get_steps(self, signature: str) -> Optional[list[str]]:
        """Return cached evaluation steps for a signature, when present."""

        artifact = self.get(signature)
        if artifact is None:
            return None
        return list(artifact.evaluation_steps)

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
        criteria: str,
        evaluation_steps: list[str],
        prompt_version: str = GEVAL_STEPS_PROMPT_VERSION,
        parser_version: str = GEVAL_STEPS_PARSER_VERSION,
    ) -> GevalStepsArtifact:
        """Create and persist a GEval artifact from generated evaluation steps."""

        artifact = GevalStepsArtifact(
            signature=signature,
            model=model,
            prompt_version=prompt_version,
            parser_version=parser_version,
            criteria=criteria.strip(),
            evaluation_steps=list(evaluation_steps),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.put(artifact)
        return artifact

    def count_missing(self, signatures: set[str]) -> int:
        """Count how many signatures are absent from the cache."""

        return sum(1 for signature in signatures if self.get(signature) is None)
