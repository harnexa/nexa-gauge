"""Dataset adapter protocol for loading canonical EvalCase rows."""

from typing import Any
from abc import ABC, abstractmethod

class DatasetAdapter(ABC):
    """Base adapter contract for all dataset sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable adapter identifier used in logs/telemetry."""

    @abstractmethod
    def iter_cases(
        self,
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Yield canonical evaluation cases."""
