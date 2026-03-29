"""Dataset adapter protocol for loading canonical EvalCase rows."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from lumiseval_core.types import EvalCase


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
    ) -> Iterator[EvalCase]:
        """Yield canonical evaluation cases."""
