"""Custom exceptions for lumis-eval."""


class LumisEvalError(Exception):
    """Base exception for all lumis-eval errors."""


class InputParseError(LumisEvalError):
    """Raised when an input file cannot be parsed.

    Attributes:
        record_index: Index of the offending record (for dataset inputs).
    """

    def __init__(self, message: str, record_index: int | None = None) -> None:
        super().__init__(message)
        self.record_index = record_index


class BudgetExceededError(LumisEvalError):
    """Raised when the cost estimate exceeds the configured budget cap."""


class EvidenceRoutingError(LumisEvalError):
    """Raised when evidence routing fails for a claim."""


class ClaimExtractionError(LumisEvalError):
    """Raised when claim extraction fails after retries."""
