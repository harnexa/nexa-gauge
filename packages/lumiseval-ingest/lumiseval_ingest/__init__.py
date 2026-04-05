__version__ = "0.1.0"

from .adapters import create_dataset_adapter
from .canonical import (
    canonical_case_from_raw,
    normalize_context,
    normalize_geval,
    normalize_reference_files,
    pick_first,
)

__all__ = [
    "create_dataset_adapter",
    "pick_first",
    "normalize_context",
    "normalize_reference_files",
    "normalize_geval",
    "canonical_case_from_raw",
]
