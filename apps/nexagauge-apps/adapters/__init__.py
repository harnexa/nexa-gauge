"""Dataset adapters for canonical EvalCase loading."""

from .base import DatasetAdapter
from .huggingface import HuggingFaceDatasetAdapter
from .local_file import LocalFileDatasetAdapter
from .registry import create_dataset_adapter

__all__ = [
    "DatasetAdapter",
    "LocalFileDatasetAdapter",
    "HuggingFaceDatasetAdapter",
    "create_dataset_adapter",
]
