"""Adapter selection utilities."""

from pathlib import Path

from lumiseval_core.errors import InputParseError

from .base import DatasetAdapter
from .huggingface import HuggingFaceDatasetAdapter
from .local_file import LocalFileDatasetAdapter


def create_dataset_adapter(
    source: str | Path,
    adapter: str = "auto",
    *,
    config_name: str | None = None,
    revision: str | None = None,
) -> DatasetAdapter:
    """Create a dataset adapter for a local file path or hf:// dataset id."""
    source_str = str(source)

    if adapter not in {"auto", "local", "huggingface"}:
        raise InputParseError(f"Unknown adapter '{adapter}'. Use one of: auto, local, huggingface.")

    if adapter == "local":
        return LocalFileDatasetAdapter(Path(source_str))

    if adapter == "huggingface":
        dataset_id = source_str.replace("hf://", "", 1)
        return HuggingFaceDatasetAdapter(
            dataset_id=dataset_id,
            config_name=config_name,
            revision=revision,
        )

    # auto mode
    if source_str.startswith("hf://"):
        dataset_id = source_str.replace("hf://", "", 1)
        return HuggingFaceDatasetAdapter(
            dataset_id=dataset_id,
            config_name=config_name,
            revision=revision,
        )

    path = Path(source_str)
    if path.exists():
        return LocalFileDatasetAdapter(path)

    raise InputParseError(
        f"Could not resolve dataset source '{source_str}'. Use a local path or hf://<dataset-id>."
    )
