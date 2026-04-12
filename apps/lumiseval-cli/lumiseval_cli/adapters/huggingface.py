"""Hugging Face dataset adapter (optional, lazy-imported)."""

from itertools import islice

from lumiseval_core.errors import InputParseError

from .base import DatasetAdapter


class HuggingFaceDatasetAdapter(DatasetAdapter):
    """Adapter for datasets loaded via `datasets.load_dataset`."""

    def __init__(
        self,
        dataset_id: str,
        config_name: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.revision = revision

    @property
    def name(self) -> str:
        return "huggingface"

    def iter_cases(
        self,
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
    ):
        del seed  # stable source order
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise InputParseError(
                "datasets package is required for hf:// adapters. Install with `uv add datasets`."
            ) from exc

        dataset = load_dataset(
            path=self.dataset_id,
            name=self.config_name,
            split=split,
            revision=self.revision,
        )

        rows = dataset
        if limit is not None:
            rows = islice(dataset, limit)

        for idx, record in enumerate(rows):
            row = dict(record)
            try:
                yield row
            except InputParseError as exc:
                if "missing required generation/response/answer/output/completion field." in str(exc):
                    raise InputParseError(
                        (
                            f"Dataset '{self.dataset_id}' split '{split}' row {idx} has no generation-like field. "
                            "Precompute model outputs or align dataset fields to expected generation keys."
                        ),
                        record_index=idx,
                    ) from exc
                raise
