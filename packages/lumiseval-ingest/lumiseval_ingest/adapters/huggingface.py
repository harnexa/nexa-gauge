"""Hugging Face dataset adapter (optional, lazy-imported)."""

from lumiseval_core.errors import InputParseError

from ..canonical import canonical_case_from_raw
from .base import DatasetAdapter


class HuggingFaceDatasetAdapter(DatasetAdapter):
    """Adapter for datasets loaded via `datasets.load_dataset`."""

    def __init__(
        self,
        dataset_id: str,
        config_name: str | None = None,
        revision: str | None = None,
        field_map: dict[str, str | list[str]] | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.revision = revision
        self.field_map = field_map or {}

    @property
    def name(self) -> str:
        return "huggingface"

    def _field_candidates(self, canonical_name: str, defaults: list[str]) -> list[str]:
        mapped = self.field_map.get(canonical_name)
        if mapped is None:
            return defaults
        if isinstance(mapped, str):
            return [mapped]
        if isinstance(mapped, list):
            return [str(item) for item in mapped]
        raise InputParseError(f"Invalid field_map entry for '{canonical_name}'.")

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

        generation_keys = self._field_candidates(
            "generation",
            ["generation", "response", "answer", "output", "completion"],
        )
        question_keys = self._field_candidates("question", ["question", "query", "prompt"])
        reference_keys = self._field_candidates("reference", ["ground_truth", "reference", "gold_answer", "label", "answer"])
        case_id_keys = self._field_candidates("case_id", ["case_id", "id", "prompt_id"])
        context_keys = self._field_candidates("context", ["context", "contexts", "documents"])
        ref_keys = self._field_candidates("reference_files", ["reference_files", "reference_paths"])
        geval_keys = self._field_candidates("geval", ["geval"])

        for idx, record in enumerate(dataset):
            if limit is not None and idx >= limit:
                break
            row = dict(record)
            try:
                yield canonical_case_from_raw(
                    row,
                    idx=idx,
                    dataset=self.dataset_id,
                    split=split,
                    metadata_mode="full",
                    field_candidates={
                        "generation": generation_keys,
                        "question": question_keys,
                        "reference": reference_keys,
                        "case_id": case_id_keys,
                        "context": context_keys,
                        "reference_files": ref_keys,
                        "geval": geval_keys,
                    },
                )
            except InputParseError as exc:
                if "missing required generation/response/answer/output/completion field." in str(exc):
                    raise InputParseError(
                        (
                            f"Dataset '{self.dataset_id}' split '{split}' row {idx} has no generation-like field. "
                            "Provide field_map={'generation': '<field>'} or precompute model outputs."
                        ),
                        record_index=idx,
                    ) from exc
                raise
