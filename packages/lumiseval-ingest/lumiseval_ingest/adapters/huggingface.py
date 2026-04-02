"""Hugging Face dataset adapter (optional, lazy-imported)."""

from typing import Any

from lumiseval_core.errors import InputParseError
from lumiseval_core.types import EvalCase

from .base import DatasetAdapter
from .local_file import _normalize_context, _normalize_reference_files, _normalize_rubric


def _first_present(record: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return None


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
        rubric_keys = self._field_candidates("rubric", ["rubric", "rubric"])

        for idx, record in enumerate(dataset):
            if limit is not None and idx >= limit:
                break
            row = dict(record)

            generation = _first_present(row, generation_keys)
            if generation is None or str(generation).strip() == "":
                raise InputParseError(
                    (
                        f"Dataset '{self.dataset_id}' split '{split}' row {idx} has no generation-like field. "
                        "Provide field_map={'generation': '<field>'} or precompute model outputs."
                    ),
                    record_index=idx,
                )

            case_id = _first_present(row, case_id_keys)
            question = _first_present(row, question_keys)
            reference = _first_present(row, reference_keys)
            context = _normalize_context(_first_present(row, context_keys))
            reference_files = _normalize_reference_files(_first_present(row, ref_keys))
            rubric = _normalize_rubric(_first_present(row, rubric_keys))

            yield EvalCase(
                case_id=str(case_id if case_id is not None else idx),
                generation=str(generation),
                dataset=self.dataset_id,
                split=split,
                question=str(question) if question is not None else None,
                reference=str(reference) if reference is not None else None,
                context=context,
                reference_files=reference_files,
                rubric=rubric,
                metadata=row,
            )
