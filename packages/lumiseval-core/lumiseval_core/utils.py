"""General-purpose utilities for lumis-eval."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from pydantic import BaseModel

import tiktoken

from .constants import TIKTOKEN_ENCODING

_ENCODER = tiktoken.get_encoding(TIKTOKEN_ENCODING)

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _to_serializable(obj: Any) -> Any:
    """Recursively convert a value to a JSON-serializable structure."""
    if isinstance(obj, BaseModel):
        return {k: _to_serializable(v) for k, v in obj.model_dump().items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_serializable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def pprint_model(obj: Any, indent: int = 2) -> None:
    """Pretty-print a Pydantic BaseModel, dataclass, dict, or list.

    Uses JSON formatting for consistent indentation and alignment.
    Non-serializable leaf values are rendered via str().
    """
    print(json.dumps(_to_serializable(obj), indent=indent, default=str))
