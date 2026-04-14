"""General-purpose utilities for lumis-eval."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import tiktoken
from pydantic import BaseModel

from .constants import TIKTOKEN_ENCODING

_ENCODER = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# Matches str.format() style placeholders: {context}, {claims}, {question}, etc.
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


def template_static_tokens(template: str) -> int:
    """Return the token count of the static (non-placeholder) portions of *template*.

    Strips all ``{placeholder}`` patterns before counting, giving the fixed
    prompt overhead that every LLM call for this node pays independent of the
    dynamic content (context passages, claims list, question, etc.).

    Example::

        template_static_tokens("Context:\\n{context}\\n\\nClaims:\\n{claims}")
        # → tokens("Context:\\n\\n\\nClaims:\\n")
    """
    stripped = _PLACEHOLDER_RE.sub("", template)
    return _count_tokens(stripped)


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
