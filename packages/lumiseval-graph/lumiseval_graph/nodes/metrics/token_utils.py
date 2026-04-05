"""
Token counting helpers for per-node cost estimation.

Provides:
  count_tokens(text)           — tiktoken token count for a plain string
  template_static_tokens(tmpl) — token count of a str.format() template with
                                  placeholders stripped, representing the fixed
                                  overhead that every call pays regardless of
                                  the dynamic content injected at runtime.
"""

import re

import tiktoken
from lumiseval_core.utils import _ENCODER


# Matches str.format() style placeholders: {context}, {claims}, {question}, etc.
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


def count_tokens(text: str) -> int:
    """Return the number of tiktoken tokens in *text*."""
    return len(_ENCODER.encode(text))


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
    return count_tokens(stripped)
