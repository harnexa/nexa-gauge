from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeLLM:
    """Reusable fake LLM for tests that need to mock LLM responses."""

    def __init__(self, response: dict[str, Any] | None = None):
        self.response = response or {
            "parsed": SimpleNamespace(),
            "parsing_error": None,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "model": "gpt-4o-mini",
        }
        self.call_count = 0
        self.messages_log: list = []

    def invoke(self, messages: Any) -> dict:
        self.call_count += 1
        self.messages_log.append(messages)
        return self.response

    def invoke_with_logprobs(self, messages: Any, **_kwargs: Any) -> dict:
        return self.invoke(messages)
