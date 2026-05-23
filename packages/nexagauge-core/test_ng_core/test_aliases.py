"""Backward-compat regression tests for the canonical alias table.

After the `generation` -> `output` and `question` -> `input` rename, records
written against the old vocabulary must still resolve correctly through
:func:`resolve_alias`.
"""

from __future__ import annotations

from ng_core.aliases import INPUT_FIELD_ALIASES, resolve_alias


def test_legacy_generation_key_resolves_to_output() -> None:
    record = {"generation": "the model's answer"}
    assert resolve_alias(record, "output") == "the model's answer"


def test_legacy_question_key_resolves_to_input() -> None:
    record = {"question": "what is 2 + 2?"}
    assert resolve_alias(record, "input") == "what is 2 + 2?"


def test_new_canonical_keys_take_priority_over_legacy() -> None:
    record = {"output": "new", "generation": "old"}
    assert resolve_alias(record, "output") == "new"

    record = {"input": "new", "question": "old"}
    assert resolve_alias(record, "input") == "new"


def test_long_standing_aliases_still_work() -> None:
    assert resolve_alias({"response": "x"}, "output") == "x"
    assert resolve_alias({"answer": "y"}, "output") == "y"
    assert resolve_alias({"completion": "z"}, "output") == "z"
    assert resolve_alias({"query": "q"}, "input") == "q"
    assert resolve_alias({"prompt": "p"}, "input") == "p"


def test_canonical_keys_listed_first_in_alias_tuple() -> None:
    assert INPUT_FIELD_ALIASES["output"][0] == "output"
    assert INPUT_FIELD_ALIASES["input"][0] == "input"
    assert "generation" in INPUT_FIELD_ALIASES["output"]
    assert "question" in INPUT_FIELD_ALIASES["input"]
