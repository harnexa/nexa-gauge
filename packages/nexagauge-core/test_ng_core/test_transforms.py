"""Unit tests for the dataset-record transform registry.

# uv run pytest packages/nexagauge-core/test_ng_core/test_transforms.py -q
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from ng_core.errors import InputParseError
from ng_core.extensions import (
    get_transform,
    list_transforms,
    load_extension_file,
    register_transform,
)
from ng_core.extensions.transforms import _clear_registry_for_tests


@pytest.fixture(autouse=True)
def _isolate_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def test_register_and_get_round_trips():
    @register_transform("identity")
    def _ident(record):
        return record

    assert get_transform("identity")({"a": 1}) == {"a": 1}
    assert list_transforms() == ["identity"]


def test_register_same_callable_under_same_name_is_idempotent():
    def fn(record):
        return record

    register_transform("same")(fn)
    register_transform("same")(fn)  # no-op
    assert list_transforms() == ["same"]


def test_register_overwrites_under_same_name():
    """Last register wins; re-loaded files don't crash on identity mismatch."""
    register_transform("dup")(lambda r: {**r, "v": "first"})
    register_transform("dup")(lambda r: {**r, "v": "second"})
    assert get_transform("dup")({})["v"] == "second"


def test_register_empty_name_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        register_transform("")


def test_get_transform_unknown_name_raises_with_listing():
    register_transform("a")(lambda r: r)
    register_transform("b")(lambda r: r)
    with pytest.raises(InputParseError) as exc:
        get_transform("missing")
    msg = str(exc.value)
    assert "missing" in msg
    assert "'a'" in msg and "'b'" in msg


def test_get_transform_when_registry_empty():
    with pytest.raises(InputParseError, match="No transforms registered"):
        get_transform("anything")


def test_load_extension_file_executes_decorators(tmp_path: Path):
    file = tmp_path / "user_transforms.py"
    file.write_text(
        textwrap.dedent(
            """
            from ng_core import register_transform

            @register_transform("loaded")
            def t(record):
                return {"output": record["resp"]}
            """
        ).strip()
    )
    assert list_transforms() == []
    load_extension_file(file)
    assert "loaded" in list_transforms()
    fn = get_transform("loaded")
    assert fn({"resp": "hi"}) == {"output": "hi"}


def test_load_extension_file_missing_raises(tmp_path: Path):
    with pytest.raises(InputParseError, match="not found"):
        load_extension_file(tmp_path / "does_not_exist.py")


def test_load_extension_file_with_syntax_error_wraps_in_input_parse_error(tmp_path: Path):
    file = tmp_path / "broken.py"
    file.write_text("def(\n")  # syntactically broken
    with pytest.raises(InputParseError, match="Failed to import"):
        load_extension_file(file)


def test_load_extension_file_twice_uses_unique_module_name(tmp_path: Path):
    """Loading the same file twice must not crash on sys.modules collision."""
    file = tmp_path / "twice.py"
    file.write_text(
        textwrap.dedent(
            """
            from ng_core import register_transform

            @register_transform("twice")
            def t(record):
                return record
            """
        ).strip()
    )
    load_extension_file(file)
    # Same callable identity on second load → idempotent register
    load_extension_file(file)
    assert list_transforms() == ["twice"]
