"""User-defined record transforms for arbitrary dataset schemas.

A *transform* is a pure function that reshapes one raw record into the dict
shape the rest of nexa-gauge expects: keys for any subset of ``case_id``,
``question``, ``generation``, ``context``, ``reference``. ``geval`` and
``redteam`` are nexa-gauge metric configs (not dataset data) and are
intentionally out of scope for transforms.

Transforms are registered by name via :func:`register_transform` and looked
up by :func:`get_transform`. User extension files are loaded via the
shared :func:`ng_core.extensions.load_extension_file`, which executes the
file's decorators as a side effect and populates this registry (and any
other extension registry — prompts, metrics, etc. — that lands later).

The registry is process-global. It is also the dispatch surface a future
Python API (``evaluate(record, transform="hotpot_qa")``) or REST endpoint
would reuse without any extra plumbing.
"""

from __future__ import annotations

from typing import Any, Callable

from ng_core.errors import InputParseError

TransformFn = Callable[[dict[str, Any]], dict[str, Any]]

_REGISTRY: dict[str, TransformFn] = {}


def register_transform(name: str) -> Callable[[TransformFn], TransformFn]:
    """Register ``fn`` under ``name`` in the process-global transform registry.

    Semantics: **last register wins**. Re-loading the same transform file
    (a common case in long-running REST/notebook processes) creates a fresh
    function object, so strict identity checks would crash. Users who care
    about strict uniqueness can inspect :func:`list_transforms` before
    registering.
    """
    if not name or not isinstance(name, str):
        raise ValueError("register_transform: name must be a non-empty string")

    def deco(fn: TransformFn) -> TransformFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_transform(name: str) -> TransformFn:
    """Return the transform registered under ``name`` or raise ``InputParseError``."""
    fn = _REGISTRY.get(name)
    if fn is None:
        registered = sorted(_REGISTRY)
        hint = f"Registered: {registered}" if registered else "No transforms registered."
        raise InputParseError(
            f"Transform '{name}' is not registered. "
            f"Did you forget --extension-file, or mistype the name? {hint}"
        )
    return fn


def list_transforms() -> list[str]:
    """Return the sorted names of currently registered transforms."""
    return sorted(_REGISTRY)


def _clear_registry_for_tests() -> None:
    """Internal: reset the registry. Tests only."""
    _REGISTRY.clear()
