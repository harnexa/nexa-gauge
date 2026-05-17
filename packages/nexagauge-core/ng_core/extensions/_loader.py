"""Shared loader for user-extension Python files.

Generic across every extension type (transforms today; prompts, metrics,
rubrics in the future). The loader doesn't know what the file registers —
it just executes the module so its decorators run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import uuid4

from ng_core.errors import InputParseError


def load_extension_file(path: str | Path) -> None:
    """Import ``path`` as a Python module so its ``@register_*`` decorators run.

    Uses ``importlib.util.spec_from_file_location`` so the file does not need
    to be on ``sys.path``. A unique module name is generated per load to
    avoid ``sys.modules`` collisions when a long-running process (REST
    server, notebook) loads the same file twice.

    Wraps any import-time exception in :class:`InputParseError` so the CLI's
    existing error rendering applies uniformly.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise InputParseError(f"Extension file not found: {file_path}")

    module_name = f"_nexagauge_user_extensions_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise InputParseError(f"Could not load extension file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise InputParseError(f"Failed to import extension file {file_path}: {exc}") from exc
