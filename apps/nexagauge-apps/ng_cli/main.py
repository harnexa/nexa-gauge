"""NexaGauge CLI entrypoint and compatibility exports."""

from __future__ import annotations

from typing import Any

import typer
from adapters import create_dataset_adapter
from ng_graph.runner import CachedNodeRunner

from .delete import delete_app
from .estimate import estimate as estimate_command
from .run import run as run_command
from .util import (
    DEFAULT_FALLBACK_LLM,
    DEFAULT_PRIMARY_LLM,
    _collect_estimate_rows,
    _is_case_eligible_for_target_path,
    _parse_model_overrides,
    _resolve_runtime_llm_overrides,
)

app = typer.Typer(name="nexagauge", help="Agentic LLM evaluation pipeline.")

__all__ = [
    "app",
    "main",
    "run",
    "estimate",
    "DEFAULT_PRIMARY_LLM",
    "DEFAULT_FALLBACK_LLM",
    "_collect_estimate_rows",
    "_is_case_eligible_for_target_path",
    "_parse_model_overrides",
    "_resolve_runtime_llm_overrides",
]


@app.callback()
def cli() -> None:
    """NexaGauge command group."""


app.command(name="run")(run_command)
app.command(name="estimate")(estimate_command)
app.add_typer(delete_app, name="delete")


def run(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible callable used by tests and programmatic callers."""
    import ng_cli.run as run_module

    run_module.CachedNodeRunner = CachedNodeRunner
    run_module.create_dataset_adapter = create_dataset_adapter
    run_module.run(*args, **kwargs)


def estimate(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible callable used by tests and programmatic callers."""
    import ng_cli.estimate as estimate_module

    estimate_module.CachedNodeRunner = CachedNodeRunner
    estimate_module.create_dataset_adapter = create_dataset_adapter
    estimate_module.estimate(*args, **kwargs)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
