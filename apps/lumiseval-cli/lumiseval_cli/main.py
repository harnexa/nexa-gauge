"""LumisEval CLI entrypoint and compatibility exports."""

from __future__ import annotations

from typing import Any

import typer
from lumiseval_graph.runner import CachedNodeRunner

from lumiseval_cli.adapters import create_dataset_adapter
from .cli.estimate import estimate as estimate_command
from .cli.run import run as run_command
from .cli.util import (
    DEFAULT_FALLBACK_LLM,
    DEFAULT_PRIMARY_LLM,
    _collect_estimate_rows,
    _parse_model_overrides,
    _resolve_runtime_llm_overrides,
)

app = typer.Typer(name="lumiseval", help="Agentic LLM evaluation pipeline.")


@app.callback()
def cli() -> None:
    """LumisEval command group."""


app.command(name="run")(run_command)
app.command(name="estimate")(estimate_command)


def run(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible callable used by tests and programmatic callers."""
    import lumiseval_cli.cli.run as run_module

    run_module.CachedNodeRunner = CachedNodeRunner
    run_module.create_dataset_adapter = create_dataset_adapter
    run_module.run(*args, **kwargs)


def estimate(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible callable used by tests and programmatic callers."""
    import lumiseval_cli.cli.estimate as estimate_module

    estimate_module.CachedNodeRunner = CachedNodeRunner
    estimate_module.create_dataset_adapter = create_dataset_adapter
    estimate_module.estimate(*args, **kwargs)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
