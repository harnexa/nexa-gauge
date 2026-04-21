from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import typer
from ng_core.constants import default_cache_dir

from .util import console

delete_app = typer.Typer(name="delete", help="Remove cached or generated artifacts.")


_UNIT_STEP = 1024.0
_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def _human_bytes(size: float) -> str:
    for unit in _UNITS:
        if size < _UNIT_STEP or unit == _UNITS[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= _UNIT_STEP
    return f"{size:.1f} {_UNITS[-1]}"


def _directory_size(path: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                total_bytes += file_path.stat().st_size
                total_files += 1
            except (FileNotFoundError, PermissionError):
                continue
    return total_bytes, total_files


def _resolve_cache_root(cache_dir: Optional[str]) -> Path:
    if cache_dir:
        return Path(cache_dir)
    env_value = os.getenv("NEXAGAUGE_CACHE_DIR")
    if env_value:
        return Path(env_value)
    return default_cache_dir()


@delete_app.command("cache")
def delete_cache(
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Explicit cache directory to wipe. Defaults to $NEXAGAUGE_CACHE_DIR or the per-user cache path.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be freed without deleting anything.",
    ),
) -> None:
    """Wipe the node-level execution cache."""
    root = _resolve_cache_root(cache_dir).expanduser().resolve()

    if not root.exists():
        console.print(f"[dim]No cache found at[/dim] {root}")
        raise typer.Exit(0)

    if not root.is_dir():
        console.print(f"[red]Refusing to delete: {root} is not a directory.[/red]")
        raise typer.Exit(1)

    total_bytes, total_files = _directory_size(root)
    pretty_size = _human_bytes(total_bytes)

    console.print(f"[bold]Cache location:[/bold] {root}")
    console.print(f"[bold]Files:[/bold] {total_files:,}   [bold]Size:[/bold] {pretty_size}")

    if dry_run:
        console.print("[yellow]Dry run — nothing was deleted.[/yellow]")
        raise typer.Exit(0)

    if total_files == 0:
        console.print("[dim]Cache is already empty.[/dim]")
        raise typer.Exit(0)

    if not yes:
        confirmed = typer.confirm(f"Delete {total_files:,} files ({pretty_size})?", default=False)
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(1)

    try:
        shutil.rmtree(root)
    except OSError as exc:
        console.print(f"[red]Failed to delete cache:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[bold green]Freed {pretty_size}[/bold green] ({total_files:,} files)")
