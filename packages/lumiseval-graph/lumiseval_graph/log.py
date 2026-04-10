"""
Colored node logger for the LumisEval pipeline.

Each node gets its own color so you can visually distinguish it in the terminal.
Uses Rich for colored output. Import get_node_logger() in each node module.
"""

from datetime import datetime

from lumiseval_graph.topology import NODES_BY_NAME
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_console = Console(highlight=False)


class NodeLogger:
    """Rich-colored logger scoped to a single pipeline node.

    Each instance has a fixed node name and color. Use .start() at node entry,
    .info() for intermediate details, .success() on clean completion,
    .warning() for recoverable issues, .error() for failures.
    """

    def __init__(self, node_name: str) -> None:
        self.node = node_name
        self.color = NODES_BY_NAME[node_name].color if node_name in NODES_BY_NAME else "white"

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _prefix(self) -> Text:
        t = Text()
        label = f"[{self.node}]"
        # Pad label to 20 chars so columns line up across nodes
        t.append(f"{label:<20}", style=f"bold {self.color}")
        t.append(f" {self._ts()} ", style="dim")
        return t

    def start(self, msg: str) -> None:
        t = self._prefix()
        t.append("▶  ", style=f"bold {self.color}")
        t.append(msg)
        _console.print(t)

    def info(self, msg: str) -> None:
        t = self._prefix()
        t.append("   ", style="dim")
        t.append(msg, style="dim")
        _console.print(t)

    def success(self, msg: str) -> None:
        t = self._prefix()
        t.append("✓  ", style="bold green")
        t.append(msg)
        _console.print(t)

    def warning(self, msg: str) -> None:
        t = self._prefix()
        t.append("⚠  ", style="bold yellow")
        t.append(msg, style="yellow")
        _console.print(t)

    def error(self, msg: str) -> None:
        t = self._prefix()
        t.append("✗  ", style="bold red")
        t.append(msg, style="red")
        _console.print(t)


def get_node_logger(node_name: str) -> NodeLogger:
    """Return a NodeLogger for the given pipeline node name."""
    return NodeLogger(node_name)


def print_pipeline_header(job_id: str, model: str, web_search: bool) -> None:
    """Print a banner at the start of a pipeline run."""
    web = "[green]on[/green]" if web_search else "[dim]off[/dim]"
    _console.print()
    _console.print(
        Panel(
            f"[bold]Job:[/bold] [dim]{job_id}[/dim]"
            f"  ·  [bold]Model:[/bold] {model}"
            f"  ·  [bold]Web search:[/bold] {web}",
            title="[bold cyan]LumisEval  —  evaluation pipeline[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
    _console.print()


def print_pipeline_footer(composite_score: float | None, cost_usd: float) -> None:
    """Print a summary banner at the end of a pipeline run."""
    score_str = (
        f"[bold green]{composite_score:.4f}[/bold green]"
        if composite_score is not None
        else "[dim]n/a[/dim]"
    )
    _console.print()
    _console.print(
        Panel(
            f"[bold]Composite score:[/bold] {score_str}"
            f"  ·  [bold]Actual cost:[/bold] [yellow]${cost_usd:.6f}[/yellow]",
            title="[bold green]Pipeline complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    _console.print()
