"""Thread-safe multi-line live terminal display for concurrent per-GPU
training progress (dev/checks.txt item 1: multi-GPU parallel search).

Without this, every GPU thread's progress-bar line update fought over the
same cursor position (each used a bare `\\r`-based in-place update, fine
for one GPU, garbled once 2+ threads print concurrently). This gives each
GPU its own pinned terminal line instead.

Built on `rich.live.Live` rather than hand-rolled ANSI cursor math -- it's
the well-tested tool for exactly this ("N concurrently-updating terminal
lines, thread-safe, correct across terminal emulators"), and regular
(non-progress) log lines still need to interleave correctly above the live
region, which `rich` handles natively as long as every terminal write goes
through the same Console instance.
"""

import threading
from typing import Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


class MultiGpuProgressDisplay:
    """One pinned line per GPU label, updated in place as progress arrives.
    Regular (non-progress) lines print normally above the live region via
    print_line() -- everything goes through the same rich Console, which is
    what keeps the two from clobbering each other.

    Every raw line from the remote process is wrapped in rich.text.Text
    (never a plain str) before display: train.py's stdout routinely
    contains literal `[...]` sequences (progress bars, `[hyperparams] ...`
    log lines) that rich's default markup parser would otherwise try to
    interpret as style tags.
    """

    def __init__(self, labels: List[str], console: Optional[Console] = None):
        self._console = console or Console()
        self._lock = threading.Lock()
        self._lines: Dict[str, str] = {label: "" for label in labels}
        self._live: Optional[Live] = None

    def _render(self) -> Table:
        table = Table.grid(padding=(0, 0))
        table.add_column()
        for label, line in self._lines.items():
            table.add_row(Text(line or f"[{label}] waiting..."))
        return table

    def __enter__(self) -> "MultiGpuProgressDisplay":
        self._live = Live(self._render(), console=self._console, refresh_per_second=8, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None

    def update_progress(self, label: str, line: str) -> None:
        """Overwrite this GPU's pinned line. Silently ignores an
        unregistered label rather than growing the table mid-render --
        the wave already knows every GPU it dispatched to before any
        thread starts, so this should never actually happen.
        """
        with self._lock:
            if label not in self._lines:
                return
            self._lines[label] = f"[{label}] {line}"
            if self._live is not None:
                self._live.update(self._render())

    def print_line(self, text: str) -> None:
        """A regular (non-progress) log line -- prints above the pinned
        progress rows via the same Console the Live display renders
        through."""
        with self._lock:
            self._console.print(Text(text))
