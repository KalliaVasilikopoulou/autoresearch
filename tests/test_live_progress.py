"""Synthetic-data tests for agents/live_progress.py's multi-GPU live
terminal display (dev/checks.txt item 1 follow-up): each concurrent GPU
gets its own pinned line instead of every thread's old \\r-based update
fighting over the same cursor position.

Uses an isolated rich.console.Console(file=io.StringIO()) throughout, both
to keep rich's ANSI escape codes out of pytest's captured output and to
avoid depending on real-terminal detection quirks.
"""

import io
import threading

from rich.console import Console

from agents.live_progress import MultiGpuProgressDisplay


def _isolated_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def test_update_progress_stores_line_for_registered_label():
    console = _isolated_console()
    display = MultiGpuProgressDisplay(["GPU1", "GPU2"], console=console)
    with display:
        display.update_progress("GPU1", "[====----------------]  20.0% | loss: 5.0")
    assert display._lines["GPU1"] == "[GPU1] [====----------------]  20.0% | loss: 5.0"


def test_update_progress_ignores_unregistered_label():
    console = _isolated_console()
    display = MultiGpuProgressDisplay(["GPU1"], console=console)
    with display:
        display.update_progress("GPU99", "some progress line")
    assert "GPU99" not in display._lines
    assert display._lines["GPU1"] == ""


def test_print_line_does_not_raise_and_writes_to_console():
    console = _isolated_console()
    display = MultiGpuProgressDisplay(["GPU1"], console=console)
    with display:
        display.print_line("[GPU1] Connecting to host ...")
    output = console.file.getvalue()
    assert "Connecting to host" in output


def test_print_line_handles_literal_brackets_without_markup_errors():
    """train.py's stdout routinely contains literal `[...]` sequences
    (progress bars, `[hyperparams] ...` lines) that rich's default markup
    parser would otherwise try to interpret as style tags -- must not raise
    or silently eat the text."""
    console = _isolated_console()
    display = MultiGpuProgressDisplay(["GPU1"], console=console)
    with display:
        display.print_line("[GPU1] [hyperparams] DEPTH=8 N_HEAD=4 [not a style tag]")
    output = console.file.getvalue()
    assert "[hyperparams] DEPTH=8" in output
    assert "[not a style tag]" in output


def test_context_manager_enter_exit_does_not_raise():
    console = _isolated_console()
    display = MultiGpuProgressDisplay(["GPU1", "GPU2"], console=console)
    with display:
        display.update_progress("GPU1", "50%")
        display.update_progress("GPU2", "75%")
    # exiting cleanly (no exception) is the assertion


def test_concurrent_updates_from_multiple_threads_do_not_corrupt_state():
    console = _isolated_console()
    labels = [f"GPU{i}" for i in range(4)]
    display = MultiGpuProgressDisplay(labels, console=console)

    def worker(label):
        for step in range(20):
            display.update_progress(label, f"{step * 5}%")

    with display:
        threads = [threading.Thread(target=worker, args=(label,)) for label in labels]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Each label's final line reflects its own last update, not another
    # thread's -- proves the shared dict/lock didn't let updates cross-talk.
    for label in labels:
        assert display._lines[label] == f"[{label}] 95%"
