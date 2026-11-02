"""Progress reporting that survives Slurm logs.

A single `track` wraps an iterable with a progress display. On a terminal it
renders a rich animated bar; when stdout is redirected to a file (e.g. a Slurm
job) it falls back to throttled plain-text lines so the log still shows forward
motion. Turned off per-call (`--no-progress`) or globally (`STETHO_PROGRESS=0`).
"""

from __future__ import annotations

import os
import time
import typing as t
from collections.abc import Sized

import rich.progress

if t.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from rich.console import Console

_OFF = frozenset({"0", "false", "off", "no"})
_LINE_INTERVAL_S = 5.0  # seconds between plain-text progress lines


def progress_enabled() -> bool:
    """Whether progress is enabled by the environment.

    Honours STETHO_PROGRESS; any of 0/false/off/no (case-insensitive) disables
    it, absent or anything else leaves it on. The `--no-progress` flag is a
    separate, higher-precedence override resolved via `resolve_disable`.
    """
    return os.environ.get("STETHO_PROGRESS", "1").strip().lower() not in _OFF


def resolve_disable(*, no_progress_flag: bool) -> bool:
    """Combine the per-call flag with the env var (flag wins, then env)."""
    return no_progress_flag or not progress_enabled()


def track[ItemT](
    items: Iterable[ItemT],
    *,
    description: str,
    console: Console,
    total: int | None = None,
    disable: bool = False,
) -> Iterator[ItemT]:
    """Yield from `items`, reporting progress as a side effect.

    Renders a rich bar on a terminal, throttled plain lines on a non-terminal
    (e.g. a Slurm log), and nothing when `disable` is set.
    """
    if disable:
        yield from items
    elif console.is_terminal:
        yield from rich.progress.track(items, description=description, total=total, console=console)
    else:
        yield from _track_lines(items, description=description, console=console, total=total)


def _track_lines[ItemT](
    items: Iterable[ItemT],
    *,
    description: str,
    console: Console,
    total: int | None,
) -> Iterator[ItemT]:
    """Emit a flushed plain line at start, every few seconds, and at the end."""
    if total is None and isinstance(items, Sized):
        total = len(items)
    start = last = time.monotonic()
    done = 0
    _emit(console, description, done, total)
    for item in items:
        yield item
        done += 1
        now = time.monotonic()
        if now - last >= _LINE_INTERVAL_S:
            _emit(console, description, done, total)
            last = now
    _emit(console, description, done, total, elapsed=time.monotonic() - start)


def _emit(
    console: Console,
    description: str,
    done: int,
    total: int | None,
    elapsed: float | None = None,
) -> None:
    """Write one plain, flushed progress line to the console's file."""
    msg = f"{description}: {done}/{total} ({100 * done / total:.0f}%)" if total else f"{description}: {done}"
    if elapsed is not None:
        msg += f" in {elapsed:.1f}s"
    console.file.write(msg + "\n")
    console.file.flush()
