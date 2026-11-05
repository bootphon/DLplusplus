"""Shared helpers for the root CLI and per-module command modules.

Kept separate from `cli.py` so module command modules can import the console and
shard parser without importing the root command (which imports them back).
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import rich.console
from clypi import Command, Positional, arg

from .dataset import load_dataset
from .errors import ShardError
from .process import Shard, run_local
from .progress import resolve_disable
from .workspace import Workspace

if t.TYPE_CHECKING:
    from .dataset import Dataset
    from .process import Process

OUT = rich.console.Console()
ERR = rich.console.Console(stderr=True)


class ProcessCommand(Command):
    """Base for a local process subcommand: shared params + run flow.

    Subclasses re-declare these fields (clypi reads each command's own
    annotations, so base fields are not inherited) and implement `build_process`;
    the `run` method below is inherited normally.
    """

    workspace: Positional[Path] = arg(inherited=True)
    dataset: str = arg(help="dataset to process")
    shard: str | None = arg(default=None, help="process only shard i/n, e.g. 0/4")
    no_progress: bool = arg(default=False, help="disable the progress display (also via STETHO_PROGRESS=0)")

    def build_process(self, workspace: Workspace, dataset: Dataset) -> Process:
        """Construct the process to run from this command's params."""
        raise NotImplementedError

    def post_run(self, workspace: Workspace, dataset: Dataset, process: Process) -> None:
        """Optional side effects after a successful run (default: none)."""

    async def run(self) -> None:
        """Load the dataset, run the process locally, then any post-run step."""
        workspace = Workspace.open(self.workspace)
        dataset = load_dataset(workspace, self.dataset)
        process = self.build_process(workspace, dataset)
        count = run_local(process, console=OUT, disable_progress=resolve_disable(no_progress_flag=self.no_progress))
        self.post_run(workspace, dataset, process)
        OUT.print(f"[green]{process.name}[/] processed {count} item(s) in {self.dataset!r}")


def parse_shard(spec: str | None) -> Shard | None:
    """Parse an `i/n` shard spec into a Shard, or None.

    Raises:
        ShardError: If the spec is malformed or out of range.

    """
    if spec is None:
        return None
    index_str, sep, count_str = spec.partition("/")
    try:
        index, count = int(index_str), int(count_str)
    except ValueError as err:
        raise ShardError(spec) from err
    if not sep or count < 1 or not (0 <= index < count):
        raise ShardError(spec)
    return Shard(index, count)
