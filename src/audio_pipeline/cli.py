"""`stetho` command-line interface."""

from __future__ import annotations

import datetime as dt
import sys
import typing as t
from pathlib import Path  # noqa: TC003

import rich.console
from clypi import Command, Positional, arg

from .core.config import DatasetType  # noqa: TC001
from .core.errors import StethoError
from .core.importer import LocalMode, run_import
from .core.workspace import Workspace

__all__ = ["main"]

_OUT = rich.console.Console()
_ERR = rich.console.Console(stderr=True)


def _timestamp() -> str:
    """Current UTC time, ISO-8601 to the second."""
    return dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")


class Init(Command):
    """Initialise a new workspace at the given path."""

    workspace: Positional[Path] = arg(inherited=True)

    async def run(self) -> None:
        """Create the workspace tree and a default meta.toml.

        Raises:
            WorkspaceExistsError: If the workspace already exists.

        """
        ws = Workspace.init(self.workspace)
        _OUT.print(f"[green]initialised workspace[/] {ws.root}")


class Import(Command):
    """Import a local folder as a dataset."""

    # `source` must precede the inherited `workspace`: clypi binds positional
    # args by declaration order, so the single positional goes to `source` and
    # `workspace` falls through to the parent command's value.
    source: Positional[Path] = arg(help="folder of media to import")
    workspace: Positional[Path] = arg(inherited=True)
    name: str = arg(help="dataset name")
    type: DatasetType = arg(help="dataset media type")
    mode: LocalMode = arg(default="symlink", help="copy, symlink, or reference")

    async def run(self) -> None:
        """Import SOURCE into the workspace as a new dataset.

        Raises:
            WorkspaceNotFoundError: If the workspace is not initialised.
            DatasetExistsError: If the dataset name already exists.
            NoMediaFoundError: If no matching media is found.
            ProbeError: If a media file cannot be probed.

        """
        count = run_import(
            Workspace.open(self.workspace),
            self.source,
            self.name,
            self.type,
            self.mode,
        )
        _OUT.print(f"[green]imported[/] {count} item(s) into {self.name!r}")


class Stetho(Command):
    """Orchestrate metadata extraction on audio/video datasets."""

    workspace: Positional[Path] = arg(help="path to the workspace root")
    subcommand: Init | Import | None = None

    async def post_run_hook(self, exception: Exception | None) -> None:
        """Append a successful invocation to the workspace command log."""
        if exception is not None or self.subcommand is None:
            return
        meta = Workspace(self.workspace).meta()
        meta.log_command(_timestamp(), sys.argv[1:])
        meta.save()


def _abort(err: StethoError) -> t.NoReturn:
    """Render a domain error without a traceback and exit non-zero."""
    _ERR.print(f"[bold red]error:[/] {err}")
    sys.exit(1)


def main() -> None:
    """Parse argv and run the `stetho` command tree."""
    try:
        exc = Stetho.parse().start()
    except StethoError as err:
        _abort(err)
    if exc is not None:
        sys.exit(1)
