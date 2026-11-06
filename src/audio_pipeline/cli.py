"""`stetho` command-line interface."""

from __future__ import annotations

import datetime as dt
import sys
import typing as t
from pathlib import Path

import clypi
from clypi import Command, Positional, arg

from .core.cli_common import ERR as _ERR
from .core.cli_common import OUT as _OUT
from .core.command_log import append_command
from .core.config import DatasetType
from .core.errors import StethoError
from .core.importer import LocalMode, run_hf_import, run_import
from .core.progress import resolve_disable
from .core.workspace import Workspace
from .modules.audio_tools.cli import Convert

__all__ = ["main"]


def _timestamp() -> str:
    """Current UTC time, ISO-8601 to the second."""
    return dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")


class HidingFormatter(clypi.ClypiFormatter):
    def format_help(
        self,
        full_command,
        description,
        epilog,
        options,
        positionals,
        subcommands,
        exception,
    ) -> str:
        visible = [s for s in subcommands if not getattr(s, "_hidden", False)]
        return super().format_help(full_command, description, epilog, options, positionals, visible, exception)


clypi.configure(clypi.ClypiConfig(help_formatter=HidingFormatter()))


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


class ResetMeta(Command):
    """Regenerate meta.toml from defaults (discards settings and command log)."""

    workspace: Positional[Path] = arg(inherited=True)
    _hidden = True

    async def run(self) -> None:
        """Overwrite meta.toml with a fresh default.

        Raises:
            WorkspaceNotFoundError: If the workspace is not initialised.

        """
        ws = Workspace(self.workspace)
        ws.reset_meta()
        _OUT.print(f"[green]reset[/] meta.toml at {ws.meta_file}")


class Import(Command):
    """Import a local folder as a dataset."""

    # `source` must precede the inherited `workspace`: clypi binds positional
    # args by declaration order, so the single positional goes to `source` and
    # `workspace` falls through to the parent command's value.
    source: Positional[Path] = arg(help="folder of media to import")
    workspace: Positional[Path] = arg(inherited=True)
    name: str = arg(help="dataset name")
    type: DatasetType = arg(default="audio", help="dataset media type")
    mode: LocalMode = arg(default="symlink", help="copy, symlink, or reference")
    no_progress: bool = arg(default=False, help="disable the progress display (also via STETHO_PROGRESS=0)")

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
            console=_OUT,
            disable_progress=resolve_disable(no_progress_flag=self.no_progress),
        )
        _OUT.print(f"[green]imported[/] {count} item(s) into {self.name!r}")


class ImportHF(Command):
    """Import a Hugging Face audio or video dataset."""

    # clypi derives the command name from the class name via `_camel_to_dashed`,
    # which would turn `ImportHF` into `import-h-f`. `prog()`
    @classmethod
    def prog(cls) -> str:
        return "import-hf"

    # `repo` must precede the inherited `workspace` (clypi binds positionals by
    # declaration order), mirroring `Import.source`.
    repo: Positional[str] = arg(help="Hugging Face dataset repo id")
    workspace: Positional[Path] = arg(inherited=True)
    name: str = arg(help="dataset name")
    type: DatasetType = arg(default="audio", help="dataset media type")
    revision: str | None = arg(default=None, help="git revision (branch, tag, or commit)")
    no_progress: bool = arg(default=False, help="disable the progress display (also via STETHO_PROGRESS=0)")

    async def run(self) -> None:
        """Download the HF dataset into the workspace as a new dataset.

        Raises:
            WorkspaceNotFoundError: If the workspace is not initialised.
            MissingExtraError: If the `hf` extra is not installed.
            DatasetExistsError: If the dataset name already exists.
            HFImportError: If the dataset has no usable audio column.
            ProbeError: If a materialised audio file cannot be probed.

        """
        count = run_hf_import(
            Workspace.open(self.workspace),
            self.repo,
            self.name,
            dataset_type=self.type,
            revision=self.revision,
            console=_OUT,
            disable_progress=resolve_disable(no_progress_flag=self.no_progress),
        )
        _OUT.print(f"[green]imported[/] {count} item(s) into {self.name!r}")


class Process(Command):
    """Local process runner."""

    workspace: Positional[Path] = arg(inherited=True)
    subcommand: Convert = arg()


class Stetho(Command):
    """Orchestrate metadata extraction on audio/video datasets."""

    workspace: Positional[Path] = arg(help="path to the workspace root")
    subcommand: Init | ResetMeta | Import | ImportHF | Process | None = None

    async def post_run_hook(self, exception: Exception | None) -> None:
        """Record a successful invocation in the workspace command log."""
        if exception is not None or self.subcommand is None:
            return
        append_command(Workspace(self.workspace).command_log_file, _timestamp(), sys.argv[1:])


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
