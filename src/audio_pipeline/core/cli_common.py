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
    from .process import Process, ProcessParams

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

    def process_class(self) -> type[Process]:
        """The process class this command runs (source of its name + Params)."""
        raise NotImplementedError

    def param_overrides(self) -> dict[str, t.Any]:
        """CLI params explicitly provided (non-None); these override meta.toml."""
        return {}

    def build_process(self, workspace: Workspace, dataset: Dataset, params: ProcessParams) -> Process:
        """Construct the process from the already-resolved params."""
        raise NotImplementedError

    def post_run(self, workspace: Workspace, dataset: Dataset, process: Process) -> None:
        """Optional side effects after a successful run (default: none)."""

    def _resolve_params(self, workspace: Workspace) -> ProcessParams:
        """Resolve defaults < meta.toml < CLI args, persist the result, reload it.

        The effective params are written into `[modules.<name>]` before the run so
        meta.toml records exactly what executed.
        """
        process_cls = self.process_class()
        meta = workspace.meta()
        merged = {**meta.config.modules.get(process_cls.name, {}), **self.param_overrides()}
        resolved = process_cls.Params.model_validate(merged)
        meta.set_module_config(process_cls.name, resolved.model_dump(mode="json", exclude_none=True))
        meta.save()
        reloaded = workspace.meta()
        return process_cls.Params.model_validate(reloaded.config.modules.get(process_cls.name, {}))

    async def run(self) -> None:
        """Resolve + persist params, then run the process locally."""
        workspace = Workspace.open(self.workspace)
        dataset = load_dataset(workspace, self.dataset)
        params = self._resolve_params(workspace)
        process = self.build_process(workspace, dataset, params)
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
