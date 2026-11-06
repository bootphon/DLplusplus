"""Process contract, item-level checkpointing, and the process registry.

A process wraps one extraction or transformation step over a dataset. Concrete
processes declare a `name`, `dependencies`, and a pydantic parameters model, and
register themselves by name. `ItemProcess` adds resumable per-item execution with
optional shard selection.
"""

from __future__ import annotations

import abc
import typing as t

import pydantic

from .errors import DuplicateProcessError, ProcessNotFoundError
from .progress import track
from .utils import require_binary

if t.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from rich.console import Console

    from .dataset import Dataset
    from .workspace import Workspace


class ProcessParams(pydantic.BaseModel):
    """Base parameters model for a process (hashed into the run_id in T8)."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class Shard(t.NamedTuple):
    """A 1-of-`count` slice of a process's items over a deterministic ordering."""

    index: int  # pyright: ignore[reportIncompatibleMethodOverride]
    count: int  # pyright: ignore[reportIncompatibleMethodOverride]

    def selects(self, position: int) -> bool:
        """Whether the item at `position` in the ordering belongs to this shard."""
        return position % self.count == self.index


class Process(abc.ABC):
    """A single extraction/transformation step over a dataset."""

    name: t.ClassVar[str]
    dependencies: t.ClassVar[tuple[str, ...]] = ()
    required_binaries: t.ClassVar[tuple[str, ...]] = ()
    Params: t.ClassVar[type[ProcessParams]] = ProcessParams

    def __init__(self, workspace: Workspace, dataset: Dataset, params: ProcessParams) -> None:
        self.workspace = workspace
        self.dataset = dataset
        self.params = params

    @abc.abstractmethod
    def is_complete(self) -> bool:
        """Whether every unit of work is already done."""

    @abc.abstractmethod
    def run(self) -> None:
        """Execute the process."""

    @abc.abstractmethod
    def on_complete(self) -> None:
        """Hook run once after a full (unsharded) successful run. Default: none."""


class ItemProcess[Item](Process):
    """A resumable process whose work is a set of items.

    Items already done are skipped, so an interrupted run resumes. An optional
    shard restricts execution to a disjoint 1-of-N slice. `items()` must yield in
    a deterministic order for sharding and resumption to be stable.
    """

    def __init__(
        self,
        workspace: Workspace,
        dataset: Dataset,
        params: ProcessParams,
        shard: Shard | None = None,
    ) -> None:
        super().__init__(workspace, dataset, params)
        self.shard = shard

    @abc.abstractmethod
    def items(self) -> Iterable[Item]:
        """Yield every unit of work in a deterministic order."""

    @abc.abstractmethod
    def is_done(self, item: Item) -> bool:
        """Whether one item's output already exists."""

    @abc.abstractmethod
    def process_item(self, item: Item) -> None:
        """Produce one item's output."""

    def selected_items(self) -> Iterator[Item]:
        """Items assigned to this shard (every item when unsharded)."""
        for position, item in enumerate(self.items()):
            if self.shard is None or self.shard.selects(position):
                yield item

    def is_complete(self) -> bool:
        """Whether all of this shard's items are done."""
        return all(self.is_done(item) for item in self.selected_items())

    def pending_items(self) -> Iterator[Item]:
        """Selected items whose output does not yet exist."""
        return (item for item in self.selected_items() if not self.is_done(item))

    def run(self) -> None:
        """Process each pending item, then finalize if the shard completed it all."""
        for item in self.pending_items():
            self.process_item(item)
        self._finish()

    def _finish(self) -> None:
        """Call `on_complete` when this is a full run that is now complete."""
        if self.shard is None and self.is_complete():
            self.on_complete()


_REGISTRY: dict[str, type[Process]] = {}


def register[ProcessT: type[Process]](cls: ProcessT) -> ProcessT:
    """Class decorator registering a process by its `name`.

    Raises:
        DuplicateProcessError: If another class already uses the same name.

    """
    if cls.name in _REGISTRY:
        raise DuplicateProcessError(cls.name)
    _REGISTRY[cls.name] = cls
    return cls


def get_process(name: str) -> type[Process]:
    """Look up a registered process class by name.

    Raises:
        ProcessNotFoundError: If no process is registered under `name`.

    """
    try:
        return _REGISTRY[name]
    except KeyError as err:
        raise ProcessNotFoundError(name) from err


def registered_processes() -> dict[str, type[Process]]:
    """A copy of the name → process-class registry."""
    return dict(_REGISTRY)


def _require_binaries(process: Process) -> None:
    """Preflight every external binary a process needs."""
    for binary in process.required_binaries:
        require_binary(binary)


def run_local(process: Process, *, console: Console, disable_progress: bool = False) -> int:
    """Run a process locally, returning the number of items processed.

    Shows per-item progress for checkpointed (`ItemProcess`) processes and runs
    plain processes through their own `run()`.

    Raises:
        MissingBinaryError: If a binary the process requires is not on PATH.

    """
    if not isinstance(process, ItemProcess):
        _require_binaries(process)
        process.run()
        return 0
    pending = list(process.pending_items())
    if pending:
        _require_binaries(process)
    for item in track(pending, description=process.name, console=console, total=len(pending), disable=disable_progress):
        process.process_item(item)
    process._finish()  # noqa: SLF001
    return len(pending)
