from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass

from .config import MetaFile
from .errors import WorkspaceExistsError, WorkspaceNotFoundError

if t.TYPE_CHECKING:
    from pathlib import Path

    from .dataset import Dataset

Item = t.TypeVar("Item")
_META = "meta.toml"
_DATASETS = "datasets"
_RUNS = "runs"
_RESULTS = "results"
_PLOTS = "plots"


@dataclass(frozen=True)
class Workspace:
    """Single source of truth for every path inside a workspace root.

    The root is resolved on construction so path comparisons are reliable.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def meta_file(self) -> Path:
        """Path to the meta.toml file."""
        return self.root / _META

    def dataset_dir(self, dataset: str) -> Path:
        """Path to the dataset.toml file (in each dataset)."""
        return self.root / _DATASETS / dataset

    def run_dir(self, process: str, dataset: str, run_id: str) -> Path:
        """Path to the run directory for a given run instance."""
        return self.root / _RUNS / process / dataset / run_id

    def results_dir(self, dataset: str) -> Path:
        """Path to the exported results for a given dataset."""
        return self.root / _RESULTS / dataset

    def plots_dir(self, dataset: str) -> Path:
        """Path to the plot directory for a dataset."""
        return self.root / _PLOTS / dataset

    @classmethod
    def init(cls, root: Path) -> Workspace:
        """Create a new workspace tree and a default meta.toml.

        Raises:
            WorkspaceExistsError: If a meta.toml already exists at root.

        """
        ws = cls(root)
        if ws.meta_file.exists():
            raise WorkspaceExistsError(ws.root)
        for sub in (_DATASETS, _RUNS, _RESULTS, _PLOTS):
            (ws.root / sub).mkdir(parents=True, exist_ok=True)
        MetaFile.create_default(ws.meta_file).save()
        return ws

    @classmethod
    def open(cls, root: Path) -> Workspace:
        """Open and validate an existing workspace.

        Raises:
            WorkspaceNotFoundError: If root has no meta.toml.
            ConfigError: If meta.toml fails validation.

        """
        ws = cls(root)
        if not ws.meta_file.exists():
            raise WorkspaceNotFoundError(ws.root)
        MetaFile.load(ws.meta_file)
        return ws

    def meta(self) -> MetaFile:
        """Load this workspace's meta.toml."""
        return MetaFile.load(self.meta_file)


class Process(abc.ABC):
    """Metadata extraction step.

    Wraps one external tool; declares dependencies by process name.

    Raises:
        UnsatisfiedDependencyError: If a dependency is not complete.

    """

    name: t.ClassVar[str]
    dependencies: t.ClassVar[tuple[str, ...]] = ()

    def __init__(self, workspace: Workspace, dataset: Dataset) -> None:
        self.workspace = workspace
        self.dataset = dataset

    @abc.abstractmethod
    def is_complete(self) -> bool: ...

    @abc.abstractmethod
    def run(self) -> None: ...


class ItemProcess[Item](Process):
    """Checkpointed process.

    Iterates over self-defined work items and skips those already done.

    Raises:
        NotImplementedError: If a subclass omits an item hook.

    """

    @abc.abstractmethod
    def items(self) -> t.Iterable[Item]: ...

    @abc.abstractmethod
    def is_done(self, item: Item) -> bool: ...

    @abc.abstractmethod
    def process_item(self, item: Item) -> None: ...

    def run(self) -> None:
        """Rune fn on each item?."""
        for item in self.items():
            if not self.is_done(item):
                self.process_item(item)


def missing_dependencies(process: Process, registry: dict[str, type[Process]]) -> list[str]:
    """Find unsatisfied dependencies.

    Instantiates each declared dependency on the same workspace and
    dataset and checks whether its outputs are complete.

    Raises:
        KeyError: If a dependency name is not registered.

    """
    return [dep for dep in process.dependencies if not registry[dep](process.workspace, process.dataset).is_complete()]
