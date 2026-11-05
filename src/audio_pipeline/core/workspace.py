"""Definition of a workspace, helper for building relative paths."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from .config import MetaFile
from .errors import WorkspaceExistsError, WorkspaceNotFoundError

if t.TYPE_CHECKING:
    from pathlib import Path


Item = t.TypeVar("Item")
_META = "meta.toml"
_MANIFEST = "manifest.parquet"
_DATASET_TOML = "dataset.toml"
_REGISTRY = "registry"
_SOURCE = "source"
_AUDIO = "audio"
_RUNS = "runs"
_ANNOTATIONS = "annotations"
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

    def registry_dir(self, dataset: str) -> Path:
        """Path to a dataset's registry entry (dataset.toml + manifest)."""
        return self.root / _REGISTRY / dataset

    def manifest_file(self, dataset: str) -> Path:
        """Path to a dataset's Parquet manifest."""
        return self.registry_dir(dataset) / _MANIFEST

    def dataset_toml(self, dataset: str) -> Path:
        """Path to a dataset's dataset.toml."""
        return self.registry_dir(dataset) / _DATASET_TOML

    def source_dir(self, dataset: str) -> Path:
        """Path to a dataset's imported source media."""
        return self.root / _SOURCE / dataset

    def audio_dir(self, dataset: str, fmt: str) -> Path:
        """Path to a dataset's converted audio for a given format."""
        return self.root / _AUDIO / fmt / dataset

    def run_dir(self, process: str, dataset: str, run_id: str) -> Path:
        """Path to the run directory for a given run instance."""
        return self.root / _RUNS / process / dataset / run_id

    def annotations_dir(self, process: str, dataset: str) -> Path:
        """Path to a process's exported annotations for a dataset."""
        return self.root / _ANNOTATIONS / process / dataset

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
        for sub in (_REGISTRY, _SOURCE, _RUNS, _ANNOTATIONS, _PLOTS):
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
