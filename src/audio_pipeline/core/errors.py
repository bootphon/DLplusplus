"""Domain exceptions for the package."""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from pathlib import Path


class StethoError(Exception):
    """Base class for every domain-specific error raised by the package."""


class WorkspaceError(StethoError):
    """Base class for workspace-related errors."""


class WorkspaceNotFoundError(WorkspaceError):
    """No initialised workspace at the given root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        super().__init__(f"no workspace at {root} (missing meta.toml)")


class WorkspaceExistsError(WorkspaceError):
    """A workspace is already initialised at the given root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        super().__init__(f"workspace already initialised at {root}")


class DatasetError(StethoError):
    """Base class for dataset-related errors."""


class MissingAudioError(DatasetError):
    """A video dataset's audio has not been extracted yet."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        super().__init__(
            f"dataset {dataset!r} has no extracted audio; run `extract_audio` first",
        )


class DatasetExistsError(DatasetError):
    """A dataset with the given name already exists."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"dataset {name!r} already exists")


class NoMediaFoundError(DatasetError):
    """No media of the requested type was found under the source."""

    def __init__(self, source: Path, dataset_type: str) -> None:
        self.source = source
        self.dataset_type = dataset_type
        super().__init__(f"no {dataset_type} files found under {source}")


class ProbeError(DatasetError):
    """A media file could not be probed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"could not probe media file {path}")
