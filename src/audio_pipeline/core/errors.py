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


class MissingExtraError(StethoError):
    """A required optional extra is not installed."""

    def __init__(self, extra: str) -> None:
        self.extra = extra
        super().__init__(
            f"the {extra!r} extra is required; install it with `pip install stetho[{extra}]`",
        )


class NeedsConversionError(DatasetError):
    """The dataset's active audio layer is not in the required format."""

    def __init__(self, dataset: str, required: str, current: str | None) -> None:
        self.dataset = dataset
        self.required = required
        self.current = current
        have = f"is {current!r}" if current else "has no converted layer"
        super().__init__(f"dataset {dataset!r} {have}; run `convert --format {required}` first")


class ReplaceMediaError(DatasetError):
    """`--replace-media` was requested on a non-copy import."""

    def __init__(self, dataset: str, mode: str) -> None:
        self.dataset = dataset
        self.mode = mode
        super().__init__(f"--replace-media needs a 'copy' import; dataset {dataset!r} uses mode {mode!r}")


class MissingBinaryError(StethoError):
    """A required external binary is not on PATH."""

    def __init__(self, binary: str) -> None:
        self.binary = binary
        super().__init__(f"the {binary!r} binary is required but was not found on PATH")


class ProcessError(StethoError):
    """Base class for process-related errors."""


class DuplicateProcessError(ProcessError):
    """Two process classes claim the same name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"a process named {name!r} is already registered")


class ProcessNotFoundError(ProcessError):
    """No process is registered under the given name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"no process registered under {name!r}")


class ConversionError(ProcessError):
    """An audio file could not be converted."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        super().__init__(f"could not convert {path}: {detail}")


class ShardError(ProcessError):
    """A `--shard i/n` spec is malformed."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        super().__init__(f"invalid shard {spec!r}; expected i/n with 0 <= i < n")
