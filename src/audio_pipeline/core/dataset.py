"""Dataset models and the Parquet manifest schema.

A dataset is backed by a manifest (Parquet) describing its media items.
`AudioDataset` iterates its source audio directly; `VideoDataset` iterates
the audio extracted by the `extract_audio` process.
"""

from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import DatasetFile
from .errors import MissingAudioError

if t.TYPE_CHECKING:
    from .config import DatasetConfig
    from .workspace import Workspace


MANIFEST_SCHEMA = pl.Schema(
    {
        "item_id": pl.String(),
        "path": pl.String(),
        "media_type": pl.String(),
        "duration": pl.Float64(),
        "sample_rate": pl.Int32(),
        "channels": pl.Int32(),
    },
)

_MANIFEST = "manifest.parquet"
_DATASET_TOML = "dataset.toml"


class AudioFile(t.NamedTuple):
    """An audio file to process: a stable id and its path."""

    item_id: str
    path: Path


@dataclass(frozen=True)
class Dataset(abc.ABC):
    """A dataset: its manifest plus how to iterate its audio."""

    name: str
    workspace: Workspace
    config: DatasetConfig

    required_processes: t.ClassVar[tuple[str, ...]] = ()

    @property
    def manifest_path(self) -> Path:
        """Path to this dataset's Parquet manifest."""
        return self.workspace.dataset_dir(self.name) / _MANIFEST

    def read_manifest(self) -> pl.DataFrame:
        """Read the Parquet manifest.

        Raises:
            FileNotFoundError: If the manifest does not exist.

        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        return pl.read_parquet(self.manifest_path)

    @abc.abstractmethod
    def audio_files(self) -> t.Iterator[AudioFile]:
        """Yield the audio files to process.

        Raises:
            MissingAudioError: If audio has not been extracted (video datasets).

        """


@dataclass(frozen=True)
class AudioDataset(Dataset):
    """A dataset whose source files are already audio."""

    def audio_files(self) -> t.Iterator[AudioFile]:
        """Yield each source audio file from the manifest."""
        for row in self.read_manifest().iter_rows(named=True):
            yield AudioFile(item_id=row["item_id"], path=Path(row["path"]))


@dataclass(frozen=True)
class VideoDataset(Dataset):
    """A dataset of video files; audio must be extracted first."""

    required_processes: t.ClassVar[tuple[str, ...]] = ("extract_audio",)

    def audio_files(self) -> t.Iterator[AudioFile]:
        """Yield each extracted audio file.

        Raises:
            MissingAudioError: If `extract_audio` has no kept run yet.

        """
        if "extract_audio" not in self.config.runs:
            raise MissingAudioError(self.name)
        raise NotImplementedError  # extracted-audio read is wired in T11


def load_dataset(workspace: Workspace, name: str) -> Dataset:
    """Load a dataset from its dataset.toml, returning the right subclass.

    Raises:
        FileNotFoundError: If dataset.toml does not exist.
        ConfigError: If dataset.toml fails validation.

    """
    config = DatasetFile.load(workspace.dataset_dir(name) / _DATASET_TOML).config
    match config.type:
        case "audio":
            return AudioDataset(name=name, workspace=workspace, config=config)
        case "video":
            return VideoDataset(name=name, workspace=workspace, config=config)
        case _:
            t.assert_never(config.type)
