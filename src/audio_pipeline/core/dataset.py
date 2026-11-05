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
from .errors import MissingAudioError, NeedsConversionError

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


def _suffix(file: AudioFile) -> str:
    """A file's lowercase extension without the dot."""
    return file.path.suffix.lstrip(".").lower()


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
        return self.workspace.manifest_file(self.name)

    def read_manifest(self) -> pl.DataFrame:
        """Read the Parquet manifest.

        Raises:
            FileNotFoundError: If the manifest does not exist.

        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        return pl.read_parquet(self.manifest_path)

    @abc.abstractmethod
    def audio_files(self, require: str | None = None) -> t.Iterator[AudioFile]:
        """Yield the audio files to process, preferring the converted layer.

        Raises:
            MissingAudioError: If audio has not been extracted (video datasets).
            NeedsConversionError: If `require` is set but the active layer is not
                that format.

        """


@dataclass(frozen=True)
class AudioDataset(Dataset):
    """A dataset whose source files are already audio."""

    def audio_files(self, require: str | None = None) -> t.Iterator[AudioFile]:
        """Yield audio files, preferring the converted layer when present."""
        converted = self.config.converted
        if converted is not None:
            if require is not None and converted.format != require:
                raise NeedsConversionError(self.name, require, converted.format)
            yield from self._layer_files(converted.format)
            return
        raw = list(self._raw_files())
        if require is not None and not all(_suffix(f) == require for f in raw):
            raise NeedsConversionError(self.name, require, None)
        yield from raw

    def _raw_files(self) -> t.Iterator[AudioFile]:
        """Yield the raw source audio from the manifest."""
        for row in self.read_manifest().iter_rows(named=True):
            yield AudioFile(item_id=row["item_id"], path=Path(row["path"]))

    def _layer_files(self, fmt: str) -> t.Iterator[AudioFile]:
        """Yield the converted-layer file for each manifest item."""
        base = self.workspace.audio_dir(self.name, fmt)
        for row in self.read_manifest().iter_rows(named=True):
            item_id = row["item_id"]
            yield AudioFile(item_id=item_id, path=base / f"{item_id}.{fmt}")


@dataclass(frozen=True)
class VideoDataset(Dataset):
    """A dataset of video files; audio must be extracted first."""

    required_processes: t.ClassVar[tuple[str, ...]] = ("extract_audio",)

    def audio_files(self, require: str | None = None) -> t.Iterator[AudioFile]:  # noqa: ARG002
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
    config = DatasetFile.load(workspace.dataset_toml(name)).config
    match config.type:
        case "audio":
            return AudioDataset(name=name, workspace=workspace, config=config)
        case "video":
            return VideoDataset(name=name, workspace=workspace, config=config)
        case _:
            t.assert_never(config.type)
