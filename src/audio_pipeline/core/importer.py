"""Local dataset importers and the manifest builder.

Discovers media under a source folder, probes each file, places it into the
workspace per the import mode, and writes the Parquet manifest and dataset.toml.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import typing as t
from pathlib import Path

import polars as pl
import soundfile

from .config import DatasetConfig, DatasetFile
from .dataset import MANIFEST_SCHEMA
from .errors import DatasetExistsError, NoMediaFoundError, ProbeError

if t.TYPE_CHECKING:
    from .config import DatasetType
    from .workspace import Workspace


LocalMode = t.Literal["copy", "symlink", "reference"]

_MEDIA = "media"
_MANIFEST = "manifest.parquet"
_DATASET_TOML = "dataset.toml"

_AUDIO_EXT = frozenset(
    {"wav", "flac", "mp3", "ogg", "opus", "m4a", "aac", "aiff", "aif", "wma"},
)
_VIDEO_EXT = frozenset(
    {"mp4", "mkv", "mov", "avi", "webm", "m4v", "mpg", "mpeg", "wmv"},
)


class MediaInfo(t.NamedTuple):
    """Probed metadata for one media file."""

    media_type: str
    duration: float
    sample_rate: int | None
    channels: int | None


class Importer(t.Protocol):
    """Discovers and probes media of one dataset type."""

    media_type: str
    extensions: frozenset[str]

    def probe(self, path: Path) -> MediaInfo: ...  # noqa: D102


class AudioImporter:
    """Imports audio files, probed with soundfile."""

    media_type = "audio"
    extensions = _AUDIO_EXT

    def probe(self, path: Path) -> MediaInfo:
        """Probe an audio file.

        Raises:
            ProbeError: If the file cannot be read.

        """
        try:
            info = soundfile.info(path)
        except soundfile.SoundFileError as err:
            raise ProbeError(path) from err
        return MediaInfo("audio", info.duration, info.samplerate, info.channels)


class VideoImporter:
    """Imports video files, probed with ffprobe (duration only)."""

    media_type = "video"
    extensions = _VIDEO_EXT

    def probe(self, path: Path) -> MediaInfo:
        """Probe a video file's duration via ffprobe.

        Raises:
            ProbeError: If ffprobe is unavailable or fails.

        """
        return MediaInfo("video", _probe_video_duration(path), None, None)


_IMPORTERS: dict[str, Importer] = {
    "audio": AudioImporter(),
    "video": VideoImporter(),
}


def _probe_video_duration(path: Path) -> float:
    """Return a video's duration in seconds via ffprobe.

    Raises:
        ProbeError: If ffprobe is missing or returns no usable duration.

    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(json.loads(result.stdout)["format"]["duration"])
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as err:
        raise ProbeError(path) from err


def _discover(source: Path, extensions: frozenset[str]) -> list[Path]:
    """Matching media files under source, sorted for determinism."""
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix[1:].lower() in extensions)


def _item_id(path: Path, source: Path) -> str:
    """Stable id from the file's path relative to the source root."""
    return path.relative_to(source).with_suffix("").as_posix()


def _place(path: Path, source: Path, media_dir: Path, mode: LocalMode) -> Path:
    """Place one file per the import mode; return its manifest path."""
    if mode == "reference":
        return path.resolve()
    dest = media_dir / path.relative_to(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(path, dest)
    else:
        dest.symlink_to(path.resolve())
    return dest


def _build_rows(
    importer: Importer,
    files: list[Path],
    source: Path,
    media_dir: Path,
    mode: LocalMode,
) -> list[dict[str, t.Any]]:
    """Probe and place each file, returning manifest rows."""
    rows: list[dict[str, t.Any]] = []
    for path in files:
        info = importer.probe(path)
        dest = _place(path, source, media_dir, mode)
        rows.append(
            {
                "item_id": _item_id(path, source),
                "path": str(dest),
                "media_type": info.media_type,
                "duration": info.duration,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
            },
        )
    return rows


def run_import(
    workspace: Workspace,
    source: Path,
    name: str,
    dataset_type: DatasetType,
    mode: LocalMode,
) -> int:
    """Import a local folder as a new dataset; return the item count.

    Raises:
        DatasetExistsError: If the dataset name already exists.
        NoMediaFoundError: If no matching media is found under source.
        ProbeError: If a media file cannot be probed.

    """
    dataset_dir = workspace.dataset_dir(name)
    if dataset_dir.exists():
        raise DatasetExistsError(name)
    importer = _IMPORTERS[dataset_type]
    files = _discover(source, importer.extensions)
    if not files:
        raise NoMediaFoundError(source, dataset_type)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = _build_rows(importer, files, source, dataset_dir / _MEDIA, mode)
    pl.DataFrame(rows, schema=MANIFEST_SCHEMA).write_parquet(dataset_dir / _MANIFEST)
    config = DatasetConfig(type=dataset_type, source=str(source.resolve()), mode=mode)
    DatasetFile.create(dataset_dir / _DATASET_TOML, config).save()
    return len(rows)
