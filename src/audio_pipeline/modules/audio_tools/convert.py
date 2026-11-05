"""The `convert` process: transcode dataset audio via ffmpeg into `converted/`.

Reads the dataset's raw audio (manifest/`media/`) and writes a target-format copy
under `datasets/<name>/converted/`, recording the active layer in dataset.toml.
Native process: needs the `ffmpeg` binary, no optional extra.
"""

from __future__ import annotations

import shutil
import subprocess
import typing as t
from pathlib import Path

from audio_pipeline.core.config import ConvertedLayer, DatasetFile
from audio_pipeline.core.dataset import AudioFile
from audio_pipeline.core.errors import ConversionError, MissingBinaryError
from audio_pipeline.core.process import ItemProcess, ProcessParams, register

if t.TYPE_CHECKING:
    from collections.abc import Iterator

    from audio_pipeline.core.workspace import Workspace

    from .types import AudioFormat

_CONVERTED = "converted"
_MEDIA = "media"
_DATASET_TOML = "dataset.toml"


class ConvertParams(ProcessParams):
    """Parameters for the `convert` process."""

    format: AudioFormat = "wav"
    sample_rate: int | None = None
    mono: bool = False


@register
class Convert(ItemProcess[AudioFile]):
    """Transcode each raw audio file into the dataset's converted layer."""

    name: t.ClassVar[str] = "convert"
    Params: t.ClassVar[type[ProcessParams]] = ConvertParams

    @property
    def _p(self) -> ConvertParams:
        return t.cast("ConvertParams", self.params)

    def _dest(self, item: AudioFile) -> Path:
        base = self.workspace.dataset_dir(self.dataset.name) / _CONVERTED
        return base / f"{item.item_id}.{self._p.format}"

    def items(self) -> Iterator[AudioFile]:
        """Yield each raw audio file from the manifest in a deterministic order."""
        for row in self.dataset.read_manifest().sort("item_id").iter_rows(named=True):
            yield AudioFile(item_id=row["item_id"], path=Path(row["path"]))

    def is_done(self, item: AudioFile) -> bool:
        """Whether this item's converted file already exists."""
        return self._dest(item).exists()

    def process_item(self, item: AudioFile) -> None:
        """Transcode one file into the converted layer."""
        dest = self._dest(item)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(item.path, dest, self._p)

    def on_complete(self) -> None:
        """Record the converted layer in dataset.toml once the run completes."""
        toml = self.workspace.dataset_dir(self.dataset.name) / _DATASET_TOML
        config_file = DatasetFile.load(toml)
        config_file.set_converted(
            ConvertedLayer(format=self._p.format, sample_rate=self._p.sample_rate, mono=self._p.mono),
        )
        config_file.save()


def _run_ffmpeg(src: Path, dest: Path, params: ConvertParams) -> None:
    """Transcode `src` to `dest` with ffmpeg per `params`.

    Raises:
        MissingBinaryError: If the ffmpeg binary is not on PATH.
        ConversionError: If ffmpeg exits non-zero.

    """
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if params.sample_rate is not None:
        cmd += ["-ar", str(params.sample_rate)]
    if params.mono:
        cmd += ["-ac", "1"]
    cmd.append(str(dest))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as err:
        raise MissingBinaryError("ffmpeg") from err
    except subprocess.CalledProcessError as err:
        raise ConversionError(src, err.stderr) from err


def replace_media(workspace: Workspace, dataset_name: str) -> None:
    """Delete the original media/ directory after a complete convert."""
    media = workspace.dataset_dir(dataset_name) / _MEDIA
    if media.exists():
        shutil.rmtree(media)
