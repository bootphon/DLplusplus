"""The `convert` process: transcode dataset audio via ffmpeg into the audio layer.

Reads the dataset's raw audio (manifest/`source/`) and writes a target-format copy
under `audio/<format>/<name>/`, recording the active layer in dataset.toml.
Native process: needs the `ffmpeg` binary, no optional extra.
"""

from __future__ import annotations

import subprocess
import typing as t
from pathlib import Path

from audio_pipeline.core.config import ConvertedLayer, DatasetFile
from audio_pipeline.core.dataset import AudioFile
from audio_pipeline.core.errors import ConversionError, MissingBinaryError
from audio_pipeline.core.process import ItemProcess, ProcessParams, register

if t.TYPE_CHECKING:
    from collections.abc import Iterator

    from .types import AudioFormat


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
        base = self.workspace.audio_dir(self.dataset.name, self._p.format)
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
        toml = self.workspace.dataset_toml(self.dataset.name)
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
