"""The `convert` process: transcode dataset audio via ffmpeg into the audio layer.

Reads the dataset's raw audio (manifest/`source/`) and writes a target-format copy
under `audio/<format>/<name>/`, recording the active layer in dataset.toml.
Native process: needs the `ffmpeg` binary, no optional extra.
"""

from __future__ import annotations

import logging
import math
import subprocess
import typing as t
from pathlib import Path

from audio_pipeline.core.config import ConvertedLayer, DatasetFile
from audio_pipeline.core.errors import ConversionError, MissingBinaryError, ProbeError
from audio_pipeline.core.process import ItemProcess, ProcessParams, register

from .types import AudioFormat, MonoAlgorithm

if t.TYPE_CHECKING:
    from collections.abc import Iterator


_LOG = logging.getLogger(__name__)


class AudioConvertParams(ProcessParams):
    """Parameters for the `convert` process."""

    format: AudioFormat = "wav"
    sample_rate: int | None = 16_000
    mono: bool = True
    mono_algorithm: MonoAlgorithm = "average"


class _ConvertItem(t.NamedTuple):
    """One audio item to convert: id, source path, and source channel count."""

    item_id: str
    path: Path
    channels: int | None


def _has_audio_stream(path: Path) -> bool:
    """Whether `path` has at least one audio stream, via ffprobe.

    Raises:
        MissingBinaryError: If the ffprobe binary is not on PATH.
        ProbeError: If ffprobe fails to read the file.

    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as err:
        raise MissingBinaryError("ffprobe") from err
    except subprocess.CalledProcessError as err:
        raise ProbeError(path, err.stderr.strip()) from err
    return bool(result.stdout.strip())


@register
class AudioConvert(ItemProcess[_ConvertItem]):
    """Transcode each raw audio file into the dataset's converted layer."""

    name: t.ClassVar[str] = "convert"
    required_binaries: t.ClassVar[tuple[str, ...]] = ("ffmpeg", "ffprobe")
    Params: t.ClassVar[type[ProcessParams]] = AudioConvertParams

    @property
    def _p(self) -> AudioConvertParams:
        return t.cast("AudioConvertParams", self.params)

    def _dest(self, item: _ConvertItem) -> Path:
        base = self.workspace.audio_dir(self.dataset.name, self._p.format)
        return base / f"{item.item_id}.{self._p.format}"

    def items(self) -> Iterator[_ConvertItem]:
        """Yield each raw audio file from the manifest in a deterministic order."""
        for row in self.dataset.read_manifest().sort("item_id").iter_rows(named=True):
            yield _ConvertItem(item_id=row["item_id"], path=Path(row["path"]), channels=row["channels"])

    def _marker(self, item: _ConvertItem) -> Path:
        """Sentinel marking an item skipped because its source has no audio."""
        return self._dest(item).with_suffix(".noaudio")

    def is_done(self, item: _ConvertItem) -> bool:
        """Whether this item is converted, or was skipped (no audio stream)."""
        return self._dest(item).exists() or self._marker(item).exists()

    def process_item(self, item: _ConvertItem) -> None:
        """Transcode one file, or skip it (with a marker) when it has no audio."""
        dest = self._dest(item)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not _has_audio_stream(item.path):
            self._marker(item).touch()
            _LOG.warning("no audio stream in %s; skipping", item.item_id)
            return
        _run_ffmpeg(item.path, dest, self._p, item.channels)

    def on_complete(self) -> None:
        """Record the converted layer in dataset.toml once the run completes."""
        toml = self.workspace.dataset_toml(self.dataset.name)
        config_file = DatasetFile.load(toml)
        config_file.set_converted(
            ConvertedLayer(format=self._p.format, sample_rate=self._p.sample_rate, mono=self._p.mono),
        )
        config_file.save()


def _mono_args(algorithm: MonoAlgorithm, channels: int | None) -> list[str]:
    """Arguments for ffmpeg to downmix to mono per `algorithm`.

    Uses a layout-agnostic `pan` expression for `average`/`rms`; falls back to
    ffmpeg's layout-aware downmix (`-ac 1`) for `surround`, an already-mono source,
    or an unknown channel count.
    """
    if channels is not None and channels <= 1:
        return ["-ac", "1"]
    if algorithm == "surround" or channels is None:
        return ["-ac", "1"]
    if algorithm == "left":
        return ["-af", "pan=mono|c0=c0"]
    if algorithm == "right":
        return ["-af", "pan=mono|c0=c1"]
    gain = 1 / channels if algorithm == "average" else 1 / math.sqrt(channels)
    terms = "+".join(f"{gain:.6f}*c{i}" for i in range(channels))
    return ["-af", f"pan=mono|c0={terms}"]


def _run_ffmpeg(src: Path, dest: Path, params: AudioConvertParams, channels: int | None) -> None:
    """Transcode `src` to `dest` with ffmpeg per `params`.

    Raises:
        MissingBinaryError: If the ffmpeg binary is not on PATH.
        ConversionError: If ffmpeg exits non-zero.

    """
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if params.sample_rate is not None:
        cmd += ["-ar", str(params.sample_rate)]
    if params.mono:
        cmd += _mono_args(params.mono_algorithm, channels)
    cmd.append(str(dest))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as err:
        raise MissingBinaryError("ffmpeg") from err
    except subprocess.CalledProcessError as err:
        raise ConversionError(src, err.stderr) from err
