"""CLI commands contributed by the audio_tools module."""

from __future__ import annotations

import typing as t
from pathlib import Path

from clypi import Positional, arg

from audio_pipeline.core.cli_common import ProcessCommand, parse_shard

from .convert import AudioConvert as AudioConvertProcess
from .types import AudioFormat, MonoAlgorithm

if t.TYPE_CHECKING:
    from audio_pipeline.core.dataset import Dataset
    from audio_pipeline.core.process import Process, ProcessParams
    from audio_pipeline.core.workspace import Workspace


class AudioConvert(ProcessCommand):
    """Convert a dataset's audio (format, sample rate, channels)."""

    # Shared process params — re-declared because clypi does not inherit fields.
    workspace: Positional[Path] = arg(inherited=True)
    dataset: str = arg(help="dataset to convert")
    shard: str | None = arg(default=None, help="process only shard i/n, e.g. 0/4")
    no_progress: bool = arg(default=False, help="disable the progress display (also via STETHO_PROGRESS=0)")

    # Convert-specific params.
    format: AudioFormat = arg(default="wav", help="target audio format")
    sample_rate: int | None = arg(default=None, help="resample to this rate in Hz (default: keep source)")
    mono: bool = arg(default=False, help="downmix to a single channel")
    mono_algorithm: MonoAlgorithm = arg(default="average", help="how to downmix: average, rms, surround, left, right")

    def process_class(self) -> type[Process]:
        """The convert process."""
        return AudioConvertProcess

    def param_overrides(self) -> dict[str, t.Any]:
        """Only the params explicitly passed on the CLI (non-None)."""
        given = {
            "format": self.format,
            "sample_rate": self.sample_rate,
            "mono": self.mono,
            "mono_algorithm": self.mono_algorithm,
        }
        return {key: value for key, value in given.items() if value is not None}

    def build_process(self, workspace: Workspace, dataset: Dataset, params: ProcessParams) -> Process:
        """Build the convert process from the resolved params."""
        return AudioConvertProcess(workspace, dataset, params, parse_shard(self.shard))
