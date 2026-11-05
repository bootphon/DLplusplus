"""CLI commands contributed by the audio_tools module."""

from __future__ import annotations

import typing as t
from pathlib import Path

from clypi import Positional, arg

from audio_pipeline.core.cli_common import ProcessCommand, parse_shard
from audio_pipeline.core.errors import ReplaceMediaError

from . import ConvertParams
from .convert import Convert as ConvertProcess
from .convert import replace_media
from .types import AudioFormat  # noqa: TC001

if t.TYPE_CHECKING:
    from audio_pipeline.core.dataset import Dataset
    from audio_pipeline.core.process import Process
    from audio_pipeline.core.workspace import Workspace


class Convert(ProcessCommand):
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
    replace_media: bool = arg(default=False, help="delete original media/ after a complete convert (copy imports only)")

    def build_process(self, workspace: Workspace, dataset: Dataset) -> Process:
        """Build the Convert process, rejecting --replace-media on non-copy imports.

        Raises:
            ReplaceMediaError: If --replace-media is set on a non-copy import.

        """
        if self.replace_media and dataset.config.mode != "copy":
            raise ReplaceMediaError(self.dataset, dataset.config.mode)
        params = ConvertParams(format=self.format, sample_rate=self.sample_rate, mono=self.mono)
        return ConvertProcess(workspace, dataset, params, parse_shard(self.shard))

    def post_run(self, workspace: Workspace, dataset: Dataset, process: Process) -> None:
        """Delete the originals when --replace-media and the run completed."""
        if self.replace_media and self.shard is None and process.is_complete():
            replace_media(workspace, self.dataset)
