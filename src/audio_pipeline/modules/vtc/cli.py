"""CLI command contributed by the vtc module."""

from __future__ import annotations

import typing as t
from pathlib import Path

from clypi import Positional, arg

from audio_pipeline.core.cli_common import ProcessCommand

from .infer import Vtc as VtcProcess
from .types import Device

if t.TYPE_CHECKING:
    from audio_pipeline.core.dataset import Dataset
    from audio_pipeline.core.process import Process, ProcessParams
    from audio_pipeline.core.workspace import Workspace


class Vtc(ProcessCommand):
    """Run VTC speaker-type inference (raw + merged RTTM in the run folder)."""

    # Shared process params — re-declared because clypi does not inherit fields.
    # (No --shard: vtc processes the whole wav set in one batched run.)
    workspace: Positional[Path] = arg(inherited=True)
    dataset: str = arg(help="dataset to process")
    no_progress: bool = arg(default=False, help="disable the progress display (also via STETHO_PROGRESS=0)")

    # vtc params — None = not given on the CLI (fall back to meta.toml, then the model default).
    config: str | None = arg(default=None, help="model config version/path (default 2.1)")
    checkpoint: str | None = arg(default=None, help="model checkpoint version/path (default 2.1)")
    thresholds: str | None = arg(default=None, help="thresholds name (default f1)")
    thresholds_location: str | None = arg(default=None, help="thresholds dir/version (default: alongside checkpoint)")
    high_precision: bool | None = arg(default=None, help="use high-precision thresholds")
    save_probs: bool | None = arg(default=None, help="also save frame probabilities")
    min_duration_on_s: float | None = arg(default=None, help="fill same-speaker gaps shorter than this (s)")
    min_duration_off_s: float | None = arg(default=None, help="drop speech segments shorter than this (s)")
    batch_size: int | None = arg(default=None, help="inference batch size")
    stride_pct: float | None = arg(default=None, help="sliding-window stride as a fraction of chunk duration")
    write_empty: bool | None = arg(default=None, help="write empty RTTMs for files with no speech")
    write_csv: bool | None = arg(default=None, help="also write concatenated CSVs")
    recursive_search: bool | None = arg(default=None, help="search wavs recursively")
    device: Device | None = arg(default=None, help="compute device: cpu, cuda, gpu, mps")
    keep_raw: bool | None = arg(default=None, help="keep raw (pre-merge) RTTMs")

    def process_class(self) -> type[Process]:
        """The vtc process."""
        return VtcProcess

    def param_overrides(self) -> dict[str, t.Any]:
        """Only the params explicitly passed on the CLI (non-None)."""
        given = {
            "config": self.config,
            "checkpoint": self.checkpoint,
            "thresholds": self.thresholds,
            "thresholds_location": self.thresholds_location,
            "high_precision": self.high_precision,
            "save_probs": self.save_probs,
            "min_duration_on_s": self.min_duration_on_s,
            "min_duration_off_s": self.min_duration_off_s,
            "batch_size": self.batch_size,
            "stride_pct": self.stride_pct,
            "write_empty": self.write_empty,
            "write_csv": self.write_csv,
            "recursive_search": self.recursive_search,
            "device": self.device,
            "keep_raw": self.keep_raw,
        }
        return {key: value for key, value in given.items() if value is not None}

    def build_process(self, workspace: Workspace, dataset: Dataset, params: ProcessParams) -> Process:
        """Build the vtc process from the resolved params."""
        return VtcProcess(workspace, dataset, params)
