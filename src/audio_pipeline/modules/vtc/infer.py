"""The `vtc` process: run VTC inference via `vtc_inference.run_vtc`.

Runs the full `run_vtc` pipeline (inference + segment merge, optional CSV) over the
dataset's wav audio into `runs/vtc/<dataset>/<run_id>/`:

- `rttm/<stem>.rttm`           merged per-file segments
- `raw_rttm/<stem>.rttm`       raw per-file segments (kept when `keep_raw`)
- `rttm.csv` / `raw_rttm.csv`  concatenated CSVs (when `write_csv`)

Every `run_vtc` parameter is exposed via `VtcParams`. VTC models expect 16 kHz mono
wav (convert's defaults). Needs the optional `vtc` extra; the model is fetched from
the HF Hub by version on first use.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import typing as t
from pathlib import Path

from audio_pipeline.core.config import DatasetFile
from audio_pipeline.core.errors import MissingExtraError
from audio_pipeline.core.process import Process, ProcessParams, register

from .types import Device

if t.TYPE_CHECKING:
    from audio_pipeline.core.dataset import AudioFile

_MERGED_RTTM = "rttm"
# Knobs that do not change the RTTM content; excluded from the run_id.
_RUN_ID_EXCLUDE = frozenset({"device", "batch_size", "write_csv", "keep_raw", "recursive_search"})


class VtcParams(ProcessParams):
    """Parameters for the `vtc` process — the full `run_vtc` surface."""

    config: str = "2.1"
    checkpoint: str = "2.1"
    thresholds: str = "f1"
    thresholds_location: str | None = None
    high_precision: bool = False
    save_probs: bool = False
    min_duration_on_s: float = 0.1
    min_duration_off_s: float = 0.1
    batch_size: int = 128
    stride_pct: float = 0.25
    write_empty: bool = True
    write_csv: bool = True
    recursive_search: bool = True
    device: Device = "cuda"
    keep_raw: bool = True


def _run_vtc(wav_root: Path, out: Path, params: VtcParams) -> None:
    """Run the `run_vtc` pipeline into `out`.

    Raises:
        MissingExtraError: If the `vtc` extra (vtc_inference) is not installed.

    """
    try:
        from vtc_inference import run_vtc  # noqa: PLC0415
    except ImportError as err:
        raise MissingExtraError("vtc") from err

    run_vtc(
        output=str(out),
        uris=None,
        config=params.config,
        wavs=str(wav_root),
        checkpoint=params.checkpoint,
        save_probs=params.save_probs,
        high_precision=params.high_precision,
        thresholds=params.thresholds,
        thresholds_location=params.thresholds_location,
        min_duration_on_s=params.min_duration_on_s,
        min_duration_off_s=params.min_duration_off_s,
        batch_size=params.batch_size,
        stride_pct=params.stride_pct,
        write_empty=params.write_empty,
        write_csv=params.write_csv,
        recursive_search=params.recursive_search,
        device=params.device,
        keep_raw=params.keep_raw,
    )


@register
class Vtc(Process):
    """Run VTC inference over a dataset's wav audio into the run folder."""

    name: t.ClassVar[str] = "vtc"
    Params: t.ClassVar[type[ProcessParams]] = VtcParams

    @property
    def _p(self) -> VtcParams:
        return t.cast("VtcParams", self.params)

    @functools.cached_property
    def _run_id(self) -> str:
        """Short hash of the output-determining params + installed vtc version.

        Raises:
            MissingExtraError: If the `vtc` extra is not installed.

        """
        try:
            version = importlib.metadata.version("vtc")
        except importlib.metadata.PackageNotFoundError as err:
            raise MissingExtraError("vtc") from err
        fields = self._p.model_dump(mode="json", exclude=set(_RUN_ID_EXCLUDE))
        key = json.dumps({"vtc": version, "params": fields}, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def _out_dir(self) -> Path:
        return self.workspace.run_dir(self.name, self.dataset.name, self._run_id)

    def _wav_files(self) -> list[AudioFile]:
        """The dataset's wav audio (raises NeedsConversionError if not wav)."""
        return list(self.dataset.audio_files(require="wav"))

    def _wav_root(self) -> Path:
        """Root of the active wav layer (converted layer when present, else source)."""
        # TODO: find a better way to do this
        # NOTE: vtc_infer should work on a list of files ? probably
        converted = self.dataset.config.converted
        if converted is not None and converted.format == "wav":
            return self.workspace.audio_dir(self.dataset.name, "wav")
        return self.workspace.source_dir(self.dataset.name)

    def is_complete(self) -> bool:
        """Whether every wav has a merged RTTM in the run folder."""
        merged = self._out_dir / _MERGED_RTTM
        if not merged.is_dir():
            return False
        return all((merged / f"{item.path.stem}.rttm").exists() for item in self._wav_files())

    def run(self) -> None:
        """Run VTC over all wavs, then record the kept run if complete.

        Raises:
            NeedsConversionError: If the dataset's active audio layer is not wav.
            MissingExtraError: If the `vtc` extra is not installed.

        """
        if not self._wav_files():  # validates wav up front
            return
        _run_vtc(self._wav_root(), self._out_dir, self._p)
        if self.is_complete():
            self.on_complete()

    def on_complete(self) -> None:
        """Record this run as the kept vtc run in dataset.toml."""
        config_file = DatasetFile.load(self.workspace.dataset_toml(self.dataset.name))
        config_file.set_kept_run(self.name, self._run_id)
        config_file.save()
