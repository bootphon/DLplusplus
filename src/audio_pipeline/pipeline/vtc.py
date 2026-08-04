#!/usr/bin/env python3
"""
VTC inference via vtc_inference.run_vtc.

Paths are derived from the dataset name:
    manifests/{dataset}.parquet        input manifest
    output/{dataset}/vtc_raw/          raw VTC segments   (parquet shards)
    output/{dataset}/vtc_merged/       merged VTC segments (parquet shards)
    output/{dataset}/vtc_meta/         per-file metadata   (parquet shards)

Usage:
    python -m audio_pipeline.pipeline.vtc chunks30

SLURM array:
    python -m audio_pipeline.pipeline.vtc chunks30 \\
        --array_id $SLURM_ARRAY_TASK_ID \\
        --array_count $SLURM_ARRAY_TASK_COUNT
"""

import argparse
import logging
import os
import sys
import shutil
import time
from pathlib import Path
from typing import Literal

import polars as pl

from audio_pipeline.compat import patch_torchaudio

patch_torchaudio()

from torchcodec.decoders import AudioDecoder as _AudioDecoder  # noqa: E402
from vtc_inference import run_vtc  # noqa: E402

from audio_pipeline.core.metadata import vtc_error_row, vtc_meta_row  # noqa: E402
from audio_pipeline.utils import (  # noqa: E402
    add_sample_argument,
    atomic_write_parquet,
    get_dataset_paths,
    hhmmss,
    load_completed_ids,
    load_manifest,
    log_benchmark,
    merge_segments_df,
    sample_manifest,
    set_seeds,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vtc")

MODEL_ROOT = Path(
    os.environ["MODEL_ROOT"]
    if "MODEL_ROOT" in os.environ
    else Path.home() / ".cache/dlplusplus"
)

_SEG_SCHEMA = {
    "uid": pl.String,
    "onset": pl.Float64,
    "offset": pl.Float64,
    "duration": pl.Float64,
    "label": pl.String,
}

_META_COLS = ["uid", "vtc_threshold_preset", "vtc_speech_dur", "vtc_n_segments", "vtc_label_counts", "error"]


def _load_rttm(path: Path) -> pl.DataFrame:
    """Read one RTTM file into uid/onset/offset/duration/label schema."""
    try:
        df = pl.read_csv(
            path,
            has_header=False,
            separator=" ",
            columns=[1, 3, 4, 7],
            new_columns=["uid", "onset", "duration", "label"],
        )
    except pl.exceptions.NoDataError:
        return pl.DataFrame(schema=_SEG_SCHEMA)
    return df.with_columns(
        (pl.col("onset") + pl.col("duration")).alias("offset")
    ).select(list(_SEG_SCHEMA.keys()))



def _resolve_batch_size(batch_size: int) -> int:
    if batch_size > 0:
        return batch_size
    from audio_pipeline.pipeline.resources import query_local_gpu, recommend_vtc_batch_size
    local_gpu = query_local_gpu()
    if local_gpu is not None:
        size = recommend_vtc_batch_size(local_gpu.vram_gb)
        logger.info(f"Auto batch_size={size} for {local_gpu.name} ({local_gpu.vram_gb} GB)")
        return size
    logger.info("No GPU detected — using default batch_size=128")
    return 128


def _group_by_parent(uid_to_path: dict[str, str], uids: list[str]) -> dict[Path, list[str]]:
    """Group UIDs by the resolved parent directory of their audio file."""
    groups: dict[Path, list[str]] = {}
    for uid in uids:
        parent = Path(uid_to_path[uid]).resolve().parent
        groups.setdefault(parent, []).append(uid)
    return groups



def _check_audio(uid_to_path: dict[str, str], file_ids: list[str]) -> None:
    """Log audio properties and warn on known inference-breaking conditions.

    Raises whatever torchcodec raises if a file cannot be opened — this is
    intentional: segma swallows per-file errors with no traceback, so a failure
    here gives the only visible traceback we will ever get.
    """
    for uid in file_ids:
        path = Path(uid_to_path[uid]).resolve()
        meta = _AudioDecoder(path).metadata
        dur = meta.duration_seconds_from_header
        sr = meta.sample_rate  # pyright: ignore[reportAttributeAccessIssue]
        ch = meta.num_channels  # pyright: ignore[reportAttributeAccessIssue]
        if dur is None:
            logger.warning(f"{uid}: duration_seconds_from_header=None — will fail in segma")
        if sr != 16_000:
            logger.warning(f"{uid}: sample_rate={sr} — model expects 16 kHz")
        if ch and ch > 1:
            logger.warning(f"{uid}: channels={ch} — model expects mono")
        logger.debug(f"{uid}: sr={sr} ch={ch} dur={dur:.1f}s" if dur is not None else f"{uid}: sr={sr} ch={ch} dur=None")


def _run_shard_inference(
    file_ids: list[str],
    uid_to_path: dict[str, str],
    work_dir: Path,
    *,
    model_root: Path,
    thresholds_preset: str,
    batch_size: int,
    device: str,
    stride_pct: float,
) -> tuple[pl.DataFrame, set[str]]:
    """Run VTC on the given file IDs, return (raw_segments_df, rttm_uids).

    Groups files by parent directory and calls run_vtc once per group,
    writing a URIs file into work_dir. Audio files are never copied or linked.
    """
    _check_audio(uid_to_path, file_ids)
    out_dir = work_dir / "out"
    for i, (wav_dir, uids) in enumerate(_group_by_parent(uid_to_path, file_ids).items()):
        uris_file = work_dir / f"uris_{i}.txt"
        uris_file.write_text("\n".join(uids))
        run_vtc(
            output=str(out_dir),
            wavs=str(wav_dir),
            uris=uris_file,
            config=model_root,
            checkpoint=model_root,
            thresholds=thresholds_preset,
            thresholds_location=model_root,
            batch_size=batch_size,
            device=device,
            stride_pct=stride_pct,
            keep_raw=True,
            write_csv=False,
        )
    rttm_files = sorted((out_dir / "raw_rttm").glob("*.rttm"))
    rttm_uids = {f.stem for f in rttm_files}
    raw_df = (
        pl.concat([_load_rttm(f) for f in rttm_files], how="vertical")
        if rttm_files
        else pl.DataFrame(schema=_SEG_SCHEMA)
    )
    return raw_df, rttm_uids


def _build_meta_df(
    raw_df: pl.DataFrame,
    rttm_uids: set[str],
    file_ids: list[str],
    thresholds_preset: str,
    prev_meta_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Build the per-shard metadata DataFrame, merged with any previous run."""
    empty_seg = pl.DataFrame(schema=_SEG_SCHEMA)
    produced_uids = set(raw_df["uid"].unique().to_list()) if not raw_df.is_empty() else set()
    empty_uids = rttm_uids - produced_uids
    missing_uids = set(file_ids) - rttm_uids

    rows = (
        [vtc_meta_row(uid, thresholds_preset, raw_df.filter(pl.col("uid") == uid)) for uid in produced_uids]
        + [vtc_meta_row(uid, thresholds_preset, empty_seg) for uid in empty_uids]
        + [vtc_error_row(uid, "no RTTM produced") for uid in missing_uids]
    )
    new_df = pl.DataFrame(rows) if rows else None

    parts: list[pl.DataFrame] = []
    if prev_meta_df is not None:
        kept = (
            prev_meta_df.filter(~pl.col("uid").is_in(new_df["uid"].to_list()))
            if new_df is not None
            else prev_meta_df
        )
        if not kept.is_empty():
            parts.append(kept)
    if new_df is not None:
        parts.append(new_df)

    result = pl.concat(parts) if parts else pl.DataFrame()
    return result.unique(subset=["uid"], keep="last") if not result.is_empty() else result

def main(
    dataset: str,
    model_root: Path = MODEL_ROOT / "vtc",
    thresholds_preset: str = "f1",
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.3,
    batch_size: int = 0,
    stride_pct: float = 0.25,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
    sample: int | float | None = None,
) -> None:
    set_seeds(42)
    batch_size = _resolve_batch_size(batch_size)

    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  manifest   : {paths.manifest}")
    logger.info(f"  output     : {paths.output}")
    logger.info(f"  model_root : {model_root}")
    logger.info(f"  thresholds : {thresholds_preset}")
    logger.info(f"  batch_size : {batch_size}")
    logger.info(f"  stride_pct : {stride_pct}")

    manifest_df = load_manifest(paths.manifest)
    manifest_df = sample_manifest(manifest_df, sample)
    if sample is not None:
        logger.info(f"  sample     : {len(manifest_df)} files")

    resolved_paths = manifest_df["path"].drop_nulls().to_list()
    file_ids = [Path(p).stem for p in resolved_paths]
    uid_to_path: dict[str, str] = dict(zip(file_ids, resolved_paths))

    if array_id is not None and array_count is not None:
        file_ids = shard_list(file_ids, array_id, array_count)
        logger.info(f"Shard {array_id}/{array_count - 1}: {len(file_ids)} files")

    shard_id = array_id if array_id is not None else 0
    meta_dir = paths.output / "vtc_meta"
    meta_path = meta_dir / f"shard_{shard_id}.parquet"

    completed_uids = load_completed_ids(meta_dir, id_column="uid", pattern="shard_*.parquet")
    if meta_path.exists():
        _pm = pl.read_parquet(meta_path)
        if "uid" in _pm.columns:
            prev_meta_df = _pm.select([c for c in _META_COLS if c in _pm.columns])
        else:
            prev_meta_df = None
    else:
        prev_meta_df = None

    if completed_uids and meta_dir.is_dir():
        error_uids: set[str] = set()
        for f in sorted(meta_dir.glob("shard_*.parquet")):
            df = pl.read_parquet(f)
            if "uid" in df.columns and "error" in df.columns:
                error_uids.update(df.filter(pl.col("error") != "")["uid"].to_list())
        completed_uids -= error_uids

    file_ids_to_process = [uid for uid in file_ids if uid not in completed_uids]
    if len(file_ids_to_process) < len(file_ids):
        logger.info(
            f"Resume: {len(file_ids) - len(file_ids_to_process)} done, "
            f"{len(file_ids_to_process)} remaining"
        )
    if not file_ids_to_process:
        logger.info("No files to process.")
        return

    t0 = time.time()
    logger.info(f"Shard {shard_id}: {len(file_ids_to_process)} files")

    work_dir = paths.output / "vtc_work" / f"shard_{shard_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_df, rttm_uids = _run_shard_inference(
            file_ids_to_process, uid_to_path, work_dir,
            model_root=model_root, thresholds_preset=thresholds_preset,
            batch_size=batch_size, device=device, stride_pct=stride_pct,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    meta_df = _build_meta_df(raw_df, rttm_uids, file_ids_to_process, thresholds_preset, prev_meta_df)

    empty_seg = pl.DataFrame(schema=_SEG_SCHEMA)
    prev_seg_path = paths.output / "vtc_raw" / f"shard_{shard_id}.parquet"
    if prev_seg_path.exists() and completed_uids:
        prev_seg_df = pl.read_parquet(prev_seg_path)
        if all(c in prev_seg_df.columns for c in _SEG_SCHEMA):
            kept = prev_seg_df.filter(~pl.col("uid").is_in(list(rttm_uids)))
            parts = [p for p in [kept, raw_df] if not p.is_empty()]
            seg_df = pl.concat(parts, how="vertical") if parts else empty_seg
        else:
            seg_df = raw_df
    else:
        seg_df = raw_df

    merged_df = merge_segments_df(seg_df, min_duration_off_s, min_duration_on_s)

    meta_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(meta_df, meta_path)
    (paths.output / "vtc_raw").mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(seg_df, paths.output / "vtc_raw" / f"shard_{shard_id}.parquet")
    (paths.output / "vtc_merged").mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(merged_df, paths.output / "vtc_merged" / f"shard_{shard_id}.parquet")

    wall = time.time() - t0
    missing_uids = set(file_ids_to_process) - rttm_uids
    logger.info("─" * 50)
    logger.info(f"Shard {shard_id} complete")
    logger.info(f"  Files    : {len(rttm_uids)}/{len(file_ids_to_process)}  ({len(missing_uids)} errors)")
    logger.info(f"  Segments : {len(seg_df):,} raw, {len(merged_df):,} merged")
    logger.info(f"  Wall time: {hhmmss(wall)}")
    logger.info("─" * 50)

    log_benchmark(
        step="vtc",
        dataset=dataset,
        n_files=len(file_ids_to_process),
        wall_seconds=wall,
        total_bytes=sum(
            os.path.getsize(uid_to_path[uid])
            for uid in file_ids_to_process
            if os.path.exists(uid_to_path[uid])
        ),
        n_workers=1,
        extra={"device": device, "shard_id": shard_id},
    )


def entrypoint() -> None:
    parser = argparse.ArgumentParser(
        description="VTC inference via vtc_inference.run_vtc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m audio_pipeline.pipeline.vtc chunks30\n"
            "  python -m audio_pipeline.pipeline.vtc chunks30 --sample 500\n"
        ),
    )
    parser.add_argument("dataset", help="Dataset name.")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT / "vtc",
        help=(
            "Directory containing model/best.ckpt, model/config.toml, "
            f"and thresholds/. (default: {MODEL_ROOT / 'vtc'})"
        ),
    )
    parser.add_argument(
        "--thresholds-preset",
        default="f1",
        choices=["f1", "hp"],
        help="Threshold preset to use (default: f1)",
    )
    parser.add_argument(
        "--min-duration-on-s",
        type=float,
        default=0.1,
        help="Remove segments shorter than this (default: 0.1s)",
    )
    parser.add_argument(
        "--min-duration-off-s",
        type=float,
        default=0.3,
        help="Merge same-label gaps smaller than this (default: 0.3s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size. 0 = auto-detect from GPU VRAM (default: 0).",
    )
    parser.add_argument(
        "--stride-pct",
        type=float,
        default=0.25,
        help=(
            "Sliding window stride as fraction of chunk duration "
            "(default: 0.25 = 75%% overlap)"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument("--array-id", type=int, help="SLURM array task ID")
    parser.add_argument("--array-count", type=int, help="Total SLURM array tasks")
    add_sample_argument(parser)

    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    entrypoint()
