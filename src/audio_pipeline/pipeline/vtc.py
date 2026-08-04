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
import tempfile
import time
from pathlib import Path
from typing import Literal

import polars as pl

from audio_pipeline.compat import patch_torchaudio

patch_torchaudio()

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

    if batch_size <= 0:
        from audio_pipeline.pipeline.resources import (
            query_local_gpu,
            recommend_vtc_batch_size,
        )

        local_gpu = query_local_gpu()
        if local_gpu is not None:
            batch_size = recommend_vtc_batch_size(local_gpu.vram_gb)
            logger.info(
                f"Auto batch_size={batch_size} for {local_gpu.name} ({local_gpu.vram_gb} GB)"
            )
        else:
            batch_size = 128
            logger.info(f"No GPU detected — using default batch_size={batch_size}")

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
    prev_meta_df: pl.DataFrame | None = None

    completed_uids = load_completed_ids(
        meta_dir, id_column="uid", pattern="shard_*.parquet"
    )
    if meta_path.exists():
        prev_meta_df = pl.read_parquet(meta_path)

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
    empty_seg = pl.DataFrame(schema=_SEG_SCHEMA)
    rttm_uids: set[str] = set()
    raw_df: pl.DataFrame = empty_seg
    logger.info(f"Shard {shard_id}: {len(file_ids_to_process)} files")

    with tempfile.TemporaryDirectory(prefix="vtc_") as tmp:
        tmp_wavs = Path(tmp) / "wavs"
        tmp_out = Path(tmp) / "out"
        tmp_wavs.mkdir()

        for uid in file_ids_to_process:
            src = Path(uid_to_path[uid])
            (tmp_wavs / src.name).symlink_to(src)

        run_vtc(
            output=str(tmp_out),
            wavs=str(tmp_wavs),
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

        rttm_files = sorted((tmp_out / "raw_rttm").glob("*.rttm"))
        rttm_uids = {f.stem for f in rttm_files}
        if rttm_files:
            raw_df = pl.concat(
                [_load_rttm(f) for f in rttm_files], how="vertical"
            )

    produced_uids = (
        set(raw_df["uid"].unique().to_list()) if not raw_df.is_empty() else set()
    )
    empty_uids = rttm_uids - produced_uids   # processed, no speech detected
    missing_uids = set(file_ids_to_process) - rttm_uids  # no RTTM created

    meta_rows: list[dict] = (
        [
            vtc_meta_row(uid, thresholds_preset, raw_df.filter(pl.col("uid") == uid))
            for uid in produced_uids
        ]
        + [vtc_meta_row(uid, thresholds_preset, empty_seg) for uid in empty_uids]
        + [vtc_error_row(uid, "no RTTM produced") for uid in missing_uids]
    )

    new_meta_df = pl.DataFrame(meta_rows) if meta_rows else None
    meta_parts: list[pl.DataFrame] = []
    if prev_meta_df is not None:
        if new_meta_df is not None:
            new_uids = set(new_meta_df["uid"].to_list())
            kept = prev_meta_df.filter(~pl.col("uid").is_in(list(new_uids)))
            if not kept.is_empty():
                meta_parts.append(kept)
        else:
            meta_parts.append(prev_meta_df)
    if new_meta_df is not None:
        meta_parts.append(new_meta_df)
    meta_df = pl.concat(meta_parts) if meta_parts else pl.DataFrame()
    if not meta_df.is_empty():
        meta_df = meta_df.unique(subset=["uid"], keep="last")

    prev_seg_path = paths.output / "vtc_raw" / f"shard_{shard_id}.parquet"
    if prev_seg_path.exists() and completed_uids:
        prev_seg_df = pl.read_parquet(prev_seg_path)
        kept = prev_seg_df.filter(~pl.col("uid").is_in(list(rttm_uids)))
        seg_parts = [p for p in [kept, raw_df] if not p.is_empty()]
        seg_df = pl.concat(seg_parts, how="vertical") if seg_parts else empty_seg
    else:
        seg_df = raw_df

    merged_df = merge_segments_df(seg_df, min_duration_off_s, min_duration_on_s)

    meta_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(meta_df, meta_path)

    vtc_raw_dir = paths.output / "vtc_raw"
    vtc_raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(seg_df, vtc_raw_dir / f"shard_{shard_id}.parquet")

    vtc_merged_dir = paths.output / "vtc_merged"
    vtc_merged_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(merged_df, vtc_merged_dir / f"shard_{shard_id}.parquet")

    wall = time.time() - t0
    logger.info("─" * 50)
    logger.info(f"Shard {shard_id} complete")
    logger.info(
        f"  Files    : {len(produced_uids) + len(empty_uids)}/{len(file_ids_to_process)}"
        f"  ({len(missing_uids)} errors)"
    )
    logger.info(f"  Segments : {len(seg_df):,} raw, {len(merged_df):,} merged")
    logger.info(f"  Wall time: {hhmmss(wall)}")
    logger.info("─" * 50)

    total_bytes = sum(
        os.path.getsize(uid_to_path[uid])
        for uid in file_ids_to_process
        if os.path.exists(uid_to_path[uid])
    )
    log_benchmark(
        step="vtc",
        dataset=dataset,
        n_files=len(file_ids_to_process),
        wall_seconds=wall,
        total_bytes=total_bytes,
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
