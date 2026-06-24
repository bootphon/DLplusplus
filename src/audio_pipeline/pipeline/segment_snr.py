"""Per-VTC-segment SNR & C50 from pre-computed Brouhaha frames.

Reads raw per-frame arrays written by ``pipeline/snr.py``
(``snr/{uid}.npz``: fields ``snr``, ``c50``, ``step_s``) and averages the
frames that fall within each VTC segment's ``[onset, offset]``.

No GPU or model required.  Must run after both VTC and SNR complete.

Output:
    output/{dataset}/segment_snr/shard_N.parquet
        columns: uid, onset, offset, label, snr_mean, c50_mean

Usage:
    python -m audio_pipeline.pipeline.segment_snr seedlings_10
    sbatch slurm/segment_snr.slurm seedlings_10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from audio_pipeline.utils import (
    add_sample_argument,
    get_dataset_paths,
    hhmmss,
    load_manifest,
    log_benchmark,
    sample_manifest,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("segment_snr")


def _load_vtc_segments(output_dir: Path) -> pl.DataFrame:
    """Load all VTC segments from vtc_merged/*.parquet."""
    vtc_dir = output_dir / "vtc_merged"
    if not vtc_dir.exists():
        raise FileNotFoundError(f"VTC segments not found: {vtc_dir}")
    files = sorted(vtc_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {vtc_dir}")
    return pl.concat([pl.read_parquet(f) for f in files])


def _segment_means(
    raw_snr: np.ndarray,
    raw_c50: np.ndarray,
    step_s: float,
    segments: list[dict],
) -> list[dict]:
    """Compute mean SNR and C50 for each VTC segment from raw frames."""
    results = []
    for seg in segments:
        onset = seg["onset"]
        offset = seg["offset"]
        i0 = max(0, int(onset / step_s))
        i1 = min(len(raw_snr), int(np.ceil(offset / step_s)))
        if i0 >= i1:
            snr_mean = None
            c50_mean = None
        else:
            snr_mean = round(float(np.mean(raw_snr[i0:i1])), 2)
            c50_mean = round(float(np.mean(raw_c50[i0:i1])), 2)
        results.append(
            {
                "uid": seg["uid"],
                "onset": seg["onset"],
                "offset": seg["offset"],
                "label": seg["label"],
                "snr_mean": snr_mean,
                "c50_mean": c50_mean,
            }
        )
    return results


def main(
    dataset: str,
    array_id: int | None = None,
    array_count: int | None = None,
    sample: int | float | None = None,
) -> None:
    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  output    : {paths.output}")

    snr_dir = paths.output / "snr"
    if not snr_dir.exists():
        raise FileNotFoundError(
            f"SNR frames not found: {snr_dir}\nRun pipeline/snr.py first."
        )

    all_vtc = _load_vtc_segments(paths.output)
    logger.info(f"  VTC segments: {len(all_vtc):,} total")

    manifest_df = load_manifest(paths.manifest)
    manifest_df = sample_manifest(manifest_df, sample)

    file_ids = [Path(p).stem for p in manifest_df["path"].drop_nulls().to_list()]

    if array_id is not None and array_count is not None:
        file_ids = shard_list(file_ids, array_id, array_count)
        logger.info(f"Shard {array_id}/{array_count - 1}: {len(file_ids)} files")

    shard_id = array_id if array_id is not None else 0

    all_rows: list[dict] = []
    n_errors = 0
    n_missing = 0
    total = len(file_ids)
    t0 = time.time()
    log_every = max(1, total // 20)

    for i, uid in enumerate(file_ids, 1):
        npz_path = snr_dir / f"{uid}.npz"
        if not npz_path.exists():
            n_missing += 1
            logger.warning(f"{uid}: SNR npz not found, skipping")
            continue
        try:
            npz = np.load(npz_path)
            raw_snr = npz["snr"].astype(np.float32)
            raw_c50 = npz["c50"].astype(np.float32)
            step_s = float(npz["step_s"])

            seg_dicts = all_vtc.filter(pl.col("uid") == uid).to_dicts()
            all_rows.extend(_segment_means(raw_snr, raw_c50, step_s, seg_dicts))

            if i % log_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                eta = (
                    f"{remaining / 60:.0f}m"
                    if remaining < 3600
                    else f"{remaining / 3600:.1f}h"
                )
                print(
                    f"  {i:>4}/{total}  segments={len(all_rows):,}  ETA {eta}",
                    flush=True,
                )

        except Exception as e:
            n_errors += 1
            logger.warning(f"{uid}: {e}")

    out_dir = paths.output / "segment_snr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_id}.parquet"

    if all_rows:
        df = pl.DataFrame(all_rows)
        df.write_parquet(out_path)
        logger.info(f"Saved {len(df):,} segment rows → {out_path}")
        snr_vals = df["snr_mean"].drop_nulls()
        c50_vals = df["c50_mean"].drop_nulls()
        if len(snr_vals) > 0:
            logger.info(
                f"  SNR : mean={snr_vals.mean():.1f} dB  "
                f"std={snr_vals.std():.1f} dB  "
                f"range=[{snr_vals.min():.1f}, {snr_vals.max():.1f}]"
            )
        if len(c50_vals) > 0:
            logger.info(
                f"  C50 : mean={c50_vals.mean():.1f} dB  std={c50_vals.std():.1f} dB"
            )
    else:
        logger.info("No segments processed.")

    if n_missing > 0:
        logger.warning(f"{n_missing}/{total} files had no SNR npz")

    wall = time.time() - t0
    logger.info("─" * 50)
    logger.info(
        f"Shard {shard_id}: {total - n_errors - n_missing}/{total} ok  "
        f"{n_errors} errors  {n_missing} missing npz"
    )
    logger.info(f"Wall time: {hhmmss(wall)}")

    log_benchmark(
        step="segment_snr",
        dataset=dataset,
        n_files=total,
        wall_seconds=wall,
        total_bytes=0,
        n_workers=1,
        extra={"shard_id": shard_id},
    )


def entrypoint() -> None:
    parser = argparse.ArgumentParser(
        description="Per-VTC-segment SNR & C50 from pre-computed Brouhaha frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m audio_pipeline.pipeline.segment_snr seedlings_10\n"
            "  sbatch slurm/segment_snr.slurm seedlings_10\n"
        ),
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--array_id", type=int, help="SLURM array task ID")
    parser.add_argument("--array_count", type=int, help="Total SLURM array tasks")
    add_sample_argument(parser)

    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    entrypoint()
