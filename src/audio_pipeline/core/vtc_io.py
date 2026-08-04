"""Read VTC pipeline outputs from disk."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_vtc_segments(
    output_dir: Path,
    kind: str = "merged",
    uid: str | None = None,
) -> pl.DataFrame:
    """Load VTC segment shards from output_dir/vtc_{kind}/.

    Args:
        kind: "merged" or "raw".
        uid:  if given, filter to this uid only.

    Raises:
        FileNotFoundError: if the directory does not exist or has no shards.
    """
    vtc_dir = output_dir / f"vtc_{kind}"
    if not vtc_dir.exists():
        raise FileNotFoundError(f"VTC {kind} directory not found: {vtc_dir}")
    files = sorted(vtc_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {vtc_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical")
    if uid is not None:
        df = df.filter(pl.col("uid") == uid)
    return df


def load_vtc_meta(output_dir: Path) -> pl.DataFrame:
    """Load all VTC per-file metadata shards from output_dir/vtc_meta/.

    Raises:
        FileNotFoundError: if the directory does not exist or has no shards.
    """
    meta_dir = output_dir / "vtc_meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"VTC meta directory not found: {meta_dir}")
    files = sorted(meta_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {meta_dir}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")
