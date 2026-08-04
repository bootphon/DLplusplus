"""Per-file metadata row constructors for VTC."""

from __future__ import annotations

import json

import polars as pl

_EMPTY_VTC_META = dict(
    vtc_threshold_preset="",
    vtc_speech_dur=0.0,
    vtc_n_segments=0,
    vtc_label_counts="{}",
    error="",
)


def vtc_error_row(uid: str, error: str) -> dict:
    """Metadata row for a file that errored during VTC inference."""
    return {**_EMPTY_VTC_META, "uid": uid, "error": error}


def vtc_meta_row(uid: str, threshold_preset: str, uid_raw_df: pl.DataFrame) -> dict:
    """Build a metadata row from the raw VTC segments for one uid."""
    speech_dur = float(uid_raw_df["duration"].sum()) if not uid_raw_df.is_empty() else 0.0
    label_counts: dict[str, int] = {}
    if not uid_raw_df.is_empty():
        for row in uid_raw_df.group_by("label").agg(pl.len().alias("n")).to_dicts():
            label_counts[row["label"]] = row["n"]
    return {
        "uid": uid,
        "vtc_threshold_preset": threshold_preset,
        "vtc_speech_dur": round(speech_dur, 3),
        "vtc_n_segments": len(uid_raw_df),
        "vtc_label_counts": json.dumps(label_counts),
        "error": "",
    }
