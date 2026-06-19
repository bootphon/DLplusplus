#!/usr/bin/env python3
"""Download all external model assets used by the pipeline.

Two download backends:
  - HuggingFace Hub  : model weights and configs  (pinned by commit rev)
  - GitHub raw       : threshold TOML files        (not yet on HF)

Set MODEL_ROOT to override the default cache location:
    MODEL_ROOT=/my/path uv run python -m audio_pipeline.download_models
"""

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

# ── Cache root ────────────────────────────────────────────────────────
MODEL_ROOT = Path(
    os.environ["MODEL_ROOT"]
    if "MODEL_ROOT" in os.environ
    else Path.home() / ".cache/dlplusplus"
)

# ── HuggingFace models (weights + configs, pinned by commit rev) ──────
HF_MODELS: dict[str, dict] = {
    "brouhaha": {
        "repo_id": "ylacombe/brouhaha-best",
        "rev": "99bf97b13fd4dda2434a6f7c50855933076f2937",
        "files": ["best.ckpt"],
    },
    "vtc": {
        "repo_id": "coml/VTC-2",
        "rev": "6b1a95508302edc14c50f670cd9a30d66fa4f88a",
        "files": [
            "model/best.ckpt",
            "model/config.toml",
        ],
    },
}

GITHUB_FILES: dict[str, dict] = {
    "vtc": {
        "base_url": "https://raw.githubusercontent.com/LAAC-LSCP/VTC/main",
        "files": [
            "thresholds/f1.toml",
            "thresholds/hp.toml",
        ],
    },
}


# ── HuggingFace downloader ────────────────────────────────────────────


def fetch_hf_model(name: str, meta: dict) -> None:
    """Download all pinned files for one HF model entry.

    Skips the whole model if the stored commit hash already matches.
    Raises huggingface_hub.utils.HfHubHTTPError on download failure.
    """
    target = MODEL_ROOT / name
    commit_file = target / "commit"

    if commit_file.exists() and commit_file.read_text().strip() == meta["rev"]:
        print(
            f"[hf]  {meta['repo_id']}@{meta['rev'][:8]}  already up to date, skipping."
        )
        return

    print(f"\n[hf]  Fetching {meta['repo_id']}@{meta['rev'][:8]} :")
    for filename in meta["files"]:
        cached = hf_hub_download(
            repo_id=meta["repo_id"],
            filename=filename,
            revision=meta["rev"],
        )
        dest = target / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(cached).read_bytes())
        print(f"        ✓  {dest.relative_to(MODEL_ROOT)}")

    # Write commit marker only after all files succeed
    commit_file.write_text(meta["rev"])


# ── GitHub raw downloader ─────────────────────────────────────────────


def fetch_github_files(name: str, meta: dict) -> None:
    """Download raw files from a GitHub repository.

    Skips individual files that already exist on disk.
    Raises urllib.error.URLError on network failure.
    """
    base_url: str = meta["base_url"]

    print(f"\n[gh]  Fetching {name} thresholds from GitHub:")
    for filename in meta["files"]:
        dest = MODEL_ROOT / name / filename
        if dest.exists():
            print(
                f"        –  {dest.relative_to(MODEL_ROOT)}  already exists, skipping."
            )
            continue
        url = f"{base_url.rstrip('/')}/{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"        ✓  {dest.relative_to(MODEL_ROOT)}")


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force-download",
        action="store_true",
        help="Clear cache, and force re-download !!",
    )

    argv = parser.parse_args()

    if argv.force_download:
        shutil.rmtree(MODEL_ROOT)

    print(f"MODEL_ROOT: {MODEL_ROOT}\n")

    for name, meta in HF_MODELS.items():
        fetch_hf_model(name, meta)

    for name, meta in GITHUB_FILES.items():
        fetch_github_files(name, meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
