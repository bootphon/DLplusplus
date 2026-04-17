#!/usr/bin/env python3
"""All external models used by the pipeline."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# ── Constants ─────────────────────────────────────────────────────────
MODELS = {
    "brouhaha": {
        "repo_id": "ylacombe/brouhaha-best",
        "rev": "99bf97b13fd4dda2434a6f7c50855933076f2937",
        "files": ["best.ckpt"],
    },
    "vtc": {
        "repo_id": "coml/VTC-2",
        "rev": "6b1a95508302edc14c50f670cd9a30d66fa4f88a",
        "files": ["model/best.ckpt", "model/config.toml"],
    },
}

MODEL_ROOT = Path(
    os.environ["MODEL_ROOT"]
    if "MODEL_ROOT" in os.environ
    else Path.home() / ".cache/dlpluplus"
)


def fetch_model(name: str, model_meta: dict) -> None:
    """Download all pinned files from the HuggingFace repository.

    Uses the HuggingFace Hub client for caching, LFS support,
    and resumable downloads. Files are copied to OUTPUT_DIR.

    Raises:
        huggingface_hub.utils.HfHubHTTPError: If any download fails.
    """

    target_folder = MODEL_ROOT / name
    commit_file = target_folder / "commit"
    if commit_file.exists():
        commit = commit_file.read_text().strip()
        if commit == model_meta["rev"]:
            print(
                f"Model ({model_meta['repo_id']}@{model_meta['rev'][:8]}) is already downloaded !"
            )
            return

    print(
        f"\nFetching HuggingFace assets ({model_meta['repo_id']}@{model_meta['rev'][:8]}):"
    )
    for filename in model_meta["files"]:
        cached = hf_hub_download(
            repo_id=model_meta["repo_id"],
            filename=filename,
            revision=model_meta["rev"],
        )
        dest = MODEL_ROOT / name / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(cached).read_bytes())
    commit_file.write_text(f"{model_meta['rev']}")
    print(f"  ✓ {dest}")


if __name__ == "__main__":
    for _name, _model in MODELS.items():
        fetch_model(_name, _model)
