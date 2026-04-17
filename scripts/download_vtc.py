from pathlib import Path

from huggingface_hub import hf_hub_download

# ── Pin versions here ──────────────────────────────────────────────────────────

HF_REPO_ID = "coml/VTC-2"
HF_COMMIT = "6b1a95508302edc14c50f670cd9a30d66fa4f88a"
HF_FILES = ["model/best.ckpt", "model/config.json"]

# ──---------────────────────────────────────────────────────────────────────────
MODEL_DIR = Path.home() / ".cache/vtc"


def fetch_hf_files() -> None:
    """Download all pinned files from the HuggingFace repository.

    Uses the HuggingFace Hub client for caching, LFS support,
    and resumable downloads. Files are copied to OUTPUT_DIR.

    Raises:
        huggingface_hub.utils.HfHubHTTPError: If any download fails.
    """
    print(f"\nFetching HuggingFace assets ({HF_REPO_ID}@{HF_COMMIT[:8]}):")
    for filename in HF_FILES:
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            revision=HF_COMMIT,
        )
        dest = MODEL_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(cached).read_bytes())
        print(f"  ✓ {dest}")


if __name__ == "__main__":
    fetch_hf_files()
