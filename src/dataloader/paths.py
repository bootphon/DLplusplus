"""Project-level directory roots.

Single source of truth for the four top-level directories the pipeline
writes to. Each root is configurable via an environment variable so the
same code can run against different on-disk layouts without edits:

    DLPP_MANIFESTS_DIR   where manifests/{dataset}.<ext> live
    DLPP_OUTPUT_DIR      where output/{dataset}/ subtrees are written
    DLPP_FIGURES_DIR     where figures/{dataset}/ subtrees are written
    DLPP_LOGS_DIR        where SLURM / benchmark logs are written

Unset variables fall back to the current working directory, matching the
repo's original behaviour.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectPaths(BaseSettings):
    """Top-level directory roots for the pipeline.

    Instantiated once at import time as :data:`PROJECT_PATHS`. Tests or
    ad-hoc scripts can build their own instance with explicit overrides.
    """

    model_config = SettingsConfigDict(
        env_prefix="DLPP_",
        env_file=None,
        extra="ignore",
        frozen=True,
    )

    workspace: Path = Field(default_factory=Path.cwd)
    manifests_dir: Path
    output_dir: Path
    figures_dir: Path
    logs_dir: Path

    @model_validator(mode="before")
    @classmethod
    def _resolve_roots(cls, data: t.Any) -> t.Any:
        if not isinstance(data, dict):
            return data

        workspace_raw = data.get("workspace")
        root = (
            Path(workspace_raw).expanduser().resolve()
            if workspace_raw is not None
            else Path.cwd().resolve()
        )
        data["workspace"] = root

        _subs: dict[str, str] = {
            "manifests_dir": "manifests",
            "output_dir": "output",
            "figures_dir": "figures",
            "logs_dir": "logs",
        }
        for field_name, subdir in _subs.items():
            raw = data.get(field_name)
            if raw is None:
                data[field_name] = root / subdir
            else:
                p = Path(raw)
                data[field_name] = p if p.is_absolute() else root / p

        return data

    def ensure(
        self, *, kinds: t.Iterable[t.Literal["output", "figures", "logs"]] = ()
    ) -> t.Self:
        """Create the named roots on disk if missing."""
        mapping: dict[str, Path] = {
            "output": self.output_dir,
            "figures": self.figures_dir,
            "logs": self.logs_dir,
        }
        for kind in kinds:
            mapping[kind].mkdir(parents=True, exist_ok=True)
        return self


PROJECT_PATHS = ProjectPaths()  # pyright: ignore[reportCallIssue] # This is fine
