"""Append-only log of CLI invocations, stored as JSON Lines at the workspace root."""

from __future__ import annotations

import typing as t

import pydantic

if t.TYPE_CHECKING:
    from pathlib import Path


class CommandLogEntry(pydantic.BaseModel):
    """One recorded CLI invocation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    timestamp: str
    argv: list[str]


def append_command(path: Path, timestamp: str, argv: list[str]) -> None:
    """Append one invocation to the JSONL command log, creating the file if absent."""
    entry = CommandLogEntry(timestamp=timestamp, argv=argv)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")


def read_commands(path: Path) -> list[CommandLogEntry]:
    """Read every recorded invocation oldest-first (empty when the log is absent)."""
    if not path.exists():
        return []
    return [
        CommandLogEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
