"""Typed models and tomlkit-backed I/O for meta.toml and dataset.toml.

The tomlkit document is the source of truth for serialization so that
comments and formatting survive edits; the Pydantic model is a validated
snapshot refreshed after every mutation.
"""

from __future__ import annotations

import abc
import typing as t

import pydantic
import tomlkit
import tomlkit.items

from .errors import StethoError

if t.TYPE_CHECKING:
    from pathlib import Path

DatasetType = t.Literal["audio", "video"]
LogLevel = t.Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
ImportMode = t.Literal["copy", "symlink", "reference", "huggingface"]

_META_TEMPLATE = """\
# stetho workspace configuration.
# Global settings for this workspace; per-dataset settings live in
# datasets/<name>/dataset.toml.

# Worker processes for parallel steps (>= 1).
workers = 1

# Console log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
log_level = "INFO"

# Plot definitions, filled by the `plot` command.
[plots]

# Slurm execution settings (reserved; local execution for now).
[slurm]

# Dataset source registry (reserved).
[sources]

# Per-module configuration. Each [modules.<name>] block is opaque to
# stetho core and validated by the owning module.
[modules]

# Command history is appended automatically as [[command_log]] entries.
"""


class ConfigError(StethoError):
    """A config file failed schema validation."""

    def __init__(self, path: Path, error: pydantic.ValidationError) -> None:
        self.path = path
        self.error = error
        super().__init__(f"invalid config at {path}:\n{error}")


class CommandLogEntry(pydantic.BaseModel):
    """One recorded CLI invocation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    timestamp: str
    argv: list[str]


class MetaConfig(pydantic.BaseModel):
    """Validated view of meta.toml (global workspace settings).

    Module configuration is intentionally opaque here: each entry of
    ``modules`` is passed through untouched for the owning module to validate.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    workers: int = pydantic.Field(default=1, ge=1)
    log_level: LogLevel = "INFO"

    plots: dict[str, t.Any] = pydantic.Field(default_factory=dict)
    slurm: dict[str, t.Any] = pydantic.Field(default_factory=dict)
    sources: dict[str, t.Any] = pydantic.Field(default_factory=dict)
    modules: dict[str, dict[str, t.Any]] = pydantic.Field(default_factory=dict)
    command_log: list[CommandLogEntry] = pydantic.Field(default_factory=list)


class DatasetConfig(pydantic.BaseModel):
    """Validated view of a dataset.toml."""

    model_config = pydantic.ConfigDict(extra="forbid")

    type: DatasetType
    source: str
    mode: ImportMode
    runs: dict[str, str] = pydantic.Field(default_factory=dict)


def _document_from_model(model: pydantic.BaseModel) -> tomlkit.TOMLDocument:
    """Render a validated model to a fresh TOML document."""
    doc = tomlkit.document()
    for key, value in model.model_dump(mode="json").items():
        doc[key] = value
    return doc


class TomlFile[ModelT](abc.ABC):
    """A tomlkit document paired with its validated Pydantic snapshot."""

    def __init__(self, path: Path, doc: tomlkit.TOMLDocument) -> None:
        self._path = path
        self._doc = doc
        self.config: ModelT = self._validate()

    @abc.abstractmethod
    def _parse(self, data: dict[str, t.Any]) -> ModelT:
        """Validate a raw mapping into the concrete model."""

    def _validate(self) -> ModelT:
        """Validate the live document into a model snapshot.

        Routed through the concrete subclass's `_parse` so the call lands on a
        real model class, not a TypeVar-bound `type[ModelT]`.

        Raises:
            ConfigError: If the document does not satisfy the schema.

        """
        try:
            return self._parse(self._doc.unwrap())
        except pydantic.ValidationError as err:
            raise ConfigError(self._path, err) from err

    def save(self) -> None:
        """Write the document back to disk, preserving comments."""
        self._path.write_text(tomlkit.dumps(self._doc))


class MetaFile(TomlFile[MetaConfig]):
    """meta.toml, backed by its tomlkit document."""

    def _parse(self, data: dict[str, t.Any]) -> MetaConfig:
        return MetaConfig.model_validate(data)

    @classmethod
    def load(cls, path: Path) -> MetaFile:
        """Load and validate an existing meta.toml.

        Raises:
            FileNotFoundError: If the file does not exist.
            ConfigError: If the file fails validation.

        """
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(path, tomlkit.parse(path.read_text()))

    @classmethod
    def create(cls, path: Path, config: MetaConfig) -> MetaFile:
        """Build a new meta.toml document from a validated model."""
        return cls(path, _document_from_model(config))

    def log_command(self, timestamp: str, argv: list[str]) -> None:
        """Append a command-log entry as a [[command_log]] table."""
        entry = tomlkit.table()
        entry["timestamp"] = timestamp
        entry["argv"] = argv
        log = self._doc.get("command_log")
        if not isinstance(log, tomlkit.items.AoT):
            log = tomlkit.aot()
            self._doc["command_log"] = log
        log.append(entry)
        self.config = self._validate()

    @classmethod
    def create_default(cls, path: Path) -> MetaFile:
        """Build a new meta.toml from the commented starter template."""
        return cls(path, tomlkit.parse(_META_TEMPLATE))


class DatasetFile(TomlFile[DatasetConfig]):
    """dataset.toml, backed by its tomlkit document."""

    def _parse(self, data: dict[str, t.Any]) -> DatasetConfig:
        return DatasetConfig.model_validate(data)

    @classmethod
    def load(cls, path: Path) -> DatasetFile:
        """Load and validate an existing dataset.toml.

        Raises:
            FileNotFoundError: If the file does not exist.
            ConfigError: If the file fails validation.

        """
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(path, tomlkit.parse(path.read_text()))

    @classmethod
    def create(cls, path: Path, config: DatasetConfig) -> DatasetFile:
        """Build a new dataset.toml document from a validated model."""
        return cls(path, _document_from_model(config))

    def set_kept_run(self, process: str, run_id: str) -> None:
        """Record the kept run_id for a process under the [runs] table."""
        runs = self._doc.get("runs")
        if not isinstance(runs, tomlkit.items.Table):
            runs = tomlkit.table()
            self._doc["runs"] = runs
        runs[process] = run_id
        self.config = self._validate()
