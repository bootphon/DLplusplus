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
# registry/<name>/dataset.toml.

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

# Per-module configuration. Each [modules.<name>] block below lists a process's
# parameters, commented at their defaults — uncomment to override. Blocks are
# opaque to stetho core and validated by the owning module.
"""


class ConfigError(StethoError):
    """A config file failed schema validation."""

    def __init__(self, path: Path, error: pydantic.ValidationError) -> None:
        self.path = path
        self.error = error
        super().__init__(f"invalid config at {path}:\n{error}")


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


class ConvertedLayer(pydantic.BaseModel):
    """The active converted-audio layer produced by `audio_tools.convert`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    format: str
    sample_rate: int | None = None
    mono: bool = False


class DatasetConfig(pydantic.BaseModel):
    """Validated view of a dataset.toml."""

    model_config = pydantic.ConfigDict(extra="forbid")

    type: DatasetType
    source: str
    mode: ImportMode
    runs: dict[str, str] = pydantic.Field(default_factory=dict)
    converted: ConvertedLayer | None = None


def _document_from_model(model: pydantic.BaseModel) -> tomlkit.TOMLDocument:
    """Render a validated model to a fresh TOML document."""
    doc = tomlkit.document()
    for key, value in model.model_dump(mode="json", exclude_none=True).items():
        doc[key] = value
    return doc


def _render_module_block(name: str, params: type[pydantic.BaseModel]) -> str:
    """Render a commented `[modules.<name>]` block listing a model's fields at defaults."""
    lines = [f"[modules.{name}]"]
    for field_name, field in params.model_fields.items():
        if field.description:
            lines.append(f"# {field.description}")
        default = None if field.is_required() else field.default
        value = "" if default is None else tomlkit.item(default).as_string()
        suffix = f" {value}" if value else ""
        lines.append(f"# {field_name} ={suffix}")
    return "\n".join(lines) + "\n"


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

    def set_module_config(self, name: str, values: dict[str, t.Any]) -> None:
        """Replace the [modules.<name>] block with `values`."""
        modules = self._doc.get("modules")
        if not isinstance(modules, tomlkit.items.Table):
            modules = tomlkit.table(is_super_table=True)
            self._doc["modules"] = modules
        block = tomlkit.table()
        for key, value in values.items():
            block[key] = value
        modules[name] = block
        self.config = self._validate()

    @classmethod
    def create_default(
        cls,
        path: Path,
        module_params: t.Mapping[str, type[pydantic.BaseModel]] | None = None,
    ) -> MetaFile:
        """Build a new meta.toml from the commented starter template.

        When `module_params` is given, append a commented `[modules.<name>]` block
        for each process, listing its parameters at their default values.
        """
        text = _META_TEMPLATE
        if module_params:
            blocks = "\n".join(_render_module_block(n, m) for n, m in sorted(module_params.items()))
            text = f"{text}\n{blocks}"
        return cls(path, tomlkit.parse(text))


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

    def set_converted(self, layer: ConvertedLayer) -> None:
        """Record the active converted-audio layer under the [converted] table."""
        table = tomlkit.table()
        table["format"] = layer.format
        if layer.sample_rate is not None:
            table["sample_rate"] = layer.sample_rate
        table["mono"] = layer.mono
        self._doc["converted"] = table
        self.config = self._validate()
