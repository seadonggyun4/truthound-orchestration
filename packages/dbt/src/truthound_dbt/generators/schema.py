"""dbt Schema YAML Generator.

This module provides generation of dbt schema.yml files from model
definitions and data quality rules.

Example:
    >>> from truthound_dbt.generators import SchemaGenerator
    >>>
    >>> generator = SchemaGenerator()
    >>> result = generator.generate_schema(
    ...     models=[
    ...         {
    ...             "name": "stg_users",
    ...             "description": "Staged user data",
    ...             "columns": [
    ...                 {"name": "id", "tests": ["not_null", "unique"]},
    ...                 {"name": "email", "tests": ["not_null"]},
    ...             ],
    ...         }
    ...     ],
    ... )
    >>> print(result.yaml_content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class SchemaGenerationError(Exception):
    """Base exception for schema generation errors."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class SchemaGeneratorConfig:
    """Configuration for schema generator.

    Attributes:
        include_descriptions: Include model/column descriptions.
        include_tests: Include test configurations.
        include_meta: Include meta information.
        include_docs: Include docs block references.
        indent_size: YAML indentation size.
        sort_columns: Sort columns alphabetically.
        version: Schema version (default 2).
    """

    include_descriptions: bool = True
    include_tests: bool = True
    include_meta: bool = True
    include_docs: bool = True
    indent_size: int = 2
    sort_columns: bool = False
    version: int = 2

    def with_descriptions(self, include: bool = True) -> SchemaGeneratorConfig:
        """Return config with description setting."""
        return SchemaGeneratorConfig(
            include_descriptions=include,
            include_tests=self.include_tests,
            include_meta=self.include_meta,
            include_docs=self.include_docs,
            indent_size=self.indent_size,
            sort_columns=self.sort_columns,
            version=self.version,
        )

    def with_tests(self, include: bool = True) -> SchemaGeneratorConfig:
        """Return config with tests setting."""
        return SchemaGeneratorConfig(
            include_descriptions=self.include_descriptions,
            include_tests=include,
            include_meta=self.include_meta,
            include_docs=self.include_docs,
            indent_size=self.indent_size,
            sort_columns=self.sort_columns,
            version=self.version,
        )


DEFAULT_SCHEMA_GENERATOR_CONFIG = SchemaGeneratorConfig()


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Schema definition for a column.

    Attributes:
        name: Column name.
        description: Column description.
        data_type: Column data type.
        tests: List of tests for the column.
        meta: Meta information.
        quote: Whether to quote the column name.
        tags: Column tags.
    """

    name: str
    description: str | None = None
    data_type: str | None = None
    tests: tuple[Any, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    quote: bool = False
    tags: tuple[str, ...] = ()

    def to_dict(self, config: SchemaGeneratorConfig) -> dict[str, Any]:
        """Convert to dictionary for YAML output."""
        result: dict[str, Any] = {"name": self.name}

        if config.include_descriptions and self.description:
            result["description"] = self.description

        if self.data_type:
            result["data_type"] = self.data_type

        if config.include_tests and self.tests:
            result["tests"] = list(self.tests)

        if config.include_meta and self.meta:
            result["meta"] = self.meta

        if self.quote:
            result["quote"] = True

        if self.tags:
            result["tags"] = list(self.tags)

        return result


@dataclass(frozen=True, slots=True)
class ModelSchema:
    """Schema definition for a model.

    Attributes:
        name: Model name.
        description: Model description.
        columns: List of column schemas.
        tests: Model-level tests.
        meta: Meta information.
        config: Model configuration.
        docs: Documentation block reference.
        tags: Model tags.
    """

    name: str
    description: str | None = None
    columns: tuple[ColumnSchema, ...] = ()
    tests: tuple[Any, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    docs: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def to_dict(self, config: SchemaGeneratorConfig) -> dict[str, Any]:
        """Convert to dictionary for YAML output."""
        result: dict[str, Any] = {"name": self.name}

        if config.include_descriptions and self.description:
            result["description"] = self.description

        if config.include_meta and self.meta:
            result["meta"] = self.meta

        if self.config:
            result["config"] = self.config

        if config.include_docs and self.docs:
            result["docs"] = self.docs

        if self.tags:
            result["tags"] = list(self.tags)

        if config.include_tests and self.tests:
            result["tests"] = list(self.tests)

        if self.columns:
            columns = list(self.columns)
            if config.sort_columns:
                columns = sorted(columns, key=lambda c: c.name)
            result["columns"] = [col.to_dict(config) for col in columns]

        return result


@dataclass(frozen=True, slots=True)
class SchemaYAML:
    """Result of schema generation.

    Attributes:
        yaml_content: The YAML content as string.
        yaml_dict: The YAML content as dictionary.
        models: List of model schemas.
        sources: List of source schemas.
        generated_at: Timestamp of generation.
    """

    yaml_content: str
    yaml_dict: dict[str, Any]
    models: tuple[ModelSchema, ...]
    sources: tuple[dict[str, Any], ...] = ()
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "yaml_content": self.yaml_content,
            "model_count": len(self.models),
            "source_count": len(self.sources),
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# SchemaGenerator
# =============================================================================


class SchemaGenerator:
    """Generates dbt schema.yml files.

    This generator creates complete schema.yml configurations from
    model definitions and data quality rules.

    Example:
        >>> generator = SchemaGenerator()
        >>> result = generator.generate_schema(
        ...     models=[
        ...         ModelSchema(
        ...             name="stg_users",
        ...             description="Staged user data",
        ...             columns=(
        ...                 ColumnSchema("id", tests=("not_null", "unique")),
        ...                 ColumnSchema("email", tests=("not_null",)),
        ...             ),
        ...         )
        ...     ],
        ... )
        >>> print(result.yaml_content)
    """

    def __init__(self, config: SchemaGeneratorConfig | None = None) -> None:
        """Initialize schema generator.

        Args:
            config: Generator configuration.
        """
        self._config = config or DEFAULT_SCHEMA_GENERATOR_CONFIG

    @property
    def config(self) -> SchemaGeneratorConfig:
        """Return the generator configuration."""
        return self._config

    def generate_schema(
        self,
        models: Sequence[ModelSchema | dict[str, Any]],
        sources: Sequence[dict[str, Any]] | None = None,
    ) -> SchemaYAML:
        """Generate a complete schema.yml configuration.

        Args:
            models: List of model schemas or dictionaries.
            sources: Optional list of source configurations.

        Returns:
            SchemaYAML containing the complete configuration.
        """
        # Convert dicts to ModelSchema
        model_schemas: list[ModelSchema] = []
        for model in models:
            if isinstance(model, ModelSchema):
                model_schemas.append(model)
            else:
                model_schemas.append(self._dict_to_model_schema(model))

        # Build YAML dict
        yaml_dict = self._build_yaml_dict(model_schemas, sources or [])

        # Convert to YAML string
        yaml_content = self._dict_to_yaml(yaml_dict)

        return SchemaYAML(
            yaml_content=yaml_content,
            yaml_dict=yaml_dict,
            models=tuple(model_schemas),
            sources=tuple(sources or []),
        )

    def generate_model_schema(
        self,
        name: str,
        description: str | None = None,
        columns: Sequence[ColumnSchema | dict[str, Any]] | None = None,
        tests: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> ModelSchema:
        """Generate a single model schema.

        Args:
            name: Model name.
            description: Model description.
            columns: Column schemas or dictionaries.
            tests: Model-level tests.
            **kwargs: Additional model properties.

        Returns:
            ModelSchema instance.
        """
        column_schemas: list[ColumnSchema] = []
        if columns:
            for col in columns:
                if isinstance(col, ColumnSchema):
                    column_schemas.append(col)
                else:
                    column_schemas.append(self._dict_to_column_schema(col))

        return ModelSchema(
            name=name,
            description=description,
            columns=tuple(column_schemas),
            tests=tuple(tests or []),
            meta=kwargs.get("meta", {}),
            config=kwargs.get("config", {}),
            docs=kwargs.get("docs", {}),
            tags=tuple(kwargs.get("tags", [])),
        )

    def generate_column_schema(
        self,
        name: str,
        description: str | None = None,
        tests: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> ColumnSchema:
        """Generate a single column schema.

        Args:
            name: Column name.
            description: Column description.
            tests: Column tests.
            **kwargs: Additional column properties.

        Returns:
            ColumnSchema instance.
        """
        return ColumnSchema(
            name=name,
            description=description,
            data_type=kwargs.get("data_type"),
            tests=tuple(tests or []),
            meta=kwargs.get("meta", {}),
            quote=kwargs.get("quote", False),
            tags=tuple(kwargs.get("tags", [])),
        )

    def merge_schemas(
        self,
        base: SchemaYAML,
        overlay: SchemaYAML,
    ) -> SchemaYAML:
        """Merge two schema configurations.

        The overlay schema takes precedence for conflicting values.

        Args:
            base: Base schema configuration.
            overlay: Overlay schema configuration.

        Returns:
            Merged SchemaYAML.
        """
        # Build model name -> model mapping
        base_models = {m.name: m for m in base.models}
        overlay_models = {m.name: m for m in overlay.models}

        merged_models: list[ModelSchema] = []

        # Process all models
        all_names = set(base_models.keys()) | set(overlay_models.keys())
        for name in sorted(all_names):
            base_model = base_models.get(name)
            overlay_model = overlay_models.get(name)

            if overlay_model is None:
                merged_models.append(base_model)  # type: ignore
            elif base_model is None:
                merged_models.append(overlay_model)
            else:
                merged_models.append(self._merge_models(base_model, overlay_model))

        # Merge sources
        merged_sources = list(base.sources) + list(overlay.sources)

        return self.generate_schema(merged_models, merged_sources)

    def _merge_models(
        self,
        base: ModelSchema,
        overlay: ModelSchema,
    ) -> ModelSchema:
        """Merge two model schemas."""
        # Merge columns
        base_cols = {c.name: c for c in base.columns}
        overlay_cols = {c.name: c for c in overlay.columns}

        merged_cols: list[ColumnSchema] = []
        all_col_names = set(base_cols.keys()) | set(overlay_cols.keys())

        for col_name in sorted(all_col_names):
            base_col = base_cols.get(col_name)
            overlay_col = overlay_cols.get(col_name)

            if overlay_col is None:
                merged_cols.append(base_col)  # type: ignore
            elif base_col is None:
                merged_cols.append(overlay_col)
            else:
                merged_cols.append(self._merge_columns(base_col, overlay_col))

        return ModelSchema(
            name=overlay.name,
            description=overlay.description or base.description,
            columns=tuple(merged_cols),
            tests=tuple(set(base.tests) | set(overlay.tests)),
            meta={**base.meta, **overlay.meta},
            config={**base.config, **overlay.config},
            docs={**base.docs, **overlay.docs},
            tags=tuple(set(base.tags) | set(overlay.tags)),
        )

    def _merge_columns(
        self,
        base: ColumnSchema,
        overlay: ColumnSchema,
    ) -> ColumnSchema:
        """Merge two column schemas."""
        return ColumnSchema(
            name=overlay.name,
            description=overlay.description or base.description,
            data_type=overlay.data_type or base.data_type,
            tests=tuple(set(base.tests) | set(overlay.tests)),
            meta={**base.meta, **overlay.meta},
            quote=overlay.quote or base.quote,
            tags=tuple(set(base.tags) | set(overlay.tags)),
        )

    def _dict_to_model_schema(self, data: dict[str, Any]) -> ModelSchema:
        """Convert dictionary to ModelSchema."""
        columns: list[ColumnSchema] = []
        for col in data.get("columns", []):
            if isinstance(col, ColumnSchema):
                columns.append(col)
            else:
                columns.append(self._dict_to_column_schema(col))

        return ModelSchema(
            name=data["name"],
            description=data.get("description"),
            columns=tuple(columns),
            tests=tuple(data.get("tests", [])),
            meta=data.get("meta", {}),
            config=data.get("config", {}),
            docs=data.get("docs", {}),
            tags=tuple(data.get("tags", [])),
        )

    def _dict_to_column_schema(self, data: dict[str, Any]) -> ColumnSchema:
        """Convert dictionary to ColumnSchema."""
        return ColumnSchema(
            name=data["name"],
            description=data.get("description"),
            data_type=data.get("data_type"),
            tests=tuple(data.get("tests", [])),
            meta=data.get("meta", {}),
            quote=data.get("quote", False),
            tags=tuple(data.get("tags", [])),
        )

    def _build_yaml_dict(
        self,
        models: Sequence[ModelSchema],
        sources: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the YAML dictionary structure."""
        result: dict[str, Any] = {"version": self._config.version}

        if models:
            result["models"] = [m.to_dict(self._config) for m in models]

        if sources:
            result["sources"] = list(sources)

        return result

    def _dict_to_yaml(self, data: dict[str, Any]) -> str:
        """Convert dictionary to YAML string.

        A simple YAML serializer to avoid external dependencies.

        Args:
            data: Dictionary to convert.

        Returns:
            YAML formatted string.
        """
        lines: list[str] = []
        self._write_yaml_value(lines, data, 0)
        return "\n".join(lines)

    def _write_yaml_value(
        self,
        lines: list[str],
        value: Any,
        indent: int,
    ) -> None:
        """Recursively write a value as YAML."""
        prefix = " " * (indent * self._config.indent_size)

        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)) and v:
                    lines.append(f"{prefix}{k}:")
                    self._write_yaml_value(lines, v, indent + 1)
                else:
                    self._write_yaml_inline(lines, k, v, indent)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    first_key = next(iter(item.keys()), None)
                    if first_key is not None:
                        first_val = item[first_key]
                        if isinstance(first_val, (dict, list)) and first_val:
                            lines.append(f"{prefix}- {first_key}:")
                            self._write_yaml_value(lines, first_val, indent + 2)
                        else:
                            self._write_yaml_inline(lines, f"- {first_key}", first_val, indent)
                        for k, v in list(item.items())[1:]:
                            extra_prefix = " " * ((indent + 1) * self._config.indent_size)
                            if isinstance(v, (dict, list)) and v:
                                lines.append(f"{extra_prefix}{k}:")
                                self._write_yaml_value(lines, v, indent + 2)
                            else:
                                self._write_yaml_inline(lines, k, v, indent + 1)
                elif isinstance(item, str):
                    lines.append(f"{prefix}- {item}")
                else:
                    lines.append(f"{prefix}- {self._format_value(item)}")
        else:
            lines.append(f"{prefix}{self._format_value(value)}")

    def _write_yaml_inline(
        self,
        lines: list[str],
        key: str,
        value: Any,
        indent: int,
    ) -> None:
        """Write a simple key-value pair."""
        prefix = " " * (indent * self._config.indent_size)
        lines.append(f"{prefix}{key}: {self._format_value(value)}")

    def _format_value(self, value: Any) -> str:
        """Format a simple value for YAML."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            if any(c in value for c in ":#{}[]|>*&!%@\\"):
                return f'"{value}"'
            return value
        if isinstance(value, list):
            return "[" + ", ".join(self._format_value(v) for v in value) + "]"
        return str(value)
