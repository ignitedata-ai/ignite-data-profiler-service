"""Database overview request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from ignite_data_connectors import (
    BigQueryConfig,
    DatabricksConfig,
    MySQLConfig,
    PostgresConfig,
    RedshiftConfig,
    SnowflakeConfig,
)
from pydantic import BaseModel, Field

# ── Shared config ──────────────────────────────────────────────────────────────


class OverviewConfig(BaseModel):
    """Configuration for the overview endpoint."""

    include_schemas: list[str] | None = Field(
        default=None,
        description="Schemas to include; None means all non-system schemas",
    )
    exclude_schemas: list[str] = Field(
        default=["pg_catalog", "information_schema", "pg_toast"],
        description="Schemas to always exclude",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Overall timeout in seconds",
    )


# ── Per-datasource request types ───────────────────────────────────────────────


class PostgresOverviewRequest(BaseModel):
    datasource_type: Literal["postgres"] = "postgres"
    connection: PostgresConfig
    config: OverviewConfig = Field(default_factory=OverviewConfig)


class MySQLOverviewRequest(BaseModel):
    datasource_type: Literal["mysql"] = "mysql"
    connection: MySQLConfig
    config: OverviewConfig = Field(
        default_factory=lambda: OverviewConfig(
            exclude_schemas=["information_schema", "mysql", "performance_schema", "sys"],
        ),
    )


class SnowflakeOverviewRequest(BaseModel):
    datasource_type: Literal["snowflake"] = "snowflake"
    connection: SnowflakeConfig
    config: OverviewConfig = Field(
        default_factory=lambda: OverviewConfig(exclude_schemas=["INFORMATION_SCHEMA"]),
    )


class DatabricksOverviewRequest(BaseModel):
    datasource_type: Literal["databricks"] = "databricks"
    connection: DatabricksConfig
    config: OverviewConfig = Field(
        default_factory=lambda: OverviewConfig(exclude_schemas=["information_schema"]),
    )


class BigQueryOverviewRequest(BaseModel):
    datasource_type: Literal["bigquery"] = "bigquery"
    connection: BigQueryConfig
    config: OverviewConfig = Field(
        default_factory=lambda: OverviewConfig(exclude_schemas=["INFORMATION_SCHEMA"]),
    )


class RedshiftOverviewRequest(BaseModel):
    datasource_type: Literal["redshift"] = "redshift"
    connection: RedshiftConfig
    config: OverviewConfig = Field(default_factory=OverviewConfig)


OverviewRequest = Annotated[
    PostgresOverviewRequest
    | MySQLOverviewRequest
    | SnowflakeOverviewRequest
    | DatabricksOverviewRequest
    | BigQueryOverviewRequest
    | RedshiftOverviewRequest,
    Field(discriminator="datasource_type"),
]


# ── Response ───────────────────────────────────────────────────────────────────


class SchemaOverview(BaseModel):
    """Counts for a single schema/dataset."""

    schema_name: str
    table_count: int
    view_count: int
    index_count: int
    relationship_count: int
    column_count: int


class DatabaseOverview(BaseModel):
    """Top-level overview response."""

    database_name: str
    database_version: str | None = None
    total_schemas: int
    total_tables: int
    total_views: int
    total_indexes: int
    total_relationships: int
    total_columns: int
    schemas: list[SchemaOverview]
    profiled_at: datetime
    duration_ms: int = Field(description="Wall-clock time in milliseconds")
